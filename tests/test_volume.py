"""GPU image and replay-gradient checks for bounded homogeneous transport."""

import os
from pathlib import Path
import subprocess

import drjit as dr
import numpy as np
import pytest

if not dr.has_backend(dr.JitBackend.CUDA):
    pytest.skip("CUDA backend required", allow_module_level=True)

from drjit.cuda.ad import Float
import psdr_jit as psdr
from psdr_jit import HomogeneousMedium, VolumePathTracer
from psdr_jit.web.render import VOLUME_BOUNDS, volume_scene


def renderer(a, s, depth=3, bounds=VOLUME_BOUNDS):
    return VolumePathTracer(HomogeneousMedium(*bounds, a, s), depth)


def rgb(image, size):
    return image.numpy().T.reshape(size, size, 3)


@pytest.mark.parametrize("a,s,bounds", [(-1., 0., VOLUME_BOUNDS), (0., np.inf, VOLUME_BOUNDS),
                                        (0., 1., ((0, 0, 0), (0, 1, 1)))])
def test_invalid_medium(a, s, bounds):
    with pytest.raises(ValueError):
        renderer(a, s, bounds=bounds)


@pytest.mark.parametrize("z0,z1,length", [(1., 2., 1.), (-1., 1.01, 1.),
                                            (2., 4., .5), (3., 4., 0.)])
def test_beer_lambert_image_and_ad(z0, z1, length):
    scene = volume_scene(16, 8)
    a = Float(.7)
    dr.enable_grad(a)
    # Cover half the light. Adjacent rays parallel to the box faces see vacuum.
    bounds = ((0., -1., z0), (2., 1., z1))
    r = renderer(a, 0., bounds=bounds)
    image = r.renderD(scene, seed=3)
    reference = np.zeros((16, 16, 3), np.float32)
    reference[4:12, 6:10] = [4, 2, 1]
    reference[:, :8] *= np.exp(-.7 * length)
    expected_grad = np.zeros_like(reference)
    expected_grad[:, :8] = -length * reference[:, :8]
    np.testing.assert_allclose(rgb(image, 16), reference, atol=3e-6)
    dr.forward(a)
    np.testing.assert_allclose(rgb(dr.grad(image), 16), expected_grad, atol=3e-6)
    dr.set_grad(a, 0)
    dr.backward(dr.sum(r.renderD(scene, seed=11)))
    np.testing.assert_allclose(dr.grad(a)[0], expected_grad.sum(), rtol=3e-6, atol=1e-6)


@pytest.mark.parametrize("coefficient", ["absorption", "scattering"])
def test_replay_matches_rendered_finite_differences(coefficient):
    scene = volume_scene(16, 64)
    a, s = Float(.3), Float(.8)
    p = a if coefficient == "absorption" else s
    dr.enable_grad(p)
    r = renderer(a, s)
    image = r.renderD(scene, seed=23)
    np.testing.assert_allclose(image.numpy(), r.renderC(scene, seed=23).numpy(), atol=1e-6)
    dr.forward(p)
    tangent = dr.grad(image).numpy()
    def shifted(delta):
        integrator = renderer(.3 + delta, .8) if coefficient == "absorption" else renderer(.3, .8 + delta)
        return integrator.renderC(scene, seed=23).numpy()
    fd = (shifted(.002) - shifted(-.002)) / .004
    np.testing.assert_allclose(tangent, fd, rtol=.006, atol=1e-4)
    # A signed, nonuniform RGB adjoint exercises the complete image-to-loss path.
    weights = np.random.default_rng(42).uniform(-1, 1, fd.shape).astype(np.float32)
    dr.set_grad(p, 0)
    dr.backward(dr.sum(r.renderD(scene, seed=23) * dr.cuda.ad.Array3f(weights)))
    np.testing.assert_allclose(dr.grad(p)[0], (fd * weights).sum(), rtol=.001, atol=.001)


def test_multiple_scattering_and_zero_density_limit():
    scene = volume_scene(16, 128)
    s = Float(0)
    dr.enable_grad(s)
    image = renderer(0., s).renderD(scene, seed=7)
    dr.forward(s)
    fd = (renderer(0., .001).renderC(scene, seed=7).numpy() - image.numpy()) / .001
    np.testing.assert_allclose(dr.grad(image).numpy(), fd, rtol=.004, atol=.001)
    one = rgb(renderer(.3, .8, 1).renderC(scene, seed=7), 16)
    three = rgb(renderer(.3, .8, 3).renderC(scene, seed=7), 16)
    assert three[5:11, 3:5].mean() > one[5:11, 3:5].mean() * 1.1 > 0


def quad(x0, x1, y0, y1, z, toward_camera=True):
    mesh = psdr.Mesh()
    indices = [[0, 0], [2, 3], [1, 2]] if toward_camera else [[0, 0], [1, 2], [2, 3]]
    mesh.load_raw(dr.cuda.Array3f([[x0, x1, x1, x0], [y0, y0, y1, y1], [z] * 4]),
                  dr.cuda.Array3i(indices))
    return mesh


@pytest.mark.parametrize("two_sided", [False, True])
def test_back_facing_diffuse_panel(two_sided):
    # The camera sees the back of a panel. Its only light is behind the camera.
    scene = psdr.Scene()
    scene.opts = psdr.RenderOption(8, 8, 64, 0, 0)
    scene.opts.log_level = 0
    scene.add_Sensor(psdr.OrthographicCamera(.01, 1000.))
    scene.add_BSDF(psdr.DiffuseBSDF([.5] * 3), "panel", two_sided)
    scene.add_BSDF(psdr.DiffuseBSDF([0.] * 3), "black")
    scene.add_Mesh(quad(-2, 2, -2, 2, 2, False), "panel")
    scene.add_Mesh(quad(-100, 100, -100, 100, -1, False), "black", psdr.AreaLight([1.] * 3))
    scene.configure()

    # Both the primal BSDF sampler used by volume transport and the AD sampler
    # used by surface replay must send reflected light back towards the camera.
    volume = renderer(0., 0., depth=1, bounds=((10, 10, 10), (11, 11, 11)))
    expected = .5 if two_sided else 0.
    np.testing.assert_allclose(volume.renderC(scene, seed=8).numpy().mean(), expected, atol=.005)
    reflectance = Float(.5)
    dr.enable_grad(reflectance)
    scene.param_map["BSDF[id=panel]"].reflectance.data = dr.cuda.ad.Array3f(reflectance)
    image = psdr.PRBPathTracer(1).renderD(scene, seed=8, parameters=[reflectance])
    dr.forward(reflectance)
    np.testing.assert_allclose(image.numpy().mean(), expected, atol=.005)
    np.testing.assert_allclose(dr.grad(image).numpy().mean(), 2 * expected, atol=.01)


def test_opaque_blocker_extinguishes_scattered_light():
    scene = volume_scene(16, 32)
    r = renderer(.3, .8)
    assert rgb(r.renderC(scene), 16)[5:11, 3:5].mean() > .01
    scene.add_Mesh(quad(-2, 2, -2, 2, 2.25), "black", None)
    scene.configure()
    np.testing.assert_array_equal(r.renderC(scene).numpy(), 0.)


def test_diffuse_reflection_through_absorbing_fog():
    # Camera sees a Lambertian plane. A panel facing away from the camera
    # illuminates it. Both legs cross the same unit-thickness slab.
    size, spp = 8, 8192
    scene = psdr.Scene()
    scene.opts = psdr.RenderOption(size, size, spp, 0, 0)
    scene.opts.log_level = 0
    scene.add_Sensor(psdr.OrthographicCamera(.01, 100.))
    scene.add_BSDF(psdr.DiffuseBSDF([.7] * 3), "mat")
    scene.add_BSDF(psdr.DiffuseBSDF([0.] * 3), "black")
    scene.add_Mesh(quad(-2, 2, -2, 2, 3.), "mat", None)
    scene.add_Mesh(quad(1, 3, -2, 2, .5, False), "black", psdr.AreaLight([1.] * 3))
    scene.configure()
    a = Float(.3)
    dr.enable_grad(a)
    im = renderer(a, 0., depth=1, bounds=((-4, -4, 1), (4, 4, 2))).renderD(scene, seed=91)
    dr.forward(a)

    # Independent midpoint quadrature over the film and the emitting rectangle.
    def centers(lo, hi, n):
        return lo + (np.arange(n) + .5) * (hi - lo) / n
    x, y, lx, ly = np.ix_(centers(-1, 1, 20), centers(-1, 1, 20),
                           centers(1, 3, 48), centers(-2, 2, 48))
    r2 = (x - lx)**2 + (y - ly)**2 + 2.5**2
    length = 1 + np.sqrt(r2) / 2.5
    integrand = (.7 / np.pi) * 8 * 2.5**2 / r2**2 * np.exp(-.3 * length)
    np.testing.assert_allclose(im.numpy().mean(), integrand.mean(), rtol=.025)
    np.testing.assert_allclose(dr.grad(im).numpy().mean(), -(integrand * length).mean(), rtol=.025)


def test_reconfigured_scene_rejects_stale_replay():
    scene = volume_scene(8, 4)
    a = Float(1.)
    dr.enable_grad(a)
    image = renderer(a, .2).renderD(scene)
    scene.configure()
    with pytest.raises(RuntimeError, match="reconfigured"):
        dr.backward(dr.sum(image))


@pytest.mark.skipif(not os.environ.get("MITSUBA_PYTHON"), reason="Set MITSUBA_PYTHON for independent rendering")
def test_mitsuba_reference(tmp_path):
    reference = tmp_path / "mitsuba.npz"
    subprocess.run([os.environ["MITSUBA_PYTHON"], Path(__file__).with_name("mitsuba_volume_reference.py"),
                    reference], check=True, timeout=240)
    with np.load(reference) as data:
        ref, gradients, halo = data["images"], data["gradients"], data["halo"]
    size = halo.shape[0]
    scene = volume_scene(size, 256)
    a, s = Float(.3), Float(.8)
    dr.enable_grad(a, s)
    r = renderer(a, s)
    weights = Float(halo.ravel().astype("float32"))
    images, adjoints = [], []
    for seed in range(len(ref)):
        image = r.renderD(scene, seed=670 + seed)
        images.append(rgb(image, size))
        dr.set_grad(a, 0)
        dr.set_grad(s, 0)
        dr.backward(dr.sum(dr.mean(image, axis=0) * weights) / int(halo.sum()))
        adjoints.append([dr.grad(a)[0], dr.grad(s)[0]])
    images, adjoints = np.stack(images), np.array(adjoints)
    # Compare independent sample streams with a five-standard-error allowance.
    def compare_means(observed, expected):
        se = np.sqrt((observed.var(axis=0, ddof=1) + expected.var(axis=0, ddof=1)) / len(expected))
        assert np.all(np.abs(observed.mean(axis=0) - expected.mean(axis=0)) < 5 * se + 1e-5)
    compare_means(images[:, halo].mean(axis=1), ref[:, halo].mean(axis=1))
    compare_means(adjoints, gradients)
    difference = images.mean(axis=0) - ref.mean(axis=0)
    rms = np.sqrt(np.mean(difference**2))
    assert rms < .012
    print(f"Linear RGB RMSE {rms:.6f}; halo loss gradients PSDR {adjoints.mean(0)}, Mitsuba {gradients.mean(0)}")
