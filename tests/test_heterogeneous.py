"""CUDA image and adjoint regressions for spatially varying density."""
import os
from pathlib import Path
import subprocess

import drjit as dr
import numpy as np
import pytest

if not dr.has_backend(dr.JitBackend.CUDA):
    pytest.skip("CUDA backend required", allow_module_level=True)

from drjit.cuda.ad import Array3f, Float, TensorXf
from psdr_jit import HeterogeneousMedium, HomogeneousMedium, VolumePathTracer
from psdr_jit.web.volume_scenes import FOG_CUBE_BOUNDS, VOLUME_BOUNDS, fog_cube_scene, smoke_density, smoke_scene, volume_scene


def tracer(grid, a=.3, s=.8, depth=3, bounds=VOLUME_BOUNDS):
    return VolumePathTracer(HeterogeneousMedium(*bounds, a, s, grid), depth)


@pytest.mark.parametrize("data", [np.zeros((2, 3)), np.zeros((1, 4, 5)),
                                  np.full((2, 3, 4), -1.), np.full((2, 3, 4), np.nan)])
def test_invalid_grid(data):
    with pytest.raises(ValueError):
        tracer(data.astype(np.float32))


def test_layered_absorption_and_vertex_adjoints():
    scene = volume_scene(8, 4)
    layers = np.array([.2, 1., .4, .8], dtype=np.float32)
    grid = TensorXf(np.broadcast_to(layers[:, None, None], (4, 3, 5)).copy())
    dr.enable_grad(grid)
    image = tracer(grid, a=.65, s=0., depth=0).renderD(scene)
    weights = np.array([1, 2, 2, 1]) / 6
    predicted = np.array([4, 2, 1]) * np.exp(-.65 * (weights @ layers))
    np.testing.assert_allclose(image.numpy()[:, 3 * 8 + 3], predicted, rtol=1e-6)
    dr.backward(dr.sum(image))
    np.testing.assert_allclose(dr.grad(grid).numpy().sum(axis=(1, 2)),
                               -.65 * predicted.sum() * 8 * weights, rtol=2e-6)


def test_unit_density_recovers_homogeneous_scattering():
    scene = fog_cube_scene(12, 16)
    a, s = Float(.4), Float(.7)
    dr.enable_grad(s)
    grid = tracer(np.ones((4, 7, 5), dtype='f'), a, s, bounds=FOG_CUBE_BOUNDS)
    uniform = VolumePathTracer(HomogeneousMedium(*FOG_CUBE_BOUNDS, a, s))
    def sample(r):
        image = r.renderD(scene, seed=49)
        dr.forward(s)
        return image.numpy(), dr.grad(image).numpy()
    for x, y in zip(sample(grid), sample(uniform)):
        np.testing.assert_allclose(x, y, rtol=2e-5, atol=8e-6)


@pytest.mark.parametrize("vacuum", [False, True])
def test_plume_voxel_gradient(vacuum):
    scene = smoke_scene(24, 32)
    density = smoke_density((7, 11, 9))
    if vacuum:
        density.fill(0)
    grid = TensorXf(density)
    direction = np.zeros_like(density)
    direction[2:5, 3:8, 4:7] = .7
    dr.enable_grad(grid)
    integrator = tracer(grid, bounds=FOG_CUBE_BOUNDS)
    image = integrator.renderD(scene, seed=29)
    def shifted(h):
        return tracer(density + h * direction, bounds=FOG_CUBE_BOUNDS).renderC(scene, seed=29).numpy()
    h = .002
    fd = ((-3 * image.numpy() + 4 * shifted(h) - shifted(2*h)) if vacuum else
          (shifted(h) - shifted(-h))) / (2*h)
    dr.set_grad(grid, TensorXf(direction))
    tangent = dr.forward_to(image).numpy()
    assert np.max(np.abs(fd)) > .01
    np.testing.assert_allclose(tangent, fd, rtol=.006, atol=2e-4)
    weights = np.random.default_rng(2).uniform(-1, 1, fd.shape).astype('f')
    dr.set_grad(grid, 0)
    dr.backward(dr.sum(integrator.renderD(scene, seed=29) * Array3f(weights)))
    vjp = np.sum(dr.grad(grid).numpy() * direction)
    np.testing.assert_allclose(vjp, (weights * tangent).sum(), atol=4e-5, rtol=2e-5)


@pytest.mark.parametrize("front_density", [0., 1e-14])
def test_empty_front_layer_with_scattering_behind_it(front_density):
    # A zero or tiny first scattering weight can lead to a lit second event.
    scene = volume_scene(16, 256)
    values = np.ones((7, 3, 4), dtype=np.float32)
    values[:4] = front_density
    direction = np.zeros_like(values)
    direction[1:3] = 1.
    grid = TensorXf(values)
    dr.enable_grad(grid)
    render = tracer(grid, s=1.4)
    image = render.renderD(scene, seed=261)
    weights = Array3f(np.random.default_rng(54).uniform(-1, 1, image.numpy().shape).astype('f'))
    dr.set_grad(grid, TensorXf(direction))
    tangent = dr.forward_to(image)
    expected = dr.sum(dr.dot(weights, tangent))[0]
    dr.set_grad(grid, 0)
    dr.backward(dr.sum(dr.dot(weights, render.renderD(scene, seed=261))))
    adjoint = np.sum(dr.grad(grid).numpy() * direction)
    np.testing.assert_allclose(adjoint, expected, rtol=2e-5, atol=2e-5)
    def loss(step):
        image = tracer(values + step * direction, s=1.4).renderC(scene, seed=261)
        return dr.sum(dr.dot(weights, image))[0]
    h = .005
    fd = (-3 * loss(0) + 4 * loss(h) - loss(2 * h)) / (2 * h)
    np.testing.assert_allclose(adjoint, fd, rtol=.003, atol=.002)


def test_coarse_grid_accumulates_small_late_bounce_derivatives():
    scene = smoke_scene(24, 96)
    values = .25 + smoke_density((3, 4, 5))
    grid = TensorXf(values)
    dr.enable_grad(grid)
    weights = np.random.default_rng(61).uniform(-.2, 1, (3, 24 * 24)).astype('f')

    def render(data):
        return tracer(data, s=.9, depth=5, bounds=FOG_CUBE_BOUNDS)

    dr.backward(dr.sum(Array3f(weights) * render(grid).renderD(scene, seed=158)))
    index = (1, 2, 1)
    adjoint = dr.grad(grid).numpy()[index]
    direction = np.zeros_like(values)
    direction[index] = 1.
    dr.set_grad(grid, TensorXf(direction))
    tangent = dr.forward_to(render(grid).renderD(scene, seed=158)).numpy()
    np.testing.assert_allclose(adjoint, np.sum(tangent * weights, dtype=np.float64), atol=5e-5)
    h = .01
    plus = render(values + h * direction).renderC(scene, seed=158).numpy().astype(np.float64)
    minus = render(values - h * direction).renderC(scene, seed=158).numpy()
    np.testing.assert_allclose(adjoint, np.sum(weights * (plus - minus)) / (2 * h), atol=8e-5)


@pytest.mark.skipif(not os.environ.get("MITSUBA_PYTHON"), reason="Set MITSUBA_PYTHON for independent rendering")
def test_backlit_plume_mitsuba(tmp_path):
    # A thin background fog gives Mitsuba collision samples in every octant.
    density = .15 + smoke_density((13, 19, 17))
    np.save(tmp_path / "density.npy", density)
    reference = tmp_path / "reference.npz"
    subprocess.run([os.environ["MITSUBA_PYTHON"], Path(__file__).with_name("mitsuba_volume_reference.py"),
                    reference, tmp_path / "density.npy"], check=True, timeout=240)
    with np.load(reference) as data:
        reference_image = data["images"].mean(axis=0)
        gradients, halo = data["gradients"], data["halo"]
        reference_grid = data["grid_gradients"]
    size = len(halo)
    scene = volume_scene(size, 256)
    a, s = Float(.3), Float(.8)
    grid = TensorXf(density)
    dr.enable_grad(a, s, grid)
    integrator = tracer(grid, a, s)
    values, images, grid_gradients = [], [], []
    for seed in range(len(gradients)):
        image = integrator.renderD(scene, seed=230 + seed)
        images.append(image.numpy().T.reshape(size, size, 3))
        dr.set_grad(a, 0)
        dr.set_grad(s, 0)
        dr.set_grad(grid, 0)
        dr.backward(dr.sum(dr.mean(image, axis=0) * Float(halo.ravel())) / int(halo.sum()))
        values.append([dr.grad(a)[0], dr.grad(s)[0]])
        grid_gradients.append(dr.grad(grid).numpy())
    values = np.array(values)
    diff = np.mean(images, axis=0) - reference_image
    rmse = np.sqrt(np.mean(diff**2))
    print(f"Backlit plume linear RGB RMSE {rmse:.6g}, halo gradients PSDR {values.mean(0)}, Mitsuba {gradients.mean(0)}")
    assert rmse < .012
    se = np.sqrt((values.var(0, ddof=1) + gradients.var(0, ddof=1)) / len(gradients))
    np.testing.assert_array_less(abs(values.mean(0) - gradients.mean(0)), 5 * se + 1e-5)

    # Pool individual vertex adjoints into spatial octants to resolve the
    # reference null-collision estimator's sampling noise.
    def octants(samples):
        return np.stack([block.sum(axis=(1, 2, 3))
                         for z in np.array_split(samples, 2, axis=1)
                         for y in np.array_split(z, 2, axis=2)
                         for block in np.array_split(y, 2, axis=3)], axis=1)
    ours, theirs = octants(np.stack(grid_gradients)), octants(reference_grid)
    se = np.sqrt((ours.var(0, ddof=1) + theirs.var(0, ddof=1)) / len(theirs))
    np.testing.assert_array_less(abs(ours.mean(0) - theirs.mean(0)), 5 * se + 1e-5)
