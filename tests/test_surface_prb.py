"""Surface replay on an enclosed, indirectly illuminated Cornell box."""
import drjit as dr
import numpy as np
import pytest

if not dr.has_backend(dr.JitBackend.CUDA):
    pytest.skip("CUDA backend required", allow_module_level=True)

from drjit.cuda.ad import Array3f, Float
from psdr_jit import PRBPathTracer, PathTracer
from psdr_jit.web.render import _surface_setup, find_assets, validate
from psdr_jit.web.volume_scenes import volume_scene


@pytest.mark.parametrize("value", [0., .45])
def test_cornell_material_replay(value):
    scene, _ = _surface_setup(validate(dict(scene="cornell", derivative="albedo")), find_assets())
    scene.opts.width = scene.opts.height = 24
    scene.opts.spp = 256
    scene.configure()
    material = scene.param_map["BSDF[id=red]"].reflectance
    tint = Array3f(1., .65, .4)
    tracer = PRBPathTracer(4)
    def primal(v):
        material.data = tint * v
        return tracer.renderC(scene, seed=83).numpy()
    h = .002
    im = primal(value)
    fd = ((primal(value + h) - primal(value - h)) / (2*h) if value > 0 else
          (-3 * im + 4 * primal(h) - primal(2*h)) / (2*h))
    assert fd.max() > .1
    def differentiable():
        parameter = Float(value)
        dr.enable_grad(parameter)
        material.data = tint * parameter
        return tracer.renderD(scene, seed=83, parameters=[parameter]), parameter
    image, parameter = differentiable()
    np.testing.assert_allclose(image.numpy(), im, atol=1e-6)
    dr.forward(parameter)
    tangent = dr.grad(image).numpy()
    np.testing.assert_allclose(tangent, fd, atol=6e-4, rtol=.003)
    image, parameter = differentiable()
    weights = np.random.default_rng(3).uniform(-1, 1, fd.shape).astype('f')
    dr.backward(dr.sum(image * Array3f(weights)))
    np.testing.assert_allclose(dr.grad(parameter)[0], np.sum(tangent * weights), rtol=2e-4, atol=2e-4)


def test_surface_emitter_and_stale_replay():
    scene = volume_scene(8, 4)
    lamp = scene.param_map["Emitter[0]"]
    dr.enable_grad(lamp.radiance)
    tracer = PRBPathTracer(2)
    image = tracer.renderD(scene, parameters=[lamp.radiance])
    dr.backward(dr.sum(image))
    np.testing.assert_allclose(dr.grad(lamp.radiance).numpy().ravel(), [8, 8, 8])
    image = tracer.renderD(scene, parameters=[lamp.radiance])
    scene.configure()
    with pytest.raises(RuntimeError, match="reconfigured"):
        dr.backward(dr.sum(image))


def test_cornell_radiance_against_mis():
    scene, _ = _surface_setup(validate(dict(scene="cornell", derivative="albedo")), find_assets())
    scene.opts.width = scene.opts.height = 24
    scene.opts.spp = 256
    scene.configure()
    replay, mis = PRBPathTracer(4), PathTracer(4)
    images, references = [], []
    for seed in range(8):
        images.append(replay.renderC(scene, seed=seed).numpy())
        scene.seed = 200 + seed
        references.append(mis.renderC(scene, 0).numpy())
    image, reference = np.mean(images, axis=0), np.mean(references, axis=0)
    # Exclude the directly visible light when comparing reflected radiance.
    walls = reference.max(axis=0) < 2
    np.testing.assert_allclose(image[:, walls].mean(axis=1),
                               reference[:, walls].mean(axis=1), rtol=.04)
    assert np.sqrt(np.mean((image - reference)**2)) < .075
