"""Surface path replay for fixed cameras and geometry."""
from dataclasses import dataclass

import drjit as dr
from drjit import cuda
from drjit.cuda import Array2f, Array3f, PCG32, UInt

from .psdr_jit import BSDF, Emitter, Bitmap1fD, Bitmap3fD, RayC, RayD


def camera_samples(scene, sensor_id, seed):
    width, height, spp = scene.opts.width, scene.opts.height, scene.opts.spp
    size = width * height * spp
    if min(width, height, spp) <= 0 or size > 2**31 - 1:
        raise ValueError("Expected positive film dimensions and spp within the GPU lane limit")
    initstate = cuda.UInt64(seed)
    dr.make_opaque(initstate)  # Changing a seed should reuse the compiled sampling kernels.
    rng = PCG32(size, initstate=initstate)
    pixel = dr.arange(UInt, size) // spp
    uv = Array2f((pixel % width + rng.next_float32()) / width,
                 (pixel // width + rng.next_float32()) / height)
    ray = scene._primary_ray(sensor_id, uv)
    ray.d = dr.normalize(ray.d)
    return ray, rng, spp


@dataclass
class _SurfacePaths:
    scene: object
    revision: int
    ray: RayC
    rng: PCG32
    depth: int
    parameters: tuple

    def vertices(self, continuation=None):
        if self.scene._configuration != self.revision:
            raise RuntimeError("The scene was reconfigured between rendering and path replay")
        if continuation is None:
            ray = RayD(cuda.ad.Array3f(self.ray.o), cuda.ad.Array3f(self.ray.d))
            start, rng, active = 0, self.rng, cuda.ad.Bool(True)
        else:
            start, ray, rng, active = continuation
        rng = PCG32(rng)
        for depth in range(start, self.depth + 1):
            next_vertex = None
            with dr.suspend_grad():
                hit = self.scene.unit_ray_intersectAD(ray, active)
                if depth < self.depth:
                    u = cuda.ad.Array3f(*(rng.next_float32() for _ in range(3)))
                    wo, direction, pdf = hit._prb_sample(u)
                    ray, active = RayD(hit.p, direction), pdf > 0
                    dr.schedule(ray.o, ray.d, active, rng.state)
                    next_vertex = depth + 1, ray, PCG32(rng), active
            light = hit._emission()
            weight = hit._prb_weight(wo, pdf) if depth < self.depth else cuda.ad.Array3f(0)
            yield light, weight, next_vertex


def _radiance(responses, replay=False):
    # Replay subtracts local contributions from this sum. Double precision
    # retains dim indirect light in the presence of much brighter emitters.
    throughput, image = cuda.Array3f64(1), cuda.Array3f64(0)
    last_light = cuda.Array3i(-1)
    for depth, (light, weight) in enumerate(responses):
        image += throughput * light
        throughput *= weight
        if replay:
            last_light = dr.select(light != 0, depth, last_light)
        dr.eval(image, throughput, last_light)
    return image, last_light


def _compact_continuation(continuation, lanes):
    depth, ray, rng, *masks = continuation

    def gather(array):
        return dr.gather(type(array), array, lanes)

    ray = type(ray)(gather(ray.o), gather(ray.d))
    return (depth, ray, gather(rng), *(gather(mask) for mask in masks))


def _weight_adjoint(remaining, throughput, weight, has_later_light, image, continuation, trace_suffix):
    # A zero weight can have a nonzero derivative. Regenerate only affected
    # suffixes, also recovering significant digits after severe cancellation.
    has_suffix = has_later_light & (throughput != 0)
    # Local BSDF and DDA evaluations are float32 even though the path sum is
    # double. Replaying an attached evaluation can shift it by a few ulps.
    recover = has_suffix & ((abs(weight) < 1e-4) | (abs(remaining) <= 1e-3 * abs(image)))
    result = Array3f(dr.select(has_suffix, remaining / dr.select(weight != 0, weight, 1), 0))
    # A few ill-conditioned samples should not retrace the whole image.
    dr.schedule(remaining, weight)
    lanes = dr.compress(dr.any(recover))
    if dr.width(lanes):
        suffix = trace_suffix(_compact_continuation(continuation, lanes))
        prefix = dr.gather(type(throughput), throughput, lanes)
        dr.scatter(result, Array3f(prefix * suffix), lanes)
    return result


def _trace(paths, replay=False, continuation=None):
    with dr.suspend_grad():
        return _radiance(((Array3f(light), Array3f(weight)) for light, weight, _
                          in paths.vertices(continuation)), replay)


def _shading_inputs(scene):
    # The virtual BSDF/emitter calls read arrays stored on scene objects. Their
    # producers must finish before forward replay, even for a shared parameter
    # several arithmetic operations upstream of the actual texture or radiance.
    pending = [obj for obj in scene.param_map.values() if isinstance(obj, (BSDF, Emitter))]
    seen, inputs = set(), []
    while pending:
        obj = pending.pop()
        if obj in seen:
            continue
        seen.add(obj)
        for name, descriptor in vars(type(obj)).items():
            if not isinstance(descriptor, property):
                continue
            value = getattr(obj, name)
            if isinstance(value, (Bitmap1fD, Bitmap3fD, BSDF)):
                pending.append(value)
            elif dr.is_array_v(type(value)) and dr.grad_enabled(value):
                inputs.append(value)
    return inputs


class _SurfaceReplay(dr.CustomOp):
    def eval(self, paths):
        self.paths = paths
        self.image, self.last_light = _trace(paths, replay=True)
        return cuda.ad.Array3f(self.image)

    def forward(self):
        throughput, d_throughput, tangent = Array3f(1), Array3f(0), Array3f(0)
        with dr.resume_grad(*self.paths.parameters):
            for light, weight, _ in self.paths.vertices():
                dl, dw = dr.forward_to(light, weight, flags=dr.ADFlag.AllowNoGrad)
                light, weight = Array3f(light), Array3f(weight)
                tangent += d_throughput * light + throughput * Array3f(dl)
                d_throughput = d_throughput * weight + throughput * Array3f(dw)
                throughput *= weight
                dr.eval(tangent, throughput, d_throughput)
        self.set_grad_out(tangent)

    def backward(self):
        throughput, remaining = cuda.Array3f64(1), self.image
        adjoint = Array3f(self.grad_out())
        with dr.resume_grad(*self.paths.parameters), dr.isolate_grad():
            for depth, (light, weight, next_vertex) in enumerate(self.paths.vertices()):
                w = Array3f(weight)
                remaining = remaining - throughput * Array3f(light)
                weight_adjoint = Array3f(0)
                if next_vertex is not None:
                    weight_adjoint = _weight_adjoint(
                        remaining, throughput, w, self.last_light > depth, self.image, next_vertex,
                        lambda state: _trace(self.paths, continuation=state)[0])
                local = adjoint * (Array3f(throughput) * light + weight_adjoint * weight)
                dr.backward(dr.sum(local), flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
                throughput *= w
                dr.eval(throughput, remaining)


class PRBPathTracer:
    """BSDF-sampled surface PRB for material and emitter derivatives."""

    def __init__(self, max_depth=3):
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        self.max_depth = max_depth

    def _paths(self, scene, sensor_id, seed, parameters=()):
        ray, rng, spp = camera_samples(scene, sensor_id, seed)
        return _SurfacePaths(scene, scene._configuration, ray, rng, self.max_depth, tuple(parameters)), spp

    def renderC(self, scene, sensor_id=0, seed=0):
        paths, spp = self._paths(scene, sensor_id, seed)
        return dr.block_sum(Array3f(_trace(paths)[0]), spp) / spp

    def renderD(self, scene, sensor_id=0, seed=0, *, parameters):
        """Register enabled material/emitter arrays or the parameters used to build them."""
        parameters = tuple(parameters)
        if dr.grad_enabled(parameters):
            parameters += tuple(_shading_inputs(scene))
        paths, spp = self._paths(scene, sensor_id, seed, parameters)
        return dr.block_sum(dr.custom(_SurfaceReplay, paths), spp) / spp
