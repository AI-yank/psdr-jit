"""Homogeneous volume transport and path replay for fixed PSDR scenes."""

import math
from dataclasses import dataclass

import drjit as dr
from drjit import cuda
from drjit.cuda import Array2f, Array3f, Bool, Float, PCG32, UInt

from .psdr_jit import RayC


@dataclass(frozen=True)
class HomogeneousMedium:
    """An axis-aligned box. Scalar absorption and scattering are in inverse scene units."""

    lower: tuple
    upper: tuple
    sigma_a: cuda.ad.Float
    sigma_s: cuda.ad.Float

    def __post_init__(self):
        lower, upper = tuple(map(float, self.lower)), tuple(map(float, self.upper))
        if (len(lower) != 3 or len(upper) != 3 or
                not all(math.isfinite(a) and math.isfinite(b) and a < b for a, b in zip(lower, upper))):
            raise ValueError("Expected finite, ordered 3D medium bounds")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        for name in ("sigma_a", "sigma_s"):
            value = cuda.ad.Float(getattr(self, name))
            if dr.width(value) != 1 or not bool(dr.isfinite(value) & (value >= 0)):
                raise ValueError(f"{name} must be one finite, nonnegative coefficient")
            object.__setattr__(self, name, value)


def _interval(bounds, origin, direction, limit):
    """Clip a unit ray to the box and its nearest surface, including parallel rays."""
    entry, exit_t = Float(0), limit
    for lo, hi, o, d in zip(*bounds, origin, direction):
        parallel = d == 0
        reciprocal = dr.rcp(dr.select(parallel, 1, d))
        a, b = (lo - o) * reciprocal, (hi - o) * reciprocal
        entry = dr.maximum(entry, dr.select(parallel, 0, dr.minimum(a, b)))
        exit_t = dr.minimum(exit_t, dr.select(parallel, dr.inf, dr.maximum(a, b)))
        exit_t = dr.select(parallel & ((o < lo) | (o > hi)), 0, exit_t)
    return entry, dr.maximum(0, exit_t - entry)


@dataclass(frozen=True)
class _Paths:
    scene: object
    configuration: int
    ray: RayC
    rng: PCG32
    bounds: tuple
    max_depth: int

    def trace(self, absorption, scattering, remaining=None):
        if self.configuration != self.scene._configuration:
            raise RuntimeError("The scene was reconfigured between rendering and path replay")
        rng, ray = PCG32(self.rng), self.ray
        extinction = absorption + scattering
        replay = remaining is not None
        throughput, radiance = Array3f(1), Array3f(0)
        d_extinction, d_scattering, d_throughput = Array3f(0), Array3f(0), Array3f(0)
        active, include_emission = Bool(True), Bool(True)

        for depth in range(self.max_depth + 1):
            hit = self.scene.unit_ray_intersect(ray, active)
            entry, length = _interval(self.bounds, ray.o, ray.d, dr.select(hit.is_valid(), hit.t, dr.inf))
            length = dr.select(active, length, 0)
            emission = dr.select(include_emission & hit.is_valid(), hit._emission(), 0)
            emission *= dr.exp(-extinction * length)
            visible = throughput * emission
            if replay:
                remaining = remaining - visible
                d_extinction -= length * visible
                d_scattering += d_throughput * emission
            else:
                radiance += visible
            if depth == self.max_depth:
                break

            # Estimate both the surface term and the in-scattering integral.
            # When both can contribute, choose one and compensate its probability.
            surface_dir, bsdf = hit._sample_bsdf(Array3f(*(rng.next_float32() for _ in range(3))))
            surface = active & hit.is_valid() & dr.any(bsdf != 0)
            volume = active & (length > 0)
            both = surface & volume
            scatter = volume & (~surface | (rng.next_float32() < .5))
            travel = dr.select(scatter, length * rng.next_float32(), length)
            point = dr.select(scatter, ray.o + (entry + travel) * ray.d, hit.p)
            base = dr.select(scatter, Array3f(length), bsdf) * dr.select(both, 2, 1)
            base *= dr.exp(-extinction * travel)
            event = dr.select(scatter, scattering, 1)
            if replay:
                # All later light crossed this segment. Reuse the detached
                # suffix radiance for its Beer-Lambert derivative (PRB).
                d_extinction -= travel * remaining
                # Product rule for each scattering factor, also valid at sigma_s=0.
                d_throughput = base * (d_throughput * event + dr.select(scatter, throughput, 0))
            throughput *= base * event

            if self.scene.get_num_emitters():
                light_dir, distance, incident = self.scene._sample_incident_light(
                    point, Array2f(rng.next_float32(), rng.next_float32()), scatter)
                _, shadow = _interval(self.bounds, point, light_dir, distance)
                shadow = dr.select(scatter, shadow, 0)
                illumination = dr.select(scatter, incident * (1 / (4 * math.pi)), 0)
                illumination *= dr.exp(-extinction * shadow)
                direct = throughput * illumination
                if replay:
                    remaining = remaining - direct
                    d_extinction -= shadow * direct
                    d_scattering += d_throughput * illumination
                else:
                    radiance += direct

            z = 1 - 2 * rng.next_float32()
            sin_phi, cos_phi = dr.sincos(2 * math.pi * rng.next_float32())
            radius = dr.safe_sqrt(1 - z * z)
            ray = RayC(point, dr.select(scatter, Array3f(radius * cos_phi, radius * sin_phi, z), surface_dir))
            active = scatter | surface
            # Emitter sampling already accounted for light reached along a phase ray.
            include_emission = ~scatter
            dr.eval(ray.o, ray.d, rng.state, active, include_emission, throughput, radiance,
                    remaining, d_extinction, d_scattering, d_throughput)

        return (d_extinction, d_extinction + d_scattering) if replay else radiance


class _VolumeReplay(dr.CustomOp):
    def eval(self, absorption, scattering, paths):
        self.paths = paths
        self.coefficients = Float(absorption), Float(scattering)
        self.primal = paths.trace(*self.coefficients)
        dr.eval(self.primal)
        return cuda.ad.Array3f(self.primal)

    def forward(self):
        da, ds = self.paths.trace(*self.coefficients, self.primal)
        self.set_grad_out(da * self.grad_in("absorption") + ds * self.grad_in("scattering"))

    def backward(self):
        da, ds = self.paths.trace(*self.coefficients, self.primal)
        adjoint = self.grad_out()
        self.set_grad_in("absorption", dr.sum(dr.dot(da, adjoint)))
        self.set_grad_in("scattering", dr.sum(dr.dot(ds, adjoint)))


class VolumePathTracer:
    """Render a fixed scene with one isotropic medium and at most max_depth scattering events."""

    def __init__(self, medium, max_depth=3):
        if not isinstance(medium, HomogeneousMedium):
            raise TypeError("Expected a HomogeneousMedium")
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        self.medium, self.max_depth = medium, max_depth

    def _render(self, scene, sensor_id, seed, differentiable):
        width, height, spp = scene.opts.width, scene.opts.height, scene.opts.spp
        size = width * height * spp
        if min(width, height, spp) <= 0 or size > 2**31 - 1:
            raise ValueError("Expected positive film dimensions and spp within the GPU lane limit")
        rng = PCG32(size, initstate=seed)
        pixel = dr.arange(UInt, size) // spp
        uv = Array2f((pixel % width + rng.next_float32()) / width,
                     (pixel // width + rng.next_float32()) / height)
        ray = scene._primary_ray(sensor_id, uv)
        ray.d = dr.normalize(ray.d)
        medium = self.medium
        paths = _Paths(scene, scene._configuration, ray, rng, (medium.lower, medium.upper), self.max_depth)
        if differentiable:
            samples = dr.custom(_VolumeReplay, medium.sigma_a, medium.sigma_s, paths)
        else:
            samples = paths.trace(Float(medium.sigma_a), Float(medium.sigma_s))
        return dr.block_sum(samples, spp) / spp

    def renderC(self, scene, sensor_id=0, seed=0):
        return self._render(scene, sensor_id, seed, False)

    def renderD(self, scene, sensor_id=0, seed=0):
        return self._render(scene, sensor_id, seed, True)
