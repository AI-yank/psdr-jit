"""Bounded volume transport and path replay for fixed PSDR scenes."""

import math
from dataclasses import dataclass

import drjit as dr
from drjit import cuda
from drjit.dda import integrate
from drjit.cuda import Array2f, Array3f, Bool, Float, PCG32, UInt

from .psdr_jit import RayC
from .prb import _radiance, _weight_adjoint, camera_samples


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


@dataclass(frozen=True)
class HeterogeneousMedium(HomogeneousMedium):
    """Density at grid vertices in (z, y, x) order, including both box endpoints."""

    density: cuda.ad.TensorXf

    def __post_init__(self):
        super().__post_init__()
        density = cuda.ad.TensorXf(self.density)
        if len(density.shape) != 3 or min(density.shape) < 2:
            raise ValueError("Expected a 3D density grid with at least two vertices per axis")
        if not bool(dr.all(dr.isfinite(density.array) & (density.array >= 0))):
            raise ValueError("Density must be finite and nonnegative")
        object.__setattr__(self, "density", density)


@dataclass
class _Vertex:
    origin: Array3f
    direction: Array3f
    length: Float
    emission: Array3f
    scatter: Bool | None = None
    travel: Float | None = None
    point: Array3f | None = None
    base: Array3f | None = None
    light_dir: Array3f | None = None
    shadow: Float | None = None
    illumination: Array3f | None = None
    continuation: tuple | None = None


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

    def vertices(self, continuation=None):
        if self.configuration != self.scene._configuration:
            raise RuntimeError("The scene was reconfigured between rendering and path replay")
        if continuation is None:
            start, ray, rng, active, include_emission = 0, self.ray, self.rng, Bool(True), Bool(True)
        else:
            start, ray, rng, active, include_emission = continuation
        rng = PCG32(rng)

        for depth in range(start, self.max_depth + 1):
            hit = self.scene.unit_ray_intersect(ray, active)
            entry, length = _interval(self.bounds, ray.o, ray.d, dr.select(hit.is_valid(), hit.t, dr.inf))
            length = dr.select(active, length, 0)
            emission = dr.select(active & include_emission & hit.is_valid(), hit._emission(), 0)
            vertex = _Vertex(ray.o + entry * ray.d, ray.d, length, emission)
            if depth == self.max_depth:
                yield vertex
                return

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
            vertex.scatter, vertex.travel = scatter, travel
            vertex.point, vertex.base = point, base

            if self.scene.get_num_emitters():
                light_dir, distance, incident = self.scene._sample_incident_light(
                    point, Array2f(rng.next_float32(), rng.next_float32()), scatter)
                _, shadow = _interval(self.bounds, point, light_dir, distance)
                shadow = dr.select(scatter, shadow, 0)
                illumination = dr.select(scatter, incident * (1 / (4 * math.pi)), 0)
                vertex.light_dir, vertex.shadow, vertex.illumination = light_dir, shadow, illumination

            z = 1 - 2 * rng.next_float32()
            sin_phi, cos_phi = dr.sincos(2 * math.pi * rng.next_float32())
            radius = dr.safe_sqrt(1 - z * z)
            ray = RayC(point, dr.select(scatter, Array3f(radius * cos_phi, radius * sin_phi, z), surface_dir))
            active = scatter | surface
            # Emitter sampling already accounted for light reached along a phase ray.
            include_emission = ~scatter
            dr.schedule(ray.o, ray.d, rng.state, active, include_emission)
            vertex.continuation = (depth + 1, ray, PCG32(rng), active, include_emission)
            yield vertex

    def trace(self, absorption, scattering, remaining=None):
        extinction = absorption + scattering
        replay = remaining is not None
        throughput, radiance = Array3f(1), Array3f(0)
        d_extinction, d_scattering, d_throughput = Array3f(0), Array3f(0), Array3f(0)
        for v in self.vertices():
            emission = v.emission * dr.exp(-extinction * v.length)
            visible = throughput * emission
            if replay:
                remaining = remaining - visible
                d_extinction -= v.length * visible
                d_scattering += d_throughput * emission
            else:
                radiance += visible
            if v.scatter is None:
                break
            base = v.base * dr.exp(-extinction * v.travel)
            event = dr.select(v.scatter, scattering, 1)
            if replay:
                d_extinction -= v.travel * remaining
                d_throughput = base * (d_throughput * event + dr.select(v.scatter, throughput, 0))
            throughput *= base * event
            if v.shadow is not None:
                illumination = v.illumination * dr.exp(-extinction * v.shadow)
                direct = throughput * illumination
                if replay:
                    remaining = remaining - direct
                    d_extinction -= v.shadow * direct
                    d_scattering += d_throughput * illumination
                else:
                    radiance += direct
            dr.eval(throughput, radiance, remaining, d_extinction, d_scattering, d_throughput)

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


def _density_at(density, point, lower, upper, active):
    # Integer positions are grid vertices, matching dda.integrate().
    nz, ny, nx = density.shape
    pos = dr.clip((point - lower) / (upper - lower), 0, 1) * Array3f(nx - 1, ny - 1, nz - 1)
    ix, iy, iz = (dr.minimum(UInt(p), n - 2) for p, n in zip(pos, (nx, ny, nz)))
    offset = (iz * ny + iy) * nx + ix
    fx, fy, fz = pos - Array3f(ix, iy, iz)
    scalar = type(density.array)

    def row(index):
        return dr.lerp(dr.gather(scalar, density.array, index, active),
                       dr.gather(scalar, density.array, index + 1, active), fx)

    return dr.lerp(dr.lerp(row(offset), row(offset + nx), fy),
                   dr.lerp(row(offset + ny * nx), row(offset + (ny + 1) * nx), fy), fz)


def _grid_response(v, bounds, absorption, scattering, density):
    """Local emitted radiance (including direct light) and continuation weight."""
    scalar = type(density.array)
    vector = cuda.ad.Array3f if dr.is_diff_v(scalar) else Array3f
    lower, upper = map(Array3f, bounds)
    extinction = absorption + scattering

    def transmittance(origin, direction, length, active):
        active = active & (length > 0)
        # Inactive light samples can have zero or undefined directions.
        direction = dr.select(active, direction, Array3f(0, 0, 1))
        mass = integrate(vector(dr.select(active, origin, lower)), vector(direction),
                         scalar(length), vector(lower), vector(upper), density, dr.mask_t(scalar)(active))
        return dr.exp(-extinction * mass)

    emission = v.emission * transmittance(v.origin, v.direction, v.length, dr.any(v.emission != 0))
    if v.scatter is None:
        return emission, vector(0)
    local_density = _density_at(density, dr.select(v.scatter, v.point, lower), lower, upper, v.scatter)
    weight = v.base * dr.select(v.scatter, scattering * local_density, 1)
    weight *= transmittance(v.origin, v.direction, v.travel, dr.any(v.base != 0))
    if v.shadow is not None:
        emission += weight * v.illumination * transmittance(
            v.point, v.light_dir, v.shadow, dr.any(v.illumination != 0))
    return emission, weight


def _grid_trace(paths, absorption, scattering, density, *, replay=False, continuation=None):
    vertices = paths.vertices(continuation)
    return _radiance((_grid_response(v, paths.bounds, absorption, scattering, density)
                      for v in vertices), replay)


class _GridReplay(dr.CustomOp):
    def eval(self, absorption, scattering, density, paths, enabled):
        self.paths, self.enabled = paths, enabled
        self.inputs = Float(absorption), Float(scattering), cuda.TensorXf(density)
        self.image, self.last_light = _grid_trace(paths, *self.inputs, replay=True)
        return cuda.ad.Array3f(self.image)

    def _variables(self):
        a, s, grid = self.inputs
        variables = cuda.ad.Float(a), cuda.ad.Float(s), cuda.ad.TensorXf(grid)
        for variable, enabled in zip(variables, self.enabled):
            if enabled:
                dr.enable_grad(variable)
        return variables

    def forward(self):
        throughput, d_throughput, tangent = Array3f(1), Array3f(0), Array3f(0)
        directions = [self.grad_in(name) for name in ("absorption", "scattering", "density")]
        for vertex in self.paths.vertices():
            with dr.resume_grad():
                variables = self._variables()
                for variable, direction in zip(variables, directions):
                    dr.set_grad(variable, direction)
                light, weight = _grid_response(vertex, self.paths.bounds, *variables)
                dl, dw = dr.forward_to(light, weight, flags=dr.ADFlag.ClearEdges | dr.ADFlag.AllowNoGrad)
                dl, dw = Array3f(dl), Array3f(dw)
            light, weight = Array3f(light), Array3f(weight)
            tangent += d_throughput * light + throughput * dl
            d_throughput = d_throughput * weight + throughput * dw
            throughput *= weight
            dr.eval(tangent, throughput, d_throughput)
        self.set_grad_out(tangent)

    def backward(self):
        throughput, remaining = cuda.Array3f64(1), self.image
        adjoint = Array3f(self.grad_out())
        with dr.resume_grad():
            variables = self._variables()
            inputs = [(name, v) for name, v in zip(("absorption", "scattering", "density"), variables)
                      if dr.grad_enabled(v)]
            for depth, vertex in enumerate(self.paths.vertices()):
                # Scatter voxel contributions into a fresh gradient each bounce.
                # Adding tiny atomics directly to a large prefix loses precision.
                dr.clear_grad(variables)
                light, weight = _grid_response(vertex, self.paths.bounds, *variables)
                w = Array3f(weight)
                remaining = remaining - throughput * Array3f(light)
                weight_adjoint = Array3f(0)
                if vertex.continuation is not None:
                    weight_adjoint = _weight_adjoint(
                        remaining, throughput, w, self.last_light > depth, self.image, vertex.continuation,
                        lambda state: _grid_trace(self.paths, *self.inputs, continuation=state)[0])
                local = light * Array3f(throughput) + weight * weight_adjoint
                dr.backward(dr.sum(dr.dot(local, adjoint)),
                            flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
                throughput *= w
                for name, v in inputs:
                    self.set_grad_in(name, dr.grad(v))
                dr.eval(throughput, remaining, *(self.grad_in(name) for name, _ in inputs))


class VolumePathTracer:
    """Render a fixed scene with one isotropic medium and at most max_depth scattering events."""

    def __init__(self, medium, max_depth=3):
        if not isinstance(medium, HomogeneousMedium):
            raise TypeError("Expected a HomogeneousMedium or HeterogeneousMedium")
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        self.medium, self.max_depth = medium, max_depth

    def _render(self, scene, sensor_id, seed, differentiable):
        ray, rng, spp = camera_samples(scene, sensor_id, seed)
        medium = self.medium
        paths = _Paths(scene, scene._configuration, ray, rng, (medium.lower, medium.upper), self.max_depth)
        if isinstance(medium, HeterogeneousMedium):
            if differentiable:
                inputs = medium.sigma_a, medium.sigma_s, medium.density
                samples = dr.custom(_GridReplay, *inputs, paths, tuple(map(dr.grad_enabled, inputs)))
            else:
                image, _ = _grid_trace(paths, Float(medium.sigma_a), Float(medium.sigma_s),
                                       cuda.TensorXf(medium.density))
                samples = Array3f(image)
        elif differentiable:
            samples = dr.custom(_VolumeReplay, medium.sigma_a, medium.sigma_s, paths)
        else:
            samples = paths.trace(Float(medium.sigma_a), Float(medium.sigma_s))
        return dr.block_sum(samples, spp) / spp

    def renderC(self, scene, sensor_id=0, seed=0):
        return self._render(scene, sensor_id, seed, False)

    def renderD(self, scene, sensor_id=0, seed=0):
        return self._render(scene, sensor_id, seed, True)
