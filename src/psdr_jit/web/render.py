"""GPU demo scenes, gradient estimators, and preview encoding."""

import base64
import io
from functools import lru_cache
from pathlib import Path

from .volume_scenes import FOG_CUBE_BOUNDS, VOLUME_BOUNDS, fog_cube_scene, smoke_density, smoke_scene, volume_scene
from .white_room import PLUME_BOUNDS, white_room_scene

DEFAULTS = dict(scene="cornell", integrator="path", derivative="x", x=0., y=0.,
                roughness=.3, albedo=.7, light=1., resolution=128, spp=8, seed=42,
                exposure=0., absorption=.3, scattering=.8, density=1., npass=8, method="ad", fd_step=.01)
WHITE_ROOM_DEFAULTS = dict(derivative="density", integrator="volume", absorption=.24, scattering=.65,
                           method="compare", fd_step=.004, resolution=256, spp=32, npass=24, exposure=-1.)
VOLUME_SCENES = ("volume", "fog_cube", "smoke", "white_room")
ENUMS = dict(scene=("cornell", "bunny", *VOLUME_SCENES), integrator=("path", "prb", "direct", "volume"),
             method=("ad", "fd", "compare"), derivative=("x", "y", "roughness", "albedo", "light", "absorption", "scattering", "density"),
             resolution=(64, 128, 256), spp=(4, 8, 16, 32))
BOUNDS = dict(x=(-1, 1), y=(-1, 1), roughness=(.08, 1), albedo=(.05, 1),
              light=(.1, 3), seed=(0, 65535), exposure=(-3, 3), absorption=(0, 3), scattering=(0, 3),
              density=(0, 4), npass=(1, 65536), fd_step=(.0001, .25))


def translate(matrix_type, xyz):
    x, y, z = xyz
    return matrix_type([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def validate(data):
    if not isinstance(data, dict) or data.keys() - DEFAULTS.keys():
        raise ValueError("Unknown render settings")
    # Existing API clients that omit npass request a single image.
    room = data.get("scene") == "white_room"
    settings = DEFAULTS | (WHITE_ROOM_DEFAULTS if room else {}) | {"npass": 1} | data
    volume = settings["scene"] in VOLUME_SCENES
    if volume:
        settings["integrator"] = data.get("integrator", "volume")
        settings["derivative"] = data.get("derivative", "density" if room else "absorption")
    if settings["integrator"] == "prb":
        settings["derivative"] = data.get("derivative", "albedo")
        if settings["derivative"] in ("x", "y"):
            raise ValueError("Surface PRB requires fixed geometry")
    if ((settings["integrator"] == "volume") != volume or
            (settings["derivative"] in ("absorption", "scattering", "density")) != volume or
            (settings["derivative"] == "density" and settings["scene"] not in ("smoke", "white_room")) or
            (settings["method"] == "compare" and not volume)):
        raise ValueError("Invalid integrator or derivative for this scene")
    for key, value in settings.items():
        if key in ENUMS:
            if isinstance(value, bool) or value not in ENUMS[key]:
                raise ValueError(f"Invalid {key}")
        elif (isinstance(value, bool) or not isinstance(value, (float, int))
              or not BOUNDS[key][0] <= value <= BOUNDS[key][1]):
            raise ValueError(f"Invalid {key}")
    for key in ("resolution", "spp", "seed", "npass"):
        if not isinstance(settings[key], int):
            raise ValueError(f"{key} must be an integer")
    return settings


def find_assets(path=None):
    candidates = [Path(path)] if path else [Path.cwd() / "tutorials/data",
                                          Path(__file__).resolve().parents[3] / "tutorials/data"]
    required = ["mesh/bunny_low.obj", "envmap/ballroom_1k.exr"] + [
        f"cbox/cbox_{part}.obj" for part in
        ("luminaire", "smallball", "largeball", "floor", "ceiling", "back", "greenwall", "redwall")]
    for candidate in candidates:
        if all((candidate / item).is_file() for item in required):
            return str(candidate.resolve())
    raise ValueError("Scene assets missing. Run from the repository root or pass --assets PATH/to/tutorials/data.")


def _volume_setup(settings, assets):
    import psdr_jit as psdr

    room = settings["scene"] == "white_room"
    grid = None
    if room:
        sc = white_room_scene(assets, settings["resolution"], settings["spp"], settings["light"])
        bounds = PLUME_BOUNDS
    elif settings["scene"] in ("smoke", "fog_cube"):
        build_scene = smoke_scene if settings["scene"] == "smoke" else fog_cube_scene
        sc = build_scene(settings["resolution"], settings["spp"], settings["light"])
        bounds = FOG_CUBE_BOUNDS
    else:
        sc = volume_scene(settings["resolution"], settings["spp"], settings["light"])
        bounds = VOLUME_BOUNDS

    if settings["scene"] in ("smoke", "white_room"):
        from drjit.cuda.ad import TensorXf
        grid = TensorXf(smoke_density((24, 32, 28)) if room else smoke_density())

    def configure(value):
        key = settings["derivative"]
        a = value if key == "absorption" else settings["absorption"]
        s = value if key == "scattering" else settings["scattering"]
        scale = value if key == "density" else settings["density"]
        medium = (psdr.HeterogeneousMedium(*bounds, a, s, grid * scale) if grid is not None else
                  psdr.HomogeneousMedium(*bounds, a, s))
        return psdr.VolumePathTracer(medium, 4 if room else 3)

    return sc, configure


def _surface_setup(settings, assets):
    import drjit as dr
    import psdr_jit as psdr
    from drjit.cuda import Matrix4f as MatrixC
    from drjit.cuda.ad import Array3f, Float, Matrix4f

    p = {key: Float(settings[key]) for key in ("x", "y", "roughness", "albedo", "light")}
    sc = psdr.Scene()
    size, spp = settings["resolution"], settings["spp"]
    sc.opts = psdr.RenderOption(size, size, spp, 0, 0)
    sc.opts.log_level = 0
    cornell = settings["scene"] == "cornell"
    camera = psdr.PerspectiveCamera(60 if cornell else 80, .000001, 10000000.)
    camera.to_world = (translate(Matrix4f, [278., 273., -500.]) if cornell else
                       Matrix4f([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
    sc.add_Sensor(camera)
    sc.add_BSDF(psdr.MicrofacetBSDF([.12, .12, .12], [.7, .5, .15], .3), "target")
    root = Path(assets)
    if cornell:
        for name, color in dict(white=[.95]*3, gray=[.5]*3, green=[.2, .9, .2],
                                red=[.9, .2, .2], cyan=[.2, .8, .8]).items():
            sc.add_BSDF(psdr.DiffuseBSDF(color), name)
        for part, material in (("luminaire", "white"), ("smallball", "target"),
                               ("largeball", "cyan"), ("floor", "white"), ("ceiling", "white"),
                               ("back", "gray"), ("greenwall", "green"), ("redwall", "red")):
            emitter = psdr.AreaLight([20, 20, 8]) if part == "luminaire" else None
            transform = translate(MatrixC, [0, -.5, 0]) if emitter else dr.identity(MatrixC)
            sc.add_Mesh(str(root / f"cbox/cbox_{part}.obj"), transform, material, emitter)
        target, units = "Mesh[1]", 100
    else:
        sc.add_Mesh(str(root / "mesh/bunny_low.obj"), translate(MatrixC, [0, 0, -100]), "target", None)
        sc.add_EnvironmentMap(str(root / "envmap/ballroom_1k.exr"), dr.identity(dr.scalar.Matrix4f), 1.)
        target, units = "Mesh[0]", 30
    key = settings["derivative"]
    material = sc.param_map["BSDF[id=target]"]
    emitter = sc.param_map["Emitter[0]"]
    mesh = sc.param_map[target]
    color = Array3f(1, .7, .2) if cornell else Array3f(.2, .85, .8)

    def set_light(value):
        if cornell:
            emitter.radiance = Array3f(20, 20, 8) * value
        else:
            emitter.scale = value

    def set_position():
        mesh.set_transform(translate(Matrix4f, [p["x"] * units, p["y"] * units, 0]))

    material.roughness.data = p["roughness"]
    material.diffuseReflectance.data = color * p["albedo"]
    set_light(p["light"])
    set_position()
    # Initialize the camera before configuring the geometry boundary estimators.
    with dr.suspend_grad():
        sc.configure()
    if settings["method"] == "ad" and settings["derivative"] in ("x", "y"):
        sc.opts.sppe = spp
        sc.opts.sppse = spp
    integrator = {"path": psdr.PathTracer, "prb": psdr.PRBPathTracer,
                  "direct": psdr.Direct}[settings["integrator"]](2)

    def configure(value):
        # BSDF evaluations read these uniforms directly. Only light and
        # transform edits need to rebuild scene sampling or geometry data.
        if key == "roughness":
            material.roughness.data = value
        elif key == "albedo":
            material.diffuseReflectance.data = color * value
        else:
            if key == "light":
                set_light(value)
            else:
                p[key] = value
                set_position()
            sc.configure([0])
        return integrator

    return sc, configure


def _setup_scene(settings, assets):
    import drjit as dr

    if not dr.has_backend(dr.JitBackend.CUDA):
        raise RuntimeError("Rendering requires an NVIDIA GPU and a CUDA/OptiX driver.")
    dr.set_flag(dr.JitFlag.Debug, False)
    if settings["scene"] in VOLUME_SCENES:
        return _volume_setup(settings, assets)
    return _surface_setup(settings, assets)


def render_arrays(settings, assets):
    return _render_arrays(settings, settings["seed"], *_setup_scene(settings, assets))


@lru_cache(maxsize=1)
def _worker_scene(job_id, settings_items, assets):
    # Each GPU worker handles one job at a time. Seed changes between passes
    # do not change scene data. A new job always gets its own scene.
    return _setup_scene(dict(settings_items), assets)


def release_worker_scene():
    _worker_scene.cache_clear()


def render_pass(job_id, settings, assets, pass_index):
    try:
        scene, configure = _worker_scene(job_id, tuple(settings.items()), assets)
        seed = (settings["seed"] + pass_index) % 65536
        result = _render_arrays(settings, seed, scene, configure)
    except Exception:
        release_worker_scene()
        raise
    if pass_index + 1 == settings["npass"]:
        release_worker_scene()
    return result


def _render_arrays(settings, seed, scene, configure):
    import drjit as dr
    import numpy as np
    from drjit.cuda.ad import Float

    key = settings["derivative"]
    value = settings[key]

    def rgb(pixels):
        return pixels.numpy().T.reshape(scene.opts.height, scene.opts.width, 3).copy()

    method = settings["method"]
    if method in ("ad", "compare"):
        parameter = Float(value)
        dr.enable_grad(parameter)
        replay_args = {"parameters": [parameter]} if settings["integrator"] == "prb" else {}
        pixels = configure(parameter).renderD(scene, seed=seed, **replay_args)
        primal = rgb(pixels)
        dr.forward(parameter)
        gradient = rgb(dr.grad(pixels))
    if method in ("fd", "compare"):
        # Reuse the scene and random sample streams for every point in this pass.
        def sample(at):
            return rgb(configure(Float(at)).renderC(scene, seed=seed)).astype(np.float64)

        if method == "fd":
            primal = sample(value)
        step = settings["fd_step"]
        lo, hi = BOUNDS[key]
        if value - step >= lo and value + step <= hi:
            fd = (sample(value + step) - sample(value - step)) / (2 * step)
        else:
            # Second-order one-sided stencil near either end of the slider.
            step = step if value - step < lo else -step
            near, far = sample(value + step), sample(value + 2 * step)
            fd = (4 * (near - primal) - (far - primal)) / (2 * step)
        gradient = np.stack((gradient, fd)) if method == "compare" else fd
    if not np.isfinite(primal).all() or not np.isfinite(gradient).all():
        raise RuntimeError("Renderer produced non-finite pixels; reduce the settings and retry.")
    return primal, gradient


@lru_cache(maxsize=1)
def _cubicl_palette():
    import numpy as np

    rgb = np.loadtxt(Path(__file__).with_name("CubicL.txt"))
    # Match the reference plot's 256-bin LinearSegmentedColormap.
    rows = np.arange(len(rgb))
    samples = np.linspace(0, len(rgb) - 1, 256)
    colors = np.column_stack([np.interp(samples, rows, channel) for channel in rgb.T])
    return np.uint8(np.clip(colors, 0, 1) * 255)


def _png(pixels):
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@lru_cache(maxsize=1)
def _colorbar():
    return _png(_cubicl_palette()[None, :, :])


def encode_preview(settings, primal, gradient):
    import numpy as np

    linear = np.maximum(primal * 2 ** settings["exposure"], 0)
    srgb = np.where(linear <= .0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - .055)
    compare = settings["method"] == "compare"
    signed = gradient.mean(axis=-1)
    limit = float(np.quantile(np.abs(signed), .99)) or float(np.max(np.abs(signed))) or 1.
    palette = _cubicl_palette()
    normalized = .5 * (np.clip(signed / limit, -1, 1) + 1)
    bins = np.minimum((normalized * len(palette)).astype(int), len(palette) - 1)

    preview = dict(render=_png(np.uint8(np.clip(srgb, 0, 1) * 255)), gradient=_png(palette[bins[0] if compare else bins]),
                colorbar=_colorbar(), limit=limit,
                settings=settings,
                gradient_max=float(np.max(np.abs(gradient))), mean=float(primal.mean()))
    if compare:
        error = gradient[0] - gradient[1]
        norm = np.linalg.norm(gradient[1])
        preview.update(fd_gradient=_png(palette[bins[1]]), rmse=float(np.sqrt(np.mean(error ** 2))),
                       relative_l2=float(np.linalg.norm(error) / norm) if norm else None)
    return preview
