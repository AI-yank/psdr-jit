"""GPU demo scenes, gradient estimators, and preview encoding."""

import base64
import io
from functools import lru_cache
from pathlib import Path

from .volume_scenes import FOG_CUBE_BOUNDS, VOLUME_BOUNDS, fog_cube_scene, volume_scene

DEFAULTS = dict(scene="cornell", integrator="path", derivative="x", x=0., y=0.,
                roughness=.3, albedo=.7, light=1., resolution=128, spp=8, seed=42,
                exposure=0., absorption=.3, scattering=.8, npass=8, method="ad", fd_step=.01)
VOLUME_SCENES = ("volume", "fog_cube")
ENUMS = dict(scene=("cornell", "bunny", *VOLUME_SCENES), integrator=("path", "direct", "volume"),
             method=("ad", "fd"), derivative=("x", "y", "roughness", "albedo", "light", "absorption", "scattering"),
             resolution=(64, 128, 256), spp=(4, 8, 16, 32))
BOUNDS = dict(x=(-1, 1), y=(-1, 1), roughness=(.08, 1), albedo=(.05, 1),
              light=(.1, 3), seed=(0, 65535), exposure=(-3, 3), absorption=(0, 3), scattering=(0, 3),
              npass=(1, 65536), fd_step=(.0001, .25))


def translate(matrix_type, xyz):
    x, y, z = xyz
    return matrix_type([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def validate(data):
    if not isinstance(data, dict) or data.keys() - DEFAULTS.keys():
        raise ValueError("Unknown render settings")
    # Existing API clients that omit npass request a single image.
    settings = DEFAULTS | {"npass": 1} | data
    volume = settings["scene"] in VOLUME_SCENES
    if volume:
        settings["integrator"] = data.get("integrator", "volume")
        settings["derivative"] = data.get("derivative", "absorption")
    if ((settings["integrator"] == "volume") != volume or
            (settings["derivative"] in ("absorption", "scattering")) != volume):
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


def _volume_setup(settings):
    import psdr_jit as psdr

    if settings["scene"] == "fog_cube":
        sc = fog_cube_scene(settings["resolution"], settings["spp"], settings["light"])
        bounds = FOG_CUBE_BOUNDS
    else:
        sc = volume_scene(settings["resolution"], settings["spp"], settings["light"])
        bounds = VOLUME_BOUNDS

    def configure(value):
        a, s = settings["absorption"], settings["scattering"]
        if settings["derivative"] == "absorption":
            a = value
        else:
            s = value
        return psdr.VolumePathTracer(psdr.HomogeneousMedium(*bounds, a, s))

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
    integrator = psdr.PathTracer(2) if settings["integrator"] == "path" else psdr.Direct(2)

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
        return _volume_setup(settings)
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

    size = settings["resolution"]
    key = settings["derivative"]
    value = settings[key]

    def rgb(pixels):
        return pixels.numpy().T.reshape(size, size, 3).copy()

    if settings["method"] == "ad":
        parameter = Float(value)
        dr.enable_grad(parameter)
        pixels = configure(parameter).renderD(scene, seed=seed)
        primal = rgb(pixels)
        dr.forward(parameter)
        gradient = rgb(dr.grad(pixels))
    else:
        # Reuse the scene and random sample streams for every point in this pass.
        def sample(at):
            return rgb(configure(Float(at)).renderC(scene, seed=seed))

        primal = sample(value)
        step = settings["fd_step"]
        lo, hi = BOUNDS[key]
        if value - step >= lo and value + step <= hi:
            gradient = (sample(value + step) - sample(value - step)) / (2 * step)
        else:
            # Second-order one-sided stencil near either end of the slider.
            step = step if value - step < lo else -step
            near, far = sample(value + step), sample(value + 2 * step)
            gradient = (4 * (near - primal) - (far - primal)) / (2 * step)
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
    signed = gradient.mean(axis=2)
    limit = float(np.quantile(np.abs(signed), .99)) or float(np.max(np.abs(signed))) or 1.
    palette = _cubicl_palette()
    normalized = .5 * (np.clip(signed / limit, -1, 1) + 1)
    bins = np.minimum((normalized * len(palette)).astype(int), len(palette) - 1)

    return dict(render=_png(np.uint8(np.clip(srgb, 0, 1) * 255)), gradient=_png(palette[bins]),
                colorbar=_colorbar(), limit=limit,
                settings=settings,
                gradient_max=float(np.max(np.abs(gradient))), mean=float(primal.mean()))
