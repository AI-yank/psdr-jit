"""Diffuse adaptation of Jay-Artist's White Room from Bitterli's scene collection."""

from pathlib import Path
import xml.etree.ElementTree as ET


PLUME_BOUNDS = ((-.5, .25, 2.3), (1.15, 2.15, 3.95))


def white_room_scene(assets, width, spp, light_gain):
    import numpy as np
    from PIL import Image
    from drjit import cuda
    import psdr_jit as psdr

    directory = Path(assets) / "living-room-2"
    xml = directory / "scene_v3.xml"
    if not xml.is_file():
        raise ValueError(f"Extract the White Room Mitsuba archive into {directory.parent} "
                         "as described in tutorials/WEB_DEMO.md.")
    root = ET.parse(xml).getroot()

    def transform(node):
        values = np.fromstring(node.find("transform/matrix").attrib["value"], sep=" ")
        return cuda.ad.Matrix4f(values.reshape(4, 4).tolist())

    def rgb(node):
        return np.fromstring(node.attrib["value"], sep=",")

    scene = psdr.Scene()
    scene.opts = psdr.RenderOption(width, width * 9 // 16, spp, 0, 0)
    scene.opts.log_level = 0
    sensor = root.find("sensor")
    camera = psdr.PerspectiveCamera(float(sensor.find("float[@name='fov']").attrib["value"]), .01, 1000.)
    camera.to_world = transform(sensor)
    scene.add_Sensor(camera)
    scene.add_BSDF(psdr.DiffuseBSDF([0.] * 3), "black")
    for material in root.findall("bsdf"):
        name = material.attrib["id"]
        prop = material.find(".//*[@name='reflectance']")
        if prop is None:
            prop = material.find(".//*[@name='diffuse_reflectance']")
        if prop is None:
            gray = .035 if name in ("BlackMarbleBSDF", "TvScreenBSDF") else .32
            bsdf = psdr.DiffuseBSDF([gray] * 3)
        elif prop.tag == "texture":
            with Image.open(directory / prop.find("string[@name='filename']").attrib["value"]) as texture:
                pixels = np.asarray(texture.convert("RGB"), dtype=np.float32) / 255
            linear = np.where(pixels < .04045, pixels / 12.92, ((pixels + .055) / 1.055) ** 2.4)
            data = cuda.ad.Array3f(np.ascontiguousarray(linear.reshape(-1, 3).T))
            bsdf = psdr.DiffuseBSDF(psdr.Bitmap3fD(linear.shape[1], linear.shape[0], data))
        else:
            bsdf = psdr.DiffuseBSDF(rgb(prop).tolist())
        scene.add_BSDF(bsdf, name, True)

    for shape in root.findall("shape"):
        mesh = psdr.Mesh()
        mesh.enable_edges = False
        mesh.use_face_normal = True
        if shape.attrib["type"] == "rectangle":
            mesh.load_raw(cuda.Array3f([[-1, 1, 1, -1], [-1, -1, 1, 1], [0] * 4]),
                          cuda.Array3i([[0, 0], [1, 2], [2, 3]]))
        else:
            mesh.load(str(directory / shape.find("string[@name='filename']").attrib["value"]))
        mesh.to_world = transform(shape)
        material = shape.find("ref")
        radiance = shape.find("emitter/rgb[@name='radiance']")
        emitter = psdr.AreaLight((rgb(radiance) * light_gain).tolist()) if radiance is not None else None
        scene.add_Mesh(mesh, material.attrib["id"] if material is not None else "black", emitter)
    scene.configure()
    return scene
