"""Independent GPU reference for test_volume.py. Run with Mitsuba's Python."""

import sys

import drjit as dr
import mitsuba as mi
import numpy as np


def render(size=32, spp=256, batches=12, density=None):
    mi.set_variant("cuda_ad_rgb")
    transform = mi.ScalarTransform4f
    medium = {"type": "homogeneous", "sigma_t": 1.1, "albedo": .8 / 1.1}
    if density is not None:
        # Align Mitsuba texel centers with PSDR grid vertices at the box endpoints.
        resolution = np.array(density.shape[::-1])
        extent = np.array([1.5, 1.5, 1.])
        spacing = extent / (resolution - 1)
        origin = np.array([-.75, -.75, 1.]) - spacing / 2
        medium = {"type": "heterogeneous", "albedo": .8 / 1.1,
                  "sigma_t": {"type": "gridvolume", "grid": mi.VolumeGrid(density * 1.1),
                              "to_world": transform.translate(origin) @ transform.scale(extent + spacing),
                              "accel": False}}
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "prbvolpath", "max_depth": 4, "rr_depth": 8},
        "sensor": {"type": "orthographic", "near_clip": .01,
                   "film": {"type": "hdrfilm", "width": size, "height": size,
                            "rfilter": {"type": "box"}}},
        "fog": {"type": "cube", "to_world": transform.translate([0, 0, 1.5]) @
                transform.scale([.75, .75, .5]), "bsdf": {"type": "null"},
                "interior": medium},
        "panel": {"type": "rectangle", "to_world": transform.translate([0, 0, 2.5]) @
                  transform.rotate([0, 1, 0], 180) @ transform.scale([.25, .5, 1]),
                  "bsdf": {"type": "diffuse", "reflectance": 0.},
                  "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [4, 2, 1]}}},
    })
    params = mi.traverse(scene)
    images, gradients, grid_gradients = [], [], []
    # The loss measures the scattered halo. Direct transmission to the panel
    # cannot mask a missing or wrongly differentiated in-scattering term.
    y, x = np.mgrid[:size, :size] + .5
    halo = ((abs(x - size / 2) < size * .375) & (abs(y - size / 2) < size * .375) &
            ((abs(x - size / 2) > size * .125) | (abs(y - size / 2) > size * .25)))
    weights = mi.TensorXf(np.repeat(halo[..., None], 3, axis=2).astype("float32"))
    for seed in range(batches):
        a, s = mi.Float(.3), mi.Float(.8)
        dr.enable_grad(a, s)
        if density is None:
            params["fog.interior_medium.sigma_t.value.value"] = a + s
        else:
            rho = mi.TensorXf(density[..., None])
            dr.enable_grad(rho)
            params["fog.interior_medium.sigma_t.data"] = rho * (a + s)
        params["fog.interior_medium.albedo.value.value"] = s / (a + s)
        params.update()
        image = mi.render(scene, params=params, seed=17 + seed, spp=spp)
        dr.backward(dr.sum(image * weights) / (int(halo.sum()) * 3))
        images.append(np.array(image))
        gradients.append([float(dr.grad(a)[0]), float(dr.grad(s)[0])])
        if density is not None:
            grid_gradients.append(np.array(dr.grad(rho))[..., 0])
    return dict(images=np.stack(images), gradients=np.array(gradients), halo=halo,
                grid_gradients=np.array(grid_gradients))


if __name__ == "__main__":
    np.savez(sys.argv[1], **render(density=np.load(sys.argv[2]) if len(sys.argv) > 2 else None))
