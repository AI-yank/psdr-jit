"""Independent GPU reference for test_volume.py. Run with Mitsuba's Python."""

import sys

import drjit as dr
import mitsuba as mi
import numpy as np


def render(size=32, spp=256, batches=12):
    mi.set_variant("cuda_ad_rgb")
    transform = mi.ScalarTransform4f
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "prbvolpath", "max_depth": 4, "rr_depth": 8},
        "sensor": {"type": "orthographic", "near_clip": .01,
                   "film": {"type": "hdrfilm", "width": size, "height": size,
                            "rfilter": {"type": "box"}}},
        "fog": {"type": "cube", "to_world": transform.translate([0, 0, 1.5]) @
                transform.scale([.75, .75, .5]), "bsdf": {"type": "null"},
                "interior": {"type": "homogeneous", "sigma_t": 1.1, "albedo": .8 / 1.1}},
        "panel": {"type": "rectangle", "to_world": transform.translate([0, 0, 2.5]) @
                  transform.rotate([0, 1, 0], 180) @ transform.scale([.25, .5, 1]),
                  "bsdf": {"type": "diffuse", "reflectance": 0.},
                  "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [4, 2, 1]}}},
    })
    params = mi.traverse(scene)
    images, gradients = [], []
    # The loss measures the scattered halo. Direct transmission to the panel
    # cannot mask a missing or wrongly differentiated in-scattering term.
    y, x = np.mgrid[:size, :size] + .5
    halo = ((abs(x - size / 2) < size * .375) & (abs(y - size / 2) < size * .375) &
            ((abs(x - size / 2) > size * .125) | (abs(y - size / 2) > size * .25)))
    weights = mi.TensorXf(np.repeat(halo[..., None], 3, axis=2).astype("float32"))
    for seed in range(batches):
        a, s = mi.Float(.3), mi.Float(.8)
        dr.enable_grad(a, s)
        params["fog.interior_medium.sigma_t.value.value"] = a + s
        params["fog.interior_medium.albedo.value.value"] = s / (a + s)
        params.update()
        image = mi.render(scene, params=params, seed=17 + seed, spp=spp)
        dr.backward(dr.sum(image * weights) / (int(halo.sum()) * 3))
        images.append(np.array(image))
        gradients.append([float(dr.grad(a)[0]), float(dr.grad(s)[0])])
    return dict(images=np.stack(images), gradients=np.array(gradients), halo=halo)


if __name__ == "__main__":
    np.savez(sys.argv[1], **render())
