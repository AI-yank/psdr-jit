"""Exercise the demo's real scene derivatives and local HTTP job pipeline."""

import base64
from functools import partial
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time

import drjit as dr
import numpy as np
from PIL import Image
import pytest

from psdr_jit.web import render as web_render
from psdr_jit.web.render import encode_preview, find_assets, render_arrays, validate
from psdr_jit.web.server import Handler, Jobs

ASSETS = str(Path(__file__).resolve().parents[1] / "tutorials/data")
GPU = pytest.mark.skipif(not dr.has_backend(dr.JitBackend.CUDA), reason="CUDA required")


@pytest.mark.parametrize("settings", [None, [], {"scene":"missing"}, {"spp":4096},
    {"x":float("nan")}, {"light":float("inf")}, {"seed":True}, {"seed":1.5}, {"seed":10**400},
    {"resolution":128.0}, {"roughness":0}, {"exposure":4}, {"file":"/tmp/scene"},
    {"npass":0}, {"npass":2.5}, {"npass":True}, {"npass":65537},
    {"integrator":"prb", "derivative":"x"}, {"method":"unknown"}, {"fd_step":0}, {"fd_step":float("nan")}, {"fd_step":1}])
def test_reject_invalid_settings(settings):
    with pytest.raises(ValueError):
        validate(settings)


def test_missing_assets(tmp_path):
    with pytest.raises(ValueError, match="--assets"):
        find_assets(tmp_path)


def preview_pixels(preview, key):
    png = base64.b64decode(preview[key].split(",", 1)[1])
    with Image.open(io.BytesIO(png)) as image:
        return np.array(image)


def test_cubicl_gradient_and_legend():
    signed = np.linspace(-1, 1, 5).reshape(1, 5, 1)
    gradient = np.repeat(signed, 3, axis=2)
    result = encode_preview(validate({}), np.zeros_like(gradient), gradient)
    ramp = preview_pixels(result, "colorbar")[0]
    assert ramp.shape == (256, 3)
    # Published endpoints of the supplied RGB table and the zero bin.
    np.testing.assert_array_equal(ramp[[0, 128, 255]], [[120, 0, 133], [73, 207, 126], [249, 150, 91]])
    np.testing.assert_array_equal(preview_pixels(result, "gradient")[0], ramp[[0, 64, 128, 192, 255]])


@GPU
def test_white_room_density_comparison():
    settings = validate(dict(scene="white_room", resolution=64, spp=8))
    assert (settings["method"], settings["derivative"]) == ("compare", "density")
    images = []
    for density in (0., 1., 4.):
        image, (prb, fd) = render_arrays(settings | {"density": density}, ASSETS)
        assert image.shape == prb.shape == fd.shape == (36, 64, 3)
        assert np.isfinite(image).all() and image.mean() > 0
        assert np.linalg.norm(fd) > 0
        assert np.linalg.norm(prb - fd) / np.linalg.norm(fd) < .002
        images.append(image)
    assert np.linalg.norm(images[0] - images[-1]) > 1.


@GPU
def test_fog_cube_visible_against_vacuum():
    settings = validate(dict(scene="fog_cube", resolution=64, spp=16, derivative="scattering"))
    image, gradient = render_arrays(settings, ASSETS)
    vacuum, _ = render_arrays(settings | dict(absorption=0., scattering=0.), ASSETS)
    # Above the grid there is no directly visible emitter. The two colors in
    # the center of the film come from light scattered inside the cube.
    center = np.s_[16:28, 20:44, :]
    assert vacuum[center].max() == 0
    assert image[center].mean() > .1
    assert gradient[center].mean() > 0
    left = image[20:29, 18:30].mean(axis=(0, 1))
    right = image[20:29, 34:46].mean(axis=(0, 1))
    assert left[0] > left[2] and right[2] > right[0]
    assert image[:4].max() == 0


@GPU
@pytest.mark.parametrize("absorption", [0., .3, 2.999])
def test_fd_backlit_absorber(absorption):
    settings = validate(dict(scene="volume", method="fd", derivative="absorption",
                             absorption=absorption, scattering=0., fd_step=.02, resolution=64, spp=4))
    image, gradient = render_arrays(settings, ASSETS)
    # Unit-thickness slab in front of a panel with red radiance 4.
    assert image[32, 32, 0] == pytest.approx(4 * np.exp(-absorption), rel=2e-5)
    np.testing.assert_allclose(gradient, -image, rtol=3e-4, atol=1e-6)


@GPU
@pytest.mark.parametrize("scene,integrator", [("cornell", "path"), ("bunny", "direct"), ("cornell", "prb")])
def test_fd_light_linearity(scene, integrator):
    settings = validate(dict(scene=scene, integrator=integrator, method="fd", derivative="light",
                             light=1.5, fd_step=.05, resolution=64, spp=4))
    image, gradient = render_arrays(settings, ASSETS)
    expected = image / settings["light"]
    assert np.linalg.norm(gradient - expected) < .001 * np.linalg.norm(expected)


@GPU
@pytest.mark.parametrize("scene,method,parameter", [
    ("bunny", "fd", "albedo"), ("cornell", "ad", "x"),
])
def test_surface_passes_reuse_one_scene_and_match_fresh_render(scene, method, parameter, monkeypatch):
    settings = validate(dict(scene=scene, method=method, derivative=parameter, npass=2,
                             x=.2, light=1.3, resolution=64, spp=8, seed=65535))
    expected = render_arrays(settings | {"seed": 0}, ASSETS)
    setups = []
    prepare = web_render._setup_scene

    def track_setup(settings, assets):
        setups.append(settings["scene"])
        return prepare(settings, assets)

    monkeypatch.setattr(web_render, "_setup_scene", track_setup)
    web_render.render_pass(1, settings, ASSETS, 0)
    output = web_render.render_pass(1, settings, ASSETS, 1)
    assert setups == [scene]
    for observed, reference in zip(output, expected):
        assert np.linalg.norm(observed - reference) < .002 * np.linalg.norm(reference)


def wait_for_job(jobs, identifier, first_preview=False):
    completed = 0
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        state = jobs.status(identifier, after=completed)
        completed = state["completed"]
        if state["state"] not in ("running", "stopping") or (first_preview and completed):
            return jobs.status(identifier)
    pytest.fail("Progressive renderer timed out")


@GPU
@pytest.mark.parametrize("method", ["ad", "fd"])
def test_progressive_average_and_stop(method):
    settings = validate(dict(scene="fog_cube", method=method, resolution=64, spp=4, seed=65535, npass=2))
    jobs = Jobs(ASSETS)
    try:
        identifier = jobs.submit(settings)
        result = wait_for_job(jobs, identifier)
        assert (result["state"], result["completed"]) == ("done", 2)
        first = render_arrays(settings, ASSETS)
        second = render_arrays(settings | {"seed":0}, ASSETS)
        assert not np.array_equal(first[0], second[0])
        assert not np.array_equal(first[1], second[1])
        mean = [np.mean([a, b], axis=0, dtype=np.float64) for a, b in zip(first, second)]
        expected = encode_preview(settings, *mean)
        for key in ("render", "gradient"):
            np.testing.assert_allclose(preview_pixels(result["result"], key), preview_pixels(expected, key), atol=1)
        long_job = jobs.submit(settings | {"npass":65536})
        assert jobs.stop(identifier) is None
        assert wait_for_job(jobs, long_job, first_preview=True)["state"] == "running"
        assert jobs.stop(long_job)["state"] in ("stopping", "stopped")
        result = wait_for_job(jobs, long_job)
        assert result["state"] == "stopped" and 0 < result["completed"] < 65536
        assert "result" not in jobs.status(long_job, after=result["completed"])
        fresh = wait_for_job(jobs, jobs.submit(settings | {"npass":1}))
        assert fresh["completed"] == 1
        assert fresh["result"]["mean"] == pytest.approx(first[0].mean(), rel=1e-5)
    finally:
        jobs.close()


@GPU
@pytest.mark.parametrize("scene", ["cornell", "bunny"])
@pytest.mark.parametrize("derivative", ["x", "y", "roughness", "albedo", "light"])
def test_scene_gradients(scene, derivative):
    settings = validate(dict(scene=scene, derivative=derivative, resolution=64, spp=4))
    image, gradient = render_arrays(settings, ASSETS)
    assert image.shape == gradient.shape == (64, 64, 3)
    assert np.isfinite(image).all() and image.mean() > 0
    assert np.isfinite(gradient).all() and np.max(np.abs(gradient)) > 0
    if derivative == "light":
        # Both scenes are linear in illumination. This also checks the AD connection
        # survives scene configuration, emitter sampling, and the virtual dispatch.
        np.testing.assert_allclose(gradient, image / settings["light"], rtol=2e-4, atol=2e-5)
        brighter, _ = render_arrays(settings | {"light":1.5}, ASSETS)
        # Separate scene builds and float32 sampling can perturb a few pixels.
        # Keep the same-render AD identity above tighter than this comparison.
        np.testing.assert_allclose(brighter, image * 1.5, rtol=1e-3, atol=2e-5)
        np.testing.assert_allclose(brighter.mean(), image.mean() * 1.5, rtol=5e-5)


@GPU
@pytest.mark.parametrize("scene", ["cornell", "bunny"])
def test_direct_geometry(scene):
    image, gradient = render_arrays(validate(dict(scene=scene, integrator="direct", resolution=64, spp=4)), ASSETS)
    assert image.mean() > 0 and np.max(np.abs(gradient)) > 0


@GPU
def test_http_render_pipeline():
    jobs = Jobs(ASSETS)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, jobs=jobs))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            connection.request("GET" if body is None else "POST", path,
                               None if body is None else json.dumps(body),
                               headers or {"Content-Type":"application/json"})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    try:
        assert request("/")[0] == 200
        assert request("/app.js")[0] == 200
        assert request("/style.css")[0] == 200
        assert json.loads(request("/api/config")[1])["defaults"]["scene"] == "cornell"
        assert request("/../server.py")[0] == 404
        assert request("/api/jobs/0")[0] == 404
        assert request("/api/render", {"spp":100000})[0] == 400
        assert request("/api/render", {}, {"Content-Type":"application/json", "Origin":"https://example.com"})[0] == 403
        for scene in ("cornell", "bunny", "volume", "fog_cube", "smoke"):
            parameter = "scattering" if scene in web_render.VOLUME_SCENES else "light"
            status, body = request("/api/render", dict(scene=scene, resolution=64, spp=4, derivative=parameter))
            assert status == 202
            identifier = json.loads(body)["id"]
            assert request("/api/render", {})[0] == 409
            deadline = time.monotonic() + 180
            while True:
                status, body = request(f"/api/jobs/{identifier}")
                result = json.loads(body)
                assert status == 200
                if result["state"] != "running":
                    break
                assert time.monotonic() < deadline, "GPU worker timed out"
                time.sleep(.1)
            assert result["state"] == "done", result
            output = result["result"]
            assert output["gradient_max"] > 0
            for kind in ("render", "gradient"):
                png = base64.b64decode(output[kind].split(",", 1)[1])
                with Image.open(io.BytesIO(png)) as image:
                    assert image.size == (64, 64) and image.mode == "RGB"
                    assert np.asarray(image).std() > 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        jobs.close()
