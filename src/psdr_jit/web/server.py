"""Loopback HTTP server with a bounded, isolated GPU worker."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit

from .render import DEFAULTS, encode_preview, find_assets, release_worker_scene, render_pass, validate


class Jobs:
    def __init__(self, assets):
        self.assets = assets
        self.changed = threading.Condition()
        self.thread = None
        self.closing = False
        self.state = "idle"
        self.identifier = 0
        self.pool = self.make_pool()

    @staticmethod
    def make_pool():
        return ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))

    def submit(self, settings):
        with self.changed:
            if self.closing or self.state in ("running", "stopping"):
                return None
            self.identifier += 1
            self.state = "running"
            self.settings = settings
            self.completed = 0
            self.preview = None
            self.error = None
            self.thread = threading.Thread(target=self._run, args=(settings,))
            self.thread.start()
            self.changed.notify_all()
            return self.identifier

    def active_job(self):
        with self.changed:
            if self.state in ("running", "stopping"):
                return dict(id=self.identifier, settings=self.settings)
            return None

    def _run(self, settings):
        started = time.perf_counter()
        try:
            for i in range(settings["npass"]):
                with self.changed:
                    if self.state == "stopping":
                        break
                try:
                    future = self.pool.submit(render_pass, self.identifier, settings, self.assets, i)
                except BrokenProcessPool:
                    self.pool.shutdown(wait=False, cancel_futures=True)
                    self.pool = self.make_pool()
                    future = self.pool.submit(render_pass, self.identifier, settings, self.assets, i)
                image, gradient = future.result()
                if i == 0:
                    image_sum = image.astype("float64")
                    gradient_sum = gradient.astype("float64")
                else:
                    image_sum += image
                    gradient_sum += gradient
                completed = i + 1
                preview = encode_preview(settings, image_sum / completed, gradient_sum / completed)
                preview["seconds"] = round(time.perf_counter() - started, 3)
                with self.changed:
                    self.preview, self.completed = preview, completed
                    if completed == settings["npass"]:
                        self.state = "done"
                    self.changed.notify_all()
                    if self.state == "done":
                        return
            if self.completed:
                self.pool.submit(release_worker_scene).result()
            with self.changed:
                self.state = "stopped"
                self.changed.notify_all()
        except Exception as error:
            with self.changed:
                self.state = "error"
                self.error = f"{type(error).__name__}: {error}"
                self.changed.notify_all()

    def status(self, identifier, after=None):
        with self.changed:
            if after is not None:
                self.changed.wait_for(lambda: identifier != self.identifier or self.state not in ("running", "stopping")
                                      or self.completed > after, timeout=2)
            if identifier != self.identifier or self.thread is None:
                return None
            data = dict(state=self.state, completed=self.completed)
            if self.preview is not None and (after is None or self.completed > after):
                data["result"] = self.preview
            if self.error:
                data["error"] = self.error
            return data

    def stop(self, identifier):
        with self.changed:
            if identifier != self.identifier or self.thread is None:
                return None
            if self.state == "running":
                self.state = "stopping"
                self.changed.notify_all()
            return dict(state=self.state)

    def close(self):
        with self.changed:
            self.closing = True
            if self.state == "running":
                self.state = "stopping"
            self.changed.notify_all()
        if self.thread:
            self.thread.join()
        self.pool.shutdown(wait=True, cancel_futures=True)


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, jobs, **kwargs):
        self.jobs = jobs
        super().__init__(*args, **kwargs)

    def log_message(self, *_args):
        pass

    def respond(self, status, body, content_type="application/json"):
        if content_type == "application/json":
            body = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local_request(self):
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        return host in allowed and (origin is None or origin == f"http://{host}")

    def do_GET(self):
        if not self.local_request():
            return self.respond(403, dict(error="Local requests only"))
        url = urlsplit(self.path)
        path = url.path
        if path == "/api/config":
            return self.respond(200, dict(defaults=DEFAULTS, active=self.jobs.active_job()))
        if path.startswith("/api/jobs/"):
            try:
                after = parse_qs(url.query).get("after", [None])[0]
                result = self.jobs.status(int(path.rsplit("/", 1)[1]), int(after) if after is not None else None)
            except ValueError:
                result = None
            return self.respond(200 if result else 404, result or dict(error="Job not found"))
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        if path not in files:
            return self.respond(404, dict(error="Not found"))
        name, content_type = files[path]
        return self.respond(200, (Path(__file__).parent / "static" / name).read_bytes(), content_type)

    def do_POST(self):
        if not self.local_request():
            return self.respond(403, dict(error="Local requests only"))
        parts = urlsplit(self.path).path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "stop":
            try:
                result = self.jobs.stop(int(parts[2]))
            except ValueError:
                result = None
            return self.respond(200 if result else 404, result or dict(error="Job not found"))
        if self.path != "/api/render":
            return self.respond(404, dict(error="Not found"))
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096 or self.headers.get_content_type() != "application/json":
                raise ValueError("Expected a small JSON settings object")
            settings = validate(json.loads(self.rfile.read(size)))
        except (ValueError, TypeError, UnicodeError) as error:
            return self.respond(400, dict(error=str(error)))
        identifier = self.jobs.submit(settings)
        if identifier is None:
            return self.respond(409, dict(error="GPU busy; wait for the current render"))
        return self.respond(202, dict(id=identifier))


def main():
    parser = argparse.ArgumentParser(description="Local psdr-jit GPU demo")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--assets", help="Path to the repository's tutorials/data directory")
    args = parser.parse_args()
    try:
        assets = find_assets(args.assets)
    except ValueError as error:
        parser.error(str(error))
    jobs = Jobs(assets)
    try:
        with ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, jobs=jobs)) as server:
            print(f"psdr-jit: http://127.0.0.1:{server.server_port}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        jobs.close()


if __name__ == "__main__":
    main()
