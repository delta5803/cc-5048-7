"""Stdlib HTTP server backing the browser-based inspection station."""
from __future__ import annotations

import json
import mimetypes
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from . import report as report_mod
from .disparity import dense_disparity, disparity_report, load_gray, load_mask
from .matching import match_point
from .session import SESSION_LOCK, Session
from .vertical import build_model

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# Background report-build state.  The full 30-frame recompute takes several
# seconds, so it runs off the request thread and clients poll its status.
_BUILD_LOCK = threading.Lock()
BUILD_STATE = {"running": False, "done": False, "status": None, "error": None,
               "frame": 0, "total": 0, "run_id": 0}


def _build_public() -> dict:
    return {k: BUILD_STATE[k] for k in ("running", "done", "status", "error",
                                        "frame", "total", "run_id")}


class Handler(BaseHTTPRequestHandler):
    project: Path = Path("output/footage")

    def log_message(self, fmt, *args):  # quieter console
        pass

    # -- helpers ----------------------------------------------------------
    def _send(self, code: int, payload, content_type: str = "application/json"):
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _file(self, path: Path):
        if not path.is_file():
            self._send(404, {"error": "not found"})
            return
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        return json.loads(self.rfile.read(length))

    def _session(self) -> Session:
        return Session.load(self.project)

    def _frame_meta(self, session: Session, frame: int) -> dict:
        return session.scene_info["frames"][frame]

    # -- routing ----------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        route, qs = parsed.path, parse_qs(parsed.query)
        try:
            if route in ("/", "/index.html"):
                return self._file(WEB_DIR / "index.html")
            if route.startswith("/static/"):
                return self._file(WEB_DIR / route[len("/static/"):])
            if route == "/api/state":
                return self._state()
            if route == "/api/frame":
                return self._frame_image(qs)
            if route == "/api/disparity":
                return self._frame_disparity(int(qs["frame"][0]))
            if route == "/api/report":
                return self._report_payload()
            if route == "/api/report/status":
                return self._send(200, _build_public())
            if route == "/api/report-artifact":
                return self._file(self.project / qs["path"][0])
            if route == "/api/evaluation":
                return self._file(self.project / "evaluation.json")
            self._send(404, {"error": f"unknown route {route}"})
        except Exception as exc:  # surface errors to the UI instead of 500s
            traceback.print_exc()
            self._send(500, {"error": str(exc)})

    def do_POST(self):
        route = urlparse(self.path).path
        try:
            body = self._body()
            if route == "/api/point":
                return self._add_point(body)
            if route == "/api/point/status":
                return self._set_status(body)
            if route == "/api/point/delete":
                return self._delete_point(body)
            if route == "/api/fit":
                return self._fit(body)
            if route == "/api/approve":
                return self._approve(body)
            if route == "/api/display":
                return self._set_display(body)
            if route == "/api/report/build":
                return self._build_report()
            self._send(404, {"error": f"unknown route {route}"})
        except Exception as exc:
            traceback.print_exc()
            self._send(500, {"error": str(exc)})

    # -- endpoints --------------------------------------------------------
    def _state(self):
        s = self._session()
        self._send(200, {
            "scene": s.scene_info,
            "points": [c.to_dict() for c in s.correspondences],
            "model": s.model.to_dict() if s.model else None,
            "approved_snapshot": s.approved_snapshot,
            "display": s.display,
        })

    def _report_payload(self):
        path = self.project / "reports" / "report.json"
        if not path.is_file():
            # Distinct state, not an HTTP error, so the UI can poll quietly
            # before any report has been built.
            self._send(200, {"status": "not_generated"})
            return
        self._send(200, json.loads(path.read_text()))

    def _frame_image(self, qs):
        s = self._session()
        meta = self._frame_meta(s, int(qs["frame"][0]))
        eye = qs.get("eye", ["left"])[0]
        kind = qs.get("kind", ["image"])[0]
        key = {"left": "left", "right": "right"}[eye]
        if kind == "valid":
            key = "valid_left" if eye == "left" else "valid_right"
        elif kind == "repeated":
            key = "repeated"
        elif kind == "mono":
            key = "mono"
        self._file(self.project / meta[key])

    def _add_point(self, body):
        with SESSION_LOCK:
            s = self._session()
            f = int(body["frame"])
            meta = self._frame_meta(s, f)
            left = np.asarray(Image.open(self.project / meta["left"]).convert("RGB"))
            right = np.asarray(Image.open(self.project / meta["right"]).convert("RGB"))
            vl = load_mask(self.project / meta["valid_left"])
            vr = load_mask(self.project / meta["valid_right"])
            c = match_point(left, right, float(body["x"]), float(body["y"]), f,
                            vl, vr, source=body.get("source", "manual"))
            s.add_point(c)
            s.save()
        self._send(200, {"point": c.to_dict()})

    def _set_status(self, body):
        with SESSION_LOCK:
            s = self._session()
            s.set_status(int(body["index"]), body["status"], body.get("note", ""))
            s.save()
        self._send(200, {"ok": True})

    def _delete_point(self, body):
        with SESSION_LOCK:
            s = self._session()
            s.remove_point(int(body["index"]))
            s.save()
        self._send(200, {"ok": True})

    def _fit(self, body):
        with SESSION_LOCK:
            s = self._session()
            scope = body.get("scope", "global")
            shots = [tuple(x) for x in s.scene_info["shots"]]
            s.model = build_model(
                s.correspondences, scope,
                s.scene_info["frame_count"], s.scene_info["width"], shots,
            )
            s.approved_snapshot = None  # stale approval invalidated by a new fit
            s.save()
        self._send(200, {"model": s.model.to_dict()})

    def _approve(self, body):
        with SESSION_LOCK:
            s = self._session()
            if body.get("approved"):
                s.approve_model()
            else:
                s.revoke_approval()
            s.save()
            model = s.model.to_dict() if s.model else None
            snapshot = s.approved_snapshot
        self._send(200, {"model": model, "approved_snapshot": snapshot})

    def _set_display(self, body):
        with SESSION_LOCK:
            s = self._session()
            for key in ("width_px", "height_px", "width_mm", "viewing_distance_mm"):
                if key in body:
                    val = body[key]
                    s.display[key] = float(val) if val not in (None, "") else None
            s.save()
        self._send(200, {"display": s.display})

    def _frame_disparity(self, frame: int):
        s = self._session()
        approved = s.approved_model()
        meta = self._frame_meta(s, frame)
        gl = load_gray(self.project / meta["left"])
        gr = load_gray(self.project / meta["right"])
        vl = load_mask(self.project / meta["valid_left"])
        vr = load_mask(self.project / meta["valid_right"])
        dx, conf, reason = dense_disparity(gl, gr, frame, approved, vl, vr)
        payload = {
            "frame": frame,
            "based_on_approved": approved is not None,
            "stats": disparity_report(dx),
            "reason_counts": {
                "valid": int(np.sum(reason == 0)),
                "occlusion": int(np.sum(reason == 1)),
                "border": int(np.sum(reason == 2)),
                "low_confidence_blank": int(np.sum(reason == 3)),
            },
        }
        self._send(200, payload)

    def _build_report(self):
        # Reject a second concurrent build; the client polls instead.
        if BUILD_STATE["running"]:
            self._send(202, {"queued": True, **_build_public()})
            return
        run_id = BUILD_STATE["run_id"] + 1
        BUILD_STATE.update(running=True, done=False, status="starting",
                           error=None, frame=0, run_id=run_id,
                           total=Session.load(self.project).scene_info["frame_count"])

        def worker():
            try:
                # Snapshot the session under the lock, then run the slow
                # multi-frame compute without blocking reviewer requests.
                with SESSION_LOCK:
                    session = Session.load(self.project)
                def cb(done, total, stage):
                    BUILD_STATE.update(frame=done, total=total, status=stage)
                report_mod.build_report(session, progress=cb)
                BUILD_STATE.update(running=False, done=True, status="ok")
            except Exception as exc:  # surfaced to the polling client
                BUILD_STATE.update(running=False, done=True, status="error",
                                   error=str(exc))

        threading.Thread(target=worker, daemon=True).start()
        self._send(202, {"queued": True, "run_id": run_id, **_build_public()})


def serve(project: str | Path, port: int = 8765) -> None:
    Handler.project = Path(project).resolve()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"inspection station on http://127.0.0.1:{port}  (project {Handler.project})")
    httpd.serve_forever()
