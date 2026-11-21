"""Persisted review session: points, fits, approval and display geometry."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .geometry import Correspondence
from .vertical import CorrectionModel

_WRITE_LOCK = threading.Lock()
# Serialises the whole load -> mutate -> save cycle in the threaded server,
# otherwise concurrent reviewer clicks can interleave and corrupt the file.
SESSION_LOCK = threading.RLock()


class Session:
    def __init__(self, project_dir: str | Path, scene_info: dict):
        self.dir = Path(project_dir)
        self.scene_info = scene_info
        self.correspondences: list[Correspondence] = []
        self.model: CorrectionModel | None = None
        # Frozen snapshot of the transform the report was actually built from.
        self.approved_snapshot: dict | None = None
        self.display: dict = {
            # Displayed pixel size of one eye image; None means native pixels.
            "width_px": None,
            "height_px": None,
            # Physical display dimensions and viewing distance in millimetres.
            "width_mm": None,
            "viewing_distance_mm": None,
        }
        self.path = self.dir / "session.json"

    @classmethod
    def load(cls, project_dir: str | Path) -> "Session":
        d = Path(project_dir)
        scene_info = json.loads((d / "scene_info.json").read_text())
        s = cls(d, scene_info)
        if s.path.exists():
            raw = json.loads(s.path.read_text())
            s.correspondences = [Correspondence.from_dict(p) for p in raw.get("points", [])]
            if raw.get("model"):
                s.model = CorrectionModel.from_dict(raw["model"])
            s.approved_snapshot = raw.get("approved_snapshot")
            s.display.update(raw.get("display", {}))
        return s

    def save(self) -> None:
        raw = {
            "points": [c.to_dict() for c in self.correspondences],
            "model": self.model.to_dict() if self.model else None,
            "approved_snapshot": self.approved_snapshot,
            "display": self.display,
        }
        with _WRITE_LOCK:
            tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(raw, indent=2, allow_nan=False))
            os.replace(tmp, self.path)

    def frame_points(self, frame: int) -> list[Correspondence]:
        return [c for c in self.correspondences if c.frame == frame]

    def add_point(self, c: Correspondence) -> None:
        self.correspondences.append(c)

    def remove_point(self, index: int) -> None:
        self.correspondences.pop(index)

    def set_status(self, index: int, status: str, note: str = "") -> None:
        old = self.correspondences[index]
        self.correspondences[index] = Correspondence(
            frame=old.frame, left=old.left, right=old.right,
            confidence=old.confidence, status=status, source=old.source,
            note=note or old.note,
        )

    def approve_model(self) -> None:
        if self.model is None:
            raise ValueError("no model fitted yet")
        self.model.approved = True
        self.approved_snapshot = self.model.to_dict()

    def revoke_approval(self) -> None:
        if self.model is not None:
            self.model.approved = False
        self.approved_snapshot = None

    def approved_model(self) -> CorrectionModel | None:
        # The frozen snapshot is the single source of truth for what the
        # report may use.  A new fit clears the snapshot, so a stale approval
        # from a previous scope can never reach the report.
        if self.approved_snapshot:
            return CorrectionModel.from_dict(self.approved_snapshot)
        return None
