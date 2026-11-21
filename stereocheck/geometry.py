"""Basic 2D geometry helpers for the stereo inspection station."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def as_list(self) -> list[float]:
        return [self.x, self.y]


@dataclass(frozen=True)
class Correspondence:
    """A point pair between left and right view of one frame.

    ``status`` is one of:
      ok       - confident match, usable for fitting
      low      - low confidence, kept but blanked in maps and excluded from fits
      conflict - boundary/occlusion conflict, excluded from fits
    """

    frame: int
    left: Point
    right: Point
    confidence: float
    status: str = "ok"
    source: str = "auto"
    note: str = ""

    @property
    def dx(self) -> float:
        return self.left.x - self.right.x

    @property
    def dy(self) -> float:
        return self.right.y - self.left.y

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "left": self.left.as_list(),
            "right": self.right.as_list(),
            "dx": self.dx,
            "dy": self.dy,
            "confidence": self.confidence,
            "status": self.status,
            "source": self.source,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Correspondence":
        return cls(
            frame=int(d["frame"]),
            left=Point(float(d["left"][0]), float(d["left"][1])),
            right=Point(float(d["right"][0]), float(d["right"][1])),
            confidence=float(d["confidence"]),
            status=d.get("status", "ok"),
            source=d.get("source", "auto"),
            note=d.get("note", ""),
        )


def shift_right_view(points_right: list[Point], dy_func) -> list[Point]:
    """Apply a vertical correction to right-view points: y_corr = y - dy(x)."""
    return [Point(p.x, p.y - dy_func(p.x)) for p in points_right]
