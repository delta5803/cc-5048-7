"""Display-frame bookkeeping.

Reviewers may switch the monitoring window between native pixels and a
different display aspect.  Switching only changes how coordinates map to the
screen; every stored correspondence stays in original-image pixel space, and
all ratios are recomputed on demand.  Viewing-angle conversion is refused
until real physical dimensions and a viewing distance are supplied.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DisplayGeometry:
    native_w: int
    native_h: int
    shown_w: int | None = None     # CSS/device pixels occupied by one eye
    shown_h: int | None = None
    width_mm: float | None = None
    viewing_distance_mm: float | None = None

    def effective(self) -> tuple[int, int, float]:
        """Return (w, h, screen-to-native scale) without stretching the image."""
        if not self.shown_w or not self.shown_h:
            return self.native_w, self.native_h, 1.0
        native_ar = self.native_w / self.native_h
        shown_ar = self.shown_w / self.shown_h
        if shown_ar > native_ar:
            h = self.shown_h
            w = round(h * native_ar)
        else:
            w = self.shown_w
            h = round(w / native_ar)
        return w, h, w / self.native_w

    def to_native(self, sx: float, sy: float) -> tuple[float, float]:
        w, h, scale = self.effective()
        return sx / scale, sy / scale

    def to_screen(self, nx: float, ny: float) -> tuple[float, float]:
        _, _, scale = self.effective()
        return nx * scale, ny * scale

    def can_compute_angle(self) -> bool:
        return bool(self.width_mm and self.viewing_distance_mm and self.shown_w)

    def mm_per_native_pixel(self) -> float | None:
        if not self.can_compute_angle():
            return None
        w, _, _ = self.effective()
        return self.width_mm / w * (w / self.native_w)

    def to_dict(self) -> dict:
        return {
            "width_px": self.shown_w,
            "height_px": self.shown_h,
            "width_mm": self.width_mm,
            "viewing_distance_mm": self.viewing_distance_mm,
        }
