"""Horizontal disparity statistics and boundary conflict maps.

Every number reported here is recomputed from the *currently approved*
vertical correction; stale fits are never reused.  Pixels whose match is
unreliable, occluded in the other eye, or cut by the frame border are stored
as NaN and rendered blank - they are never coerced to zero disparity.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .matching import to_gray
from .vertical import CorrectionModel

HALF_H, HALF_V = 10, 5
MAX_DX = 34
BLANK = np.nan


def _warp_vertical(gray: np.ndarray, frame: int, model: CorrectionModel) -> np.ndarray:
    """Correct right view: move rows back by -dy(x), linear interpolation."""
    h, w = gray.shape
    ys = np.arange(h, dtype=np.float64)
    out = np.empty_like(gray)
    for x in range(w):
        s = ys - model.dy(frame, float(x))
        y0 = np.floor(s).astype(int)
        y1 = np.clip(y0 + 1, 0, h - 1)
        y0c = np.clip(y0, 0, h - 1)
        t = (s - y0)
        out[:, x] = gray[y0c, x] * (1 - t) + gray[y1, x] * t
    return out


def _warp_mask_vertical(mask: np.ndarray, frame: int, model: CorrectionModel) -> np.ndarray:
    h, w = mask.shape
    ys = np.arange(h)
    out = np.zeros_like(mask)
    for x in range(w):
        src = np.clip(np.rint(ys - model.dy(frame, float(x))).astype(int), 0, h - 1)
        out[:, x] = mask[src, x]
    return out


def load_gray(path: str | Path) -> np.ndarray:
    return to_gray(np.asarray(Image.open(path).convert("RGB")))


def load_mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def dense_disparity(
    left: np.ndarray,
    right: np.ndarray,
    frame: int,
    model: CorrectionModel | None,
    valid_left: np.ndarray | None = None,
    valid_right: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dx_corrected, confidence 0..1, blank_reason codes).

    Reason codes: 0 valid, 1 occlusion/monocular, 2 border, 3 low confidence.
    """
    gl, gr = to_gray(left), to_gray(right)
    h, w = gl.shape
    if model is not None and model.approved:
        # Recompute the right view under the approved correction first, so the
        # reported statistics genuinely derive from that transform.
        gr = _warp_vertical(gr, frame, model)
        if valid_right is not None:
            valid_right = _warp_mask_vertical(valid_right, frame, model)
    dx = np.full((h, w), BLANK, dtype=np.float32)
    conf = np.zeros((h, w), dtype=np.float32)
    reason = np.zeros((h, w), dtype=np.uint8)

    costs = np.full((h, w, MAX_DX + 1), np.inf, dtype=np.float32)
    for d in range(0, MAX_DX + 1):
        if d == 0:
            r = gr
        else:
            r = np.pad(gr, ((0, 0), (d, 0)), mode="edge")[:, :w]
        # Sub-pixel vertical resampling approximated by row roll per column is
        # too costly; vertical shifts here are <= ~3 px so integer rounding is
        # adequate for the statistics, and point tools keep sub-pixel values.
        diff = np.abs(gl - r)
        # Box filter via cumulative sums for a (2*HALF_H+1) x (2*HALF_V+1) SAD.
        cs = np.pad(diff, ((HALF_V, HALF_V), (HALF_H, HALF_H)), mode="edge").cumsum(0).cumsum(1)
        sad = (
            cs[2 * HALF_V:, 2 * HALF_H:]
            - cs[:-2 * HALF_V, 2 * HALF_H:]
            - cs[2 * HALF_V:, :-2 * HALF_H]
            + cs[:-2 * HALF_V, :-2 * HALF_H]
        )
        costs[..., d] = sad

    order = np.argsort(costs, axis=2)
    best = order[..., 0]
    second = order[..., 1]
    best_cost = np.take_along_axis(costs, best[..., None], axis=2)[..., 0]
    second_cost = np.take_along_axis(costs, second[..., None], axis=2)[..., 0]
    margin = (second_cost - best_cost) / (second_cost + 1e-6)

    yy, xx = np.mgrid[0:h, 0:w]
    d_int = best.astype(np.float32)
    # Sub-pixel refinement on the SAD parabola.
    for d in range(1, MAX_DX):
        sel = (best == d)
        if not sel.any():
            continue
        c0, c1, c2 = costs[..., d - 1], costs[..., d], costs[..., d + 1]
        denom = (c0 - 2 * c1 + c2)
        sub = np.where(np.abs(denom) > 1e-6,
                       0.5 * (c0 - c2) / np.where(np.abs(denom) > 1e-6, denom, 1.0),
                       0.0)
        d_int = np.where(sel, d + np.clip(sub, -1, 1), d_int)

    border = (xx < HALF_H + 2) | (xx >= w - HALF_H - 2) | (yy < HALF_V + 2) | (yy >= h - HALF_V - 2)
    low = margin < 0.08
    if valid_left is not None:
        occ = ~valid_left
    else:
        occ = np.zeros((h, w), dtype=bool)
    if valid_right is not None:
        rx = np.clip(xx - best, 0, w - 1)
        occ = occ | ~valid_right[yy, rx]

    valid = ~(border | low | occ)
    dx[valid] = d_int[valid]
    conf[valid] = np.clip(margin[valid] / 0.25, 0, 1)
    reason[occ] = 1
    reason[border & ~occ] = 2
    reason[low & ~occ & ~border] = 3
    return dx, conf, reason


def disparity_report(dx: np.ndarray) -> dict:
    vals = dx[np.isfinite(dx)]
    if vals.size == 0:
        return {"valid_pixels": 0, "blank_pixels": int(dx.size), "note": "entire frame blanked"}
    pct = np.percentile(vals, [0, 2, 50, 98, 100])
    return {
        "valid_pixels": int(vals.size),
        "blank_pixels": int(dx.size - vals.size),
        "blank_fraction": float(1 - vals.size / dx.size),
        "min_px": float(pct[0]),
        "p02_px": float(pct[1]),
        "median_px": float(pct[2]),
        "p98_px": float(pct[3]),
        "max_px": float(pct[4]),
        "range_px": float(pct[4] - pct[0]),
        "robust_range_px": float(pct[3] - pct[1]),
    }


def render_conflict_overlay(
    base_rgb: np.ndarray, reason: np.ndarray, dx: np.ndarray
) -> np.ndarray:
    """Visualise border/occlusion conflicts on a copy of the frame."""
    out = base_rgb.copy()
    red = np.array([235, 64, 52], dtype=np.uint8)
    amber = np.array([240, 170, 40], dtype=np.uint8)
    out[reason == 1] = (0.55 * out[reason == 1] + 0.45 * red).astype(np.uint8)
    out[reason == 2] = (0.5 * out[reason == 2] + 0.5 * amber).astype(np.uint8)
    # Low confidence stays blank-ish: desaturate rather than invent a value.
    low = reason == 3
    gray = out[low].mean(axis=1, keepdims=True)
    out[low] = (0.25 * out[low] + 0.75 * gray).astype(np.uint8)
    return out


def render_disparity_map(dx: np.ndarray) -> np.ndarray:
    """Colour map with NaN rendered as neutral grey (never as zero colour)."""
    h, w = dx.shape
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    finite = np.isfinite(dx)
    if finite.any():
        lo, hi = np.nanmin(dx), np.nanmax(dx)
        scale = np.clip((dx - lo) / (hi - lo + 1e-9), 0, 1)
        r = np.clip(255 * np.nan_to_num(scale, nan=0.0), 0, 255)
        b = np.clip(255 * np.nan_to_num(1 - scale, nan=0.0), 0, 255)
        g = np.clip(40 + 60 * (1 - np.abs(np.nan_to_num(scale, nan=0.5) - 0.5) * 2), 0, 255)
        stacked = np.stack([r, g, b], axis=2).astype(np.uint8)
        img[finite] = stacked[finite]
    return img


def angular_range(report: dict, display_width_px: int, viewing_distance_mm: float,
                  display_width_mm: float) -> dict | None:
    """Convert a pixel disparity range to viewing angle in degrees.

    Only available when the reviewer supplies real display geometry; otherwise
    the station declines to invent an angle.
    """
    if not display_width_mm or not viewing_distance_mm or not display_width_px:
        return None
    mm_per_px = display_width_mm / display_width_px
    lo = report["p02_px"] * mm_per_px
    hi = report["p98_px"] * mm_per_px
    return {
        "robust_range_deg": math.degrees(math.atan(hi / viewing_distance_mm)
                                         - math.atan(lo / viewing_distance_mm)),
        "min_deg": math.degrees(math.atan(report["min_px"] * mm_per_px / viewing_distance_mm)),
        "max_deg": math.degrees(math.atan(report["max_px"] * mm_per_px / viewing_distance_mm)),
    }
