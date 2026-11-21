"""Stereo template matching on grayscale image arrays.

The matcher is deliberately local and explainable for a review tool: the
reviewer clicks a point in one eye, an NCC template search locates it in the
other eye, and every result carries a confidence plus the reason it was
blanked.  Nothing is silently forced to zero disparity.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import Correspondence, Point

WINDOW = 9          # template half-width -> 19x19 window
SEARCH_Y = 14       # vertical search radius (true drifts stay inside this)
SEARCH_X = 40       # maximum positive horizontal disparity (left minus right)
LOW_CONF = 0.35     # below this NCC the point is blanked
LR_THRESHOLD = 1.5  # px: left-right consistency tolerance


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float64)
    w = np.array([0.299, 0.587, 0.114])
    return img[..., :3].astype(np.float64) @ w


def _ncc_search(
    src: np.ndarray, dst: np.ndarray, cx: int, cy: int,
    sx_min: int, sx_max: int, sy_min: int, sy_max: int,
    win: int = WINDOW,
) -> tuple[float, float, float, np.ndarray]:
    """Search dst for the src template centred at (cx, cy).

    Returns (best_x, best_y, peak_ncc, score_map).
    """
    h, w = src.shape
    tpl_raw = src[cy - win:cy + win + 1, cx - win:cx + win + 1]
    tpl_std = float(tpl_raw.std())
    tpl = tpl_raw - tpl_raw.mean()
    tpl_norm = math.sqrt(float((tpl**2).sum()))
    if tpl_norm < 1e-9:
        return float(cx), float(cy), 0.0, np.zeros((1, 1)), 1.0, tpl_std
    tpl = tpl / tpl_norm

    x0, x1 = max(win, sx_min), min(w - win - 1, sx_max)
    y0, y1 = max(win, sy_min), min(h - win - 1, sy_max)
    scores = np.full((sy_max - sy_min + 1, sx_max - sx_min + 1), np.nan)
    best = (-2.0, cx, cy)
    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            patch = dst[yy - win:yy + win + 1, xx - win:xx + win + 1]
            p = patch - patch.mean()
            pn = math.sqrt(float((p**2).sum()))
            if pn < 1e-9:
                continue
            ncc = float((tpl * (p / pn)).sum())
            scores[yy - sy_min, xx - sx_min] = ncc
            if ncc > best[0]:
                best = (ncc, xx, yy)
    peak, bx, by = best
    # Generic ambiguity signals that NCC alone cannot express:
    #  - periodic texture produces several near-equal peaks (railing/tile)
    #  - flat texture has very low local variance and no real gradient to lock
    finite = scores[np.isfinite(scores)]
    if finite.size:
        ordered = np.sort(finite)[::-1]
        second_peak = ordered[1] if ordered.size > 1 else peak
        peak_ratio = second_peak / peak if peak > 1e-9 else 1.0
    else:
        peak_ratio = 1.0
    # Parabolic sub-pixel refinement on the 3x3 peak neighbourhood.
    sx, sy = float(bx), float(by)
    r, c = by - sy_min, bx - sx_min
    if 0 < r < scores.shape[0] - 1 and 0 < c < scores.shape[1] - 1:
        dx_num = scores[r, c - 1] - scores[r, c + 1]
        dx_den = scores[r, c - 1] - 2 * scores[r, c] + scores[r, c + 1]
        dy_num = scores[r - 1, c] - scores[r + 1, c]
        dy_den = scores[r - 1, c] - 2 * scores[r, c] + scores[r + 1, c]
        if abs(dx_den) > 1e-9:
            sx += float(np.clip(0.5 * dx_num / dx_den, -1, 1))
        if abs(dy_den) > 1e-9:
            sy += float(np.clip(0.5 * dy_num / dy_den, -1, 1))
    return sx, sy, peak, scores, peak_ratio, tpl_std


def match_point(
    left: np.ndarray, right: np.ndarray,
    x: float, y: float, frame: int,
    valid_left: np.ndarray | None = None,
    valid_right: np.ndarray | None = None,
    source: str = "manual",
    dx_prior: float | None = None,
) -> Correspondence:
    """Match one reviewer click (left view) into the right view.

    Epipolar geometry constrains the right-view point to lie to the left of
    the left-view point (right_x = left_x - disparity, disparity >= 0).  An
    optional ``dx_prior`` narrows the horizontal search around an expected
    disparity; without it the full disparity range is scanned.
    """
    gl, gr = to_gray(left), to_gray(right)
    h, w = gl.shape
    cx, cy = int(round(x)), int(round(y))
    note = ""

    near_boundary = not (WINDOW + 2 <= cx < w - WINDOW - 2 and WINDOW + 2 <= cy < h - WINDOW - 2)
    left_occluded = valid_left is not None and not bool(valid_left[cy, cx])

    if dx_prior is None:
        sx_lo, sx_hi = max(WINDOW, cx - SEARCH_X), min(w - WINDOW - 1, cx + 2)
    else:
        p = int(round(dx_prior))
        sx_lo = max(WINDOW, cx - p - 8)
        sx_hi = min(w - WINDOW - 1, cx - p + 8)
    rx, ry, ncc, _, peak_ratio, tpl_std = _ncc_search(
        gl, gr, cx, cy, sx_lo, sx_hi,
        max(WINDOW, cy - SEARCH_Y), min(h - WINDOW - 1, cy + SEARCH_Y),
    )

    status, confidence = "ok", float(max(ncc, 0.0))
    if left_occluded:
        status, note = "conflict", "clicked point is inside a left-view occlusion mask"
    elif near_boundary:
        status, note = "conflict", "point too close to frame border: truncated template"
    elif tpl_std < 6.0:
        status, note = "low", f"flat/low-texture region (sigma {tpl_std:.1f})"
    elif peak_ratio > 0.97:
        status, note = "low", (
            f"ambiguous periodic texture: second NCC peak at {peak_ratio:.2f} of best"
        )
    elif ncc < LOW_CONF:
        status, note = "low", f"NCC {ncc:.2f} below threshold {LOW_CONF:.2f}"
    else:
        # Left-right consistency: match back from right into left. The back
        # search lies to the right, mirroring the forward constraint.
        rcx, rcy = int(round(rx)), int(round(ry))
        if WINDOW < rcx < w - WINDOW and WINDOW < rcy < h - WINDOW:
            bx, by, ncc_back, _, _, _ = _ncc_search(
                gr, gl, rcx, rcy,
                max(WINDOW, rcx - 2), min(w - WINDOW - 1, rcx + SEARCH_X),
                max(WINDOW, rcy - SEARCH_Y), min(h - WINDOW - 1, rcy + SEARCH_Y),
            )
            if math.hypot(bx - cx, by - cy) > LR_THRESHOLD:
                status, note = "low", (
                    f"left-right consistency failed "
                    f"({math.hypot(bx - cx, by - cy):.1f}px)"
                )
                confidence = min(confidence, 0.3)
        if valid_right is not None and WINDOW < rcy < h - WINDOW and WINDOW < rcx < w - WINDOW:
            if not bool(valid_right[rcy, rcx]):
                status, note = "conflict", "match lands in a right-view only/occluded region"

    return Correspondence(
        frame=frame,
        left=Point(float(cx), float(cy)),
        right=Point(rx, ry),
        confidence=confidence,
        status=status,
        source=source,
        note=note,
    )


def local_texture(left: np.ndarray, x: int, y: int, win: int = WINDOW) -> float:
    """Standard deviation of the grayscale template around a point."""
    g = to_gray(left)
    patch = g[max(0, y - win):y + win + 1, max(0, x - win):x + win + 1]
    return float(patch.std())


def auto_correspondences(
    left: np.ndarray, right: np.ndarray, frame: int,
    grid_step: int = 48,
    valid_left: np.ndarray | None = None,
    valid_right: np.ndarray | None = None,
) -> list[Correspondence]:
    """Lay a regular grid of candidate points over the left view and match."""
    h, w = to_gray(left).shape
    out: list[Correspondence] = []
    for cy in range(WINDOW * 2, h - WINDOW * 2, grid_step):
        for cx in range(WINDOW * 2, w - WINDOW * 2, grid_step):
            out.append(
                match_point(left, right, cx, cy, frame, valid_left, valid_right,
                            source="auto")
            )
    return out
