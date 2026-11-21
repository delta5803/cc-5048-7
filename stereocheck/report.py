"""Reviewer report generation.

Hard rule enforced here: the horizontal disparity statistics and boundary
conflict maps are recomputed from the *approved* transform snapshot.  If no
transform has been approved, no disparity numbers are emitted - the report
states that approval is pending instead of falling back to stale or raw data.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from .disparity import (
    angular_range,
    dense_disparity,
    disparity_report,
    load_gray,
    load_mask,
    render_conflict_overlay,
    render_disparity_map,
)
from .session import Session


def build_report(session: Session, progress=None) -> dict:
    project = session.dir
    info = session.scene_info
    approved = session.approved_model()
    report_dir = project / "reports"
    report_dir.mkdir(exist_ok=True)

    base = {
        "project": str(project),
        "frame_count": info["frame_count"],
        "shots": info["shots"],
        "approved_transform": approved.to_dict() if approved else None,
        "display": session.display,
        "frames": [],
    }
    if approved is None:
        base["status"] = "blocked"
        base["reason"] = (
            "no approved vertical correction; fit and approve a model before "
            "disparity ranges or boundary conflict maps may be reported"
        )
        (report_dir / "report.json").write_text(json.dumps(base, indent=2))
        if progress:
            progress(0, info["frame_count"], "blocked")
        return base

    base["status"] = "ok"
    totals = []
    for f in range(info["frame_count"]):
        if progress:
            progress(f, info["frame_count"], "computing")
        meta = info["frames"][f]
        left_img = np.asarray(Image.open(project / meta["left"]).convert("RGB"))
        gl = load_gray(project / meta["left"])
        gr = load_gray(project / meta["right"])
        vl = load_mask(project / meta["valid_left"])
        vr = load_mask(project / meta["valid_right"])

        dx, conf, reason = dense_disparity(gl, gr, f, approved, vl, vr)
        stats = disparity_report(dx)
        angle = angular_range(
            stats,
            session.display.get("width_px") or info["width"],
            session.display.get("viewing_distance_mm") or 0.0,
            session.display.get("width_mm") or 0.0,
        )
        stats["angular"] = angle

        conflict = render_conflict_overlay(left_img, reason, dx)
        dmap = render_disparity_map(dx)
        conflict_path = report_dir / f"f{f:03d}_conflict.png"
        dmap_path = report_dir / f"f{f:03d}_disparity.png"
        Image.fromarray(conflict).save(conflict_path)
        Image.fromarray(dmap).save(dmap_path)
        stats["conflict_map"] = str(conflict_path.relative_to(project))
        stats["disparity_map"] = str(dmap_path.relative_to(project))
        stats["frame"] = f
        base["frames"].append(stats)
        totals.append(dx[np.isfinite(dx)])

    all_vals = np.concatenate([t for t in totals if t.size])
    base["aggregate"] = _aggregate(all_vals)
    (report_dir / "report.json").write_text(json.dumps(base, indent=2))
    return base


def _aggregate(vals: np.ndarray) -> dict:
    pct = np.percentile(vals, [0, 2, 50, 98, 100])
    return {
        "valid_pixels": int(vals.size),
        "min_px": float(pct[0]),
        "p02_px": float(pct[1]),
        "median_px": float(pct[2]),
        "p98_px": float(pct[3]),
        "max_px": float(pct[4]),
        "robust_range_px": float(pct[3] - pct[1]),
    }
