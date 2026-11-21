"""Headless quantitative evaluation against the synthetic ground truth.

For every sampled reviewer point we know the true counterpart from the
disparity layers and the true vertical drift, and we know from the visibility
masks whether the point should match at all.  That lets us measure:

* localisation error of accepted matches (x/y, px)
* false alarms    - points the tool keeps as confident matches where the
                    ground truth says occlusion / monocular / low texture
* missed blanks   - ground-truth invalid points that were not blanked
* over-blanking   - ground-truth valid points blanked unnecessarily

and compare the full-frame shared correction against per-shot fitting to show
which segment produces spurious vertical alarms.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import footage
from .geometry import Correspondence, Point
from .matching import auto_correspondences, local_texture
from .vertical import build_model, corrected_dy

FLAT_SIGMA = 6.0


def _load(project: Path, rel: str) -> np.ndarray:
    return np.asarray(Image.open(project / rel))


def _load_mask(project: Path, rel: str) -> np.ndarray:
    return np.asarray(Image.open(project / rel).convert("L")) > 127


def true_counterpart(frame: int, p: Point, dx: float) -> Point:
    return Point(p.x - dx, p.y + footage.true_dy(frame, p.x))


def evaluate(project_dir: str | Path, grid_step: int = 48) -> dict:
    project = Path(project_dir)
    info = json.loads((project / "scene_info.json").read_text())
    n = info["frame_count"]
    w, h = info["width"], info["height"]

    rows = []
    all_corrs: list[Correspondence] = []
    for f in range(n):
        meta = info["frames"][f]
        left = _load(project, meta["left"])
        right = _load(project, meta["right"])
        vl = _load_mask(project, meta["valid_left"])
        vr = _load_mask(project, meta["valid_right"])
        repeated = _load_mask(project, meta["repeated"])
        mono = _load_mask(project, meta["mono"])
        gt_dx = np.load(project / meta["gt_dx"])

        corrs = auto_correspondences(left, right, f, grid_step, vl, vr)
        all_corrs.extend(corrs)
        for c in corrs:
            x, y = int(round(c.left.x)), int(round(c.left.y))
            truth_valid = bool(vl[y, x]) and not bool(mono[y, x])
            kept = c.status == "ok"
            truth = None
            err_x = err_y = None
            if truth_valid and np.isfinite(gt_dx[y, x]):
                truth = true_counterpart(f, c.left, float(gt_dx[y, x]))
                err_x = c.right.x - truth.x
                err_y = c.right.y - truth.y
            rows.append({
                "frame": f,
                "x": x, "y": y,
                "status": c.status,
                "kept": kept,
                "truth_valid": truth_valid,
                "repeated": bool(repeated[y, x]),
                "flat": local_texture(left, x, y) < FLAT_SIGMA,
                "ncc": c.confidence,
                "err_x": err_x,
                "err_y": err_y,
                "pred_dx": c.dx,
                "true_dx": float(gt_dx[y, x]) if np.isfinite(gt_dx[y, x]) else None,
                "raw_dy": c.dy,
                "true_dy": footage.true_dy(f, x),
                "note": c.note,
            })

    def localise(scope_rows):
        good = [r for r in scope_rows if r["kept"] and r["truth_valid"]
                and r["err_x"] is not None]
        if not good:
            return {"n": 0}
        ex = np.array([r["err_x"] for r in good])
        ey = np.array([r["err_y"] for r in good])
        return {
            "n": len(good),
            "rmse_x_px": float(np.sqrt(np.mean(ex**2))),
            "rmse_y_px": float(np.sqrt(np.mean(ey**2))),
            "bias_x_px": float(np.mean(ex)),
            "bias_y_px": float(np.mean(ey)),
            "p95_xy_px": float(np.percentile(np.hypot(ex, ey), 95)),
        }

    def alarms(scope_rows):
        fa = [r for r in scope_rows if r["kept"] and not r["truth_valid"]]
        missed = [r for r in scope_rows if (not r["kept"]) and r["truth_valid"]]
        repeat_fa = [r for r in fa if r["repeated"]]
        # Over-blanking is only meaningful on textured, non-repeated, valid
        # ground truth: periodic, repeated and flat regions are blanked by
        # design rather than representing an error.
        over = [r for r in missed if not r["repeated"] and not r["flat"]
                and "ambiguous periodic" not in r["note"]]
        designed = [r for r in missed if r["repeated"] or r["flat"]
                    or "ambiguous periodic" in r["note"]]
        return {
            "false_alarms": len(fa),
            "false_alarms_repeated_texture": len(repeat_fa),
            "missed_blanks": len(fa),
            "over_blanked_textured_valid": len(over),
            "correctly_blanked_repeated_or_flat": len(designed),
            "total_kept": sum(1 for r in scope_rows if r["kept"]),
            "total_invalid_truth": sum(1 for r in scope_rows if not r["truth_valid"]),
        }

    # --- correction scope comparison -------------------------------------
    shots = [tuple(s) for s in info["shots"]]
    global_model = build_model(all_corrs, "global", n, w, shots)
    shot_model = build_model(all_corrs, "per-shot", n, w, shots)
    truth_drift = info["true_drift"]

    def model_summary(model):
        out = []
        for seg in model.segments:
            out.append({
                "frames": [seg.start, seg.end],
                "kind": seg.kind,
                "a": None if math.isnan(seg.a) else round(seg.a, 3),
                "b": round(seg.b, 5),
                "ci_a": [round(v, 3) for v in seg.ci_a] if not math.isnan(seg.ci_a[0]) else None,
                "holdout_rmse": None if seg.holdout_rmse is None else round(seg.holdout_rmse, 3),
                "holdout_accepted": seg.accepted,
                "decision": seg.decision,
            })
        return out

    # Vertical false alarms: residuals beyond 1 px after correction, using
    # held-out points only so the judgement is independent of the fit.  The
    # split mirrors vertical._split_holdout (every third eligible point).
    def vertical_alarms(model):
        per_shot = []
        for seg in model.segments:
            eligible = [c for c in all_corrs
                        if c.status == "ok" and seg.start <= c.frame <= seg.end]
            held = [c for i, c in enumerate(eligible) if i % 3 == 2]
            res = np.array([corrected_dy(c, model) for c in held]) if held else np.array([])
            per_shot.append({
                "frames": [seg.start, seg.end],
                "held_out": len(res),
                "residual_rmse_px": float(np.sqrt(np.mean(res**2))) if res.size else None,
                "vertical_false_alarms": int(np.sum(np.abs(res) > 1.0)) if res.size else 0,
            })
        return per_shot

    result = {
        "project": str(project),
        "frame_count": n,
        "localisation": localise(rows),
        "alarms": alarms(rows),
        "by_shot": [
            {
                "frames": shots[i],
                "localisation": localise([r for r in rows if shots[i][0] <= r["frame"] <= shots[i][1]]),
                "alarms": alarms([r for r in rows if shots[i][0] <= r["frame"] <= shots[i][1]]),
                "true_drift": truth_drift["shot_A" if i == 0 else "shot_B"],
            }
            for i in range(len(shots))
        ],
        "global_model": model_summary(global_model),
        "per_shot_model": model_summary(shot_model),
        "vertical_alarms_global": vertical_alarms(global_model),
        "vertical_alarms_per_shot": vertical_alarms(shot_model),
        "points": rows,
    }
    out_path = project / "evaluation.json"
    out_path.write_text(json.dumps(result, indent=2))
    return result
