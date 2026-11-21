"""Vertical (epipolar) correction estimated from reviewer correspondences.

Two model scopes are supported:

* ``global``  - one correction applied to every frame
* ``per-shot``- an independent correction per shot (frame range)

Within each segment the correction is either a constant vertical offset or a
linear tilt ``dy(x) = a + b*x``.  The linear form is only accepted when the
slope is statistically distinguishable from zero; otherwise it collapses to
the constant model.  All fitting excludes low-confidence and conflict points.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import Correspondence

T_CRIT_95 = 1.96  # normal approximation; exact t used when n is small


@dataclass
class SegmentFit:
    start: int
    end: int  # inclusive
    kind: str  # "constant" | "linear"
    a: float
    b: float
    n_fit: int
    n_holdout: int
    residual_std: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float] | None
    holdout_rmse: float | None = None
    holdout_mean: float | None = None
    pred_band: float | None = None  # 95% prediction half-width at segment centre
    accepted: bool | None = None  # decided from hold-out points
    decision: str = ""  # human readable reason

    def dy(self, x: float) -> float:
        return self.a + self.b * x if self.kind == "linear" else self.a

    def to_dict(self) -> dict:
        def num(v):
            return None if v is None or not math.isfinite(v) else v
        return {
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
            "a": num(self.a),
            "b": num(self.b),
            "n_fit": self.n_fit,
            "n_holdout": self.n_holdout,
            "residual_std": num(self.residual_std),
            "ci_a": [num(self.ci_a[0]), num(self.ci_a[1])],
            "ci_b": [num(self.ci_b[0]), num(self.ci_b[1])] if self.ci_b is not None else None,
            "holdout_rmse": num(self.holdout_rmse),
            "holdout_mean": num(self.holdout_mean),
            "pred_band": num(self.pred_band),
            "accepted": self.accepted,
            "decision": self.decision,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentFit":
        def num(v):
            return float("nan") if v is None else float(v)
        return cls(
            start=int(d["start"]),
            end=int(d["end"]),
            kind=d["kind"],
            a=num(d["a"]),
            b=num(d["b"]),
            n_fit=int(d["n_fit"]),
            n_holdout=int(d["n_holdout"]),
            residual_std=num(d["residual_std"]),
            ci_a=(num(d["ci_a"][0]), num(d["ci_a"][1])),
            ci_b=(num(d["ci_b"][0]), num(d["ci_b"][1]))
            if d.get("ci_b") is not None
            else None,
            holdout_rmse=None if d.get("holdout_rmse") is None else num(d["holdout_rmse"]),
            holdout_mean=None if d.get("holdout_mean") is None else num(d["holdout_mean"]),
            pred_band=None if d.get("pred_band") is None else num(d["pred_band"]),
            accepted=d.get("accepted"),
            decision=d.get("decision", ""),
        )


@dataclass
class CorrectionModel:
    scope: str  # "global" | "per-shot"
    width: int
    segments: list[SegmentFit] = field(default_factory=list)
    approved: bool = False

    def dy(self, frame: int, x: float) -> float:
        covered = [s for s in self.segments if s.start <= frame <= s.end]
        if covered:
            return covered[0].dy(x)
        if self.scope == "global" and self.segments:
            # A global correction applies to the whole clip even where no
            # reviewer point happened to land; extrapolate its single line.
            return self.segments[0].dy(x)
        if self.segments:
            # Per-shot gap: use the nearest fitted segment rather than fail.
            nearest = min(self.segments,
                          key=lambda s: min(abs(frame - s.start), abs(frame - s.end)))
            return nearest.dy(x)
        raise KeyError(f"frame {frame} is not covered by any fitted segment")

    def covers(self, frame: int) -> bool:
        if self.scope == "global" and self.segments:
            return True
        return any(s.start <= frame <= s.end for s in self.segments)

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "width": self.width,
            "approved": self.approved,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CorrectionModel":
        return cls(
            scope=d["scope"],
            width=int(d["width"]),
            approved=bool(d.get("approved", False)),
            segments=[SegmentFit.from_dict(s) for s in d["segments"]],
        )


def _eligible(corrs: list[Correspondence]) -> list[Correspondence]:
    return [c for c in corrs if c.status == "ok"]


def _split_holdout(corrs: list[Correspondence], holdout_frac: float = 0.3, seed: int = 7):
    """Deterministic split: every third eligible point is held out."""
    fit, hold = [], []
    for i, c in enumerate(corrs):
        (hold if i % 3 == 2 else fit).append(c)
    # Guarantee at least one hold-out point when there are enough of them.
    if corrs and not hold and len(fit) > 2:
        hold.append(fit.pop())
    return fit, hold


def _t_critical(df: int) -> float:
    if df <= 0:
        return T_CRIT_95
    # Student t quantile lookup; beyond 30 df the normal approximation
    # is within 0.05.
    return _student_t_0975(df)


def _student_t_0975(df: int) -> float:
    # Table lookup covers the degrees of freedom that occur in practice;
    # beyond 30 df the normal approximation is within 0.05.
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return table.get(df, T_CRIT_95)


def _huber_fit(x: np.ndarray, y: np.ndarray, linear: bool, iterations: int = 4):
    """Weighted least squares with Huber re-weighting; returns params and sigma."""
    n = len(x)
    w = np.ones(n)
    params = None
    for _ in range(iterations):
        if linear:
            X = np.column_stack([np.ones(n), x])
        else:
            X = np.ones((n, 1))
        Xw = X * w[:, None]
        params, *_ = np.linalg.lstsq(Xw.T @ X, Xw.T @ y, rcond=None)
        resid = y - X @ params
        sigma = 1.345 * np.median(np.abs(resid - np.median(resid))) + 1e-9
        scaled = np.abs(resid) / sigma
        w = np.where(scaled <= 1.345, 1.0, 1.345 / np.maximum(scaled, 1e-12))
    X = np.column_stack([np.ones(n), x]) if linear else np.ones((n, 1))
    resid = y - X @ params
    # Unbiased residual scale on the robust weights.
    dof = 2 if linear else 1
    sigma = math.sqrt(float(np.sum(w * resid**2)) / max(1, np.sum(w) - dof))
    return params, sigma, w


def fit_segment(corrs: list[Correspondence], start: int, end: int, width: int) -> SegmentFit:
    """Fit one segment using its own points, holding 30% out for validation."""
    seg = [c for c in _eligible(corrs) if start <= c.frame <= end]
    fit_pts, hold_pts = _split_holdout(seg)
    if len(fit_pts) < 2:
        return SegmentFit(
            start, end, "constant", float("nan"), 0.0,
            len(fit_pts), len(hold_pts), float("nan"),
            (float("nan"), float("nan")), None,
            accepted=None, decision="insufficient points to fit",
        )

    x = np.array([c.left.x for c in fit_pts], dtype=float)
    y = np.array([c.dy for c in fit_pts], dtype=float)
    xn = x - width / 2.0  # centre so a/b confidence bands stay interpretable

    (a_c,), sigma_c, _ = _huber_fit(xn, y, linear=False)
    kind, params, sigma = "constant", np.array([a_c, 0.0]), sigma_c
    ci_b = None
    if len(fit_pts) >= 6:
        lin, sigma_l, w = _huber_fit(xn, y, linear=True)
        # Slope significance on the final robust weights.
        X = np.column_stack([np.ones(len(xn)), xn])
        Xw = X * w[:, None]
        cov = np.linalg.pinv(Xw.T @ X) * sigma_l**2
        se_b = math.sqrt(max(cov[1, 1], 0.0))
        t = _t_critical(len(fit_pts) - 2)
        if se_b > 0 and abs(lin[1]) > t * se_b:
            kind, params, sigma = "linear", lin, sigma_l
            ci_b = (lin[1] - t * se_b, lin[1] + t * se_b)

    a, b = float(params[0]), float(params[1])
    n = len(fit_pts)
    dof = 2 if kind == "linear" else 1
    se_a = sigma / math.sqrt(max(n - dof, 1))
    t = _t_critical(max(n - dof, 1))
    ci_a = (a - t * se_a, a + t * se_a)

    fit = SegmentFit(
        start, end, kind, a, b, n, len(hold_pts), sigma, ci_a, ci_b,
    )

    if hold_pts:
        errs = np.array([c.dy - fit.dy(c.left.x) for c in hold_pts])
        fit.holdout_mean = float(np.mean(errs))
        fit.holdout_rmse = float(np.sqrt(np.mean(errs**2)))
        # Prediction interval at segment centre: sigma * t * sqrt(1 + 1/n)
        fit.pred_band = sigma * t * math.sqrt(1.0 + 1.0 / max(n, 1))
        fit.accepted, fit.decision = _judge_holdout(fit, errs)
    else:
        fit.accepted = None
        fit.decision = "no held-out points available"
    return fit


def _judge_holdout(fit: SegmentFit, errs: np.ndarray) -> tuple[bool | None, str]:
    n = len(errs)
    if n < 2:
        return None, "too few held-out points to generalise"
    rmse = math.sqrt(float(np.mean(errs**2)))
    # Tolerance combines the model's own prediction band with the 1 px
    # sub-pixel localisation floor; a tilt whose residuals are sub-pixel is
    # not evidence against generalisation.
    tolerance = max(fit.pred_band or fit.residual_std * 2, 1.0)
    inside = float(np.mean(np.abs(errs) <= tolerance))
    bias = float(np.mean(errs))
    if inside >= 0.8 and abs(bias) <= max(fit.residual_std, 1.0):
        return True, (
            f"hold-out RMSE {rmse:.2f}px within tolerance "
            f"{tolerance:.2f}px ({inside:.0%} inside)"
        )
    return False, (
        f"hold-out RMSE {rmse:.2f}px exceeds tolerance {tolerance:.2f}px "
        f"({inside:.0%} inside, bias {bias:+.2f}px)"
    )


def build_model(
    corrs: list[Correspondence],
    scope: str,
    frame_count: int,
    width: int,
    shots: list[tuple[int, int]] | None = None,
) -> CorrectionModel:
    if scope == "global":
        ranges = [(0, frame_count - 1)]
    elif scope == "per-shot":
        ranges = shots or [(0, frame_count - 1)]
    else:
        raise ValueError(f"unknown scope {scope}")
    model = CorrectionModel(scope=scope, width=width)
    for start, end in ranges:
        model.segments.append(fit_segment(corrs, start, end, width))
    return model


def corrected_dy(c: Correspondence, model: CorrectionModel) -> float:
    """Residual vertical mismatch after applying the fitted correction."""
    return c.dy - model.dy(c.frame, c.left.x)
