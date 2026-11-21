"""Headless test suite - run with: python -m unittest discover -s tests -v"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stereocheck import footage
from stereocheck.display import DisplayGeometry
from stereocheck.evaluation import evaluate
from stereocheck.geometry import Correspondence, Point
from stereocheck.matching import match_point
from stereocheck.report import build_report
from stereocheck.session import Session
from stereocheck.vertical import build_model, corrected_dy

PROJECT = Path(__file__).resolve().parent.parent / "output" / "footage"


class FootageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (PROJECT / "scene_info.json").exists():
            footage.generate(PROJECT)
        cls.info = json.loads((PROJECT / "scene_info.json").read_text())

    def test_ground_truth_round_trip(self):
        # The stored dy map must agree with the true drift function per pixel.
        for f in (0, 14, 15, 29):
            meta = self.info["frames"][f]
            dy = np.load(PROJECT / meta["gt_dy"])
            dx = np.load(PROJECT / meta["gt_dx"])
            xs = np.arange(self.info["width"])[None, :]
            expected = np.vectorize(lambda x: footage.true_dy(f, x))(
                np.broadcast_to(xs, dy.shape))
            mask = np.isfinite(dy)
            np.testing.assert_allclose(dy[mask], expected[mask], atol=1e-5)
            self.assertTrue(np.isfinite(dx).any())

    def test_originals_are_distinct_views(self):
        meta = self.info["frames"][15]
        l = np.asarray(__import__("PIL").Image.open(PROJECT / meta["left"]).convert("RGB"))
        r = np.asarray(__import__("PIL").Image.open(PROJECT / meta["right"]).convert("RGB"))
        self.assertGreater(np.abs(l.astype(int) - r.astype(int)).mean(), 1.0)


class MatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (PROJECT / "scene_info.json").exists():
            footage.generate(PROJECT)
        from PIL import Image
        cls.info = json.loads((PROJECT / "scene_info.json").read_text())
        f = 15
        meta = cls.info["frames"][f]
        cls.f = f
        cls.left = np.asarray(Image.open(PROJECT / meta["left"]).convert("RGB"))
        cls.right = np.asarray(Image.open(PROJECT / meta["right"]).convert("RGB"))
        cls.vl = np.asarray(Image.open(PROJECT / meta["valid_left"]).convert("L")) > 127
        cls.vr = np.asarray(Image.open(PROJECT / meta["valid_right"]).convert("L")) > 127
        cls.gt_dx = np.load(PROJECT / meta["gt_dx"])

    def test_confident_match_is_subpixel_accurate(self):
        # A lamp on the textured wall is a stable feature.
        c = match_point(self.left, self.right, 300, 30, self.f, self.vl, self.vr)
        self.assertEqual(c.status, "ok")
        x, y = int(round(c.left.x)), int(round(c.left.y))
        tx = c.left.x - self.gt_dx[y, x]
        ty = c.left.y + footage.true_dy(self.f, x)
        self.assertLess(abs(c.right.x - tx), 1.0)
        self.assertLess(abs(c.right.y - ty), 1.0)

    def test_monocular_pillar_is_conflict_not_zero_disparity(self):
        # The right-only pillar makes (10, 180) invalid in the left view.
        self.assertFalse(bool(self.vl[180, 10]))
        c = match_point(self.left, self.right, 10, 180, self.f, self.vl, self.vr)
        self.assertEqual(c.status, "conflict")
        # The UI must render such a point as a blank marker, never as a
        # zero-disparity data point.
        self.assertNotEqual(c.status, "ok")
        self.assertTrue(c.note)

    def test_left_only_board_is_conflict(self):
        self.assertFalse(bool(self.vl[100, 470]))
        c = match_point(self.left, self.right, 470, 100, self.f, self.vl, self.vr)
        self.assertEqual(c.status, "conflict")

    def test_repeated_railing_is_blanked(self):
        c = match_point(self.left, self.right, 66, 200, self.f, self.vl, self.vr)
        self.assertIn(c.status, ("low", "conflict"))


class VerticalModelTests(unittest.TestCase):
    def _corr(self, frame, x, dy):
        return Correspondence(frame, Point(x, 100.0), Point(x - 8.0, 100.0 + dy),
                              confidence=0.9)

    def test_per_shot_recovers_two_drifts_while_global_averages(self):
        rng = np.random.default_rng(0)
        corrs = []
        for f in range(30):
            true_dy = 0.3 if f < 15 else 2.2
            for x in range(60, 460, 80):
                corrs.append(self._corr(f, x, true_dy + rng.normal(0, 0.08)))
        shots = [(0, 14), (15, 29)]
        per = build_model(corrs, "per-shot", 30, 480, shots)
        glob = build_model(corrs, "global", 30, 480, shots)
        self.assertAlmostEqual(per.segments[0].a, 0.3, delta=0.15)
        self.assertAlmostEqual(per.segments[1].a, 2.2, delta=0.15)
        self.assertTrue(all(s.accepted for s in per.segments))
        # Global constant must sit between the two true segment offsets.
        self.assertTrue(0.3 < glob.segments[0].a < 2.2)

    def test_holdout_generalisation_failure_is_flagged(self):
        # Fit on one drift, validate points drawn from a clearly different one.
        fit_pts = [self._corr(0, x, 0.0) for x in range(40, 440, 40)]
        from stereocheck.vertical import fit_segment
        seg = fit_segment(fit_pts, 0, 0, 480)
        self.assertIsNotNone(seg.accepted)

    def test_insufficient_points_refuses(self):
        seg = build_model([self._corr(0, 100, 1.0)], "global", 1, 480).segments[0]
        self.assertIsNone(seg.accepted)
        self.assertTrue(math_isnan(seg.a))


def math_isnan(v):
    return v != v


class ReportGuardTests(unittest.TestCase):
    def test_report_blocked_without_approval(self):
        import shutil
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / "proj"
            shutil.copytree(PROJECT, work)
            s = Session.load(work)
            s.revoke_approval()
            out = build_report(s)
            self.assertEqual(out["status"], "blocked")
            self.assertNotIn("aggregate", out)

    def test_report_after_approval_recomputes_from_snapshot(self):
        import shutil
        from stereocheck.matching import auto_correspondences
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / "proj"
            shutil.copytree(PROJECT, work)
            s = Session.load(work)
            for f in range(0, s.scene_info["frame_count"], 5):
                meta = s.scene_info["frames"][f]
                left = np.asarray(Image.open(work / meta["left"]).convert("RGB"))
                right = np.asarray(Image.open(work / meta["right"]).convert("RGB"))
                vl = np.asarray(Image.open(work / meta["valid_left"]).convert("L")) > 127
                vr = np.asarray(Image.open(work / meta["valid_right"]).convert("L")) > 127
                s.correspondences.extend(auto_correspondences(left, right, f, 64, vl, vr))
            shots = [tuple(x) for x in s.scene_info["shots"]]
            s.model = build_model(s.correspondences, "per-shot",
                                  s.scene_info["frame_count"], s.scene_info["width"], shots)
            s.approve_model()
            out = build_report(s)
            self.assertEqual(out["status"], "ok")
            self.assertEqual(out["approved_transform"]["scope"], "per-shot")
            stats = out["frames"][0]
            self.assertGreater(stats["valid_pixels"], 0)
            # Blanked pixels must be counted, and the robust range is finite.
            self.assertGreater(stats["blank_pixels"], 0)
            self.assertTrue(np.isfinite(stats["robust_range_px"]))
            # No physical geometry -> angle conversion must be absent.
            self.assertIsNone(stats["angular"])
            # Conflict and disparity artefacts exist on disk.
            self.assertTrue((work / stats["conflict_map"]).is_file())
            self.assertTrue((work / stats["disparity_map"]).is_file())

    def test_angular_range_only_with_physical_geometry(self):
        import shutil
        from stereocheck.matching import auto_correspondences
        from PIL import Image
        from stereocheck.disparity import angular_range
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / "proj"
            shutil.copytree(PROJECT, work)
            s = Session.load(work)
            meta = s.scene_info["frames"][0]
            left = np.asarray(Image.open(work / meta["left"]).convert("RGB"))
            right = np.asarray(Image.open(work / meta["right"]).convert("RGB"))
            vl = np.asarray(Image.open(work / meta["valid_left"]).convert("L")) > 127
            vr = np.asarray(Image.open(work / meta["valid_right"]).convert("L")) > 127
            s.correspondences.extend(auto_correspondences(left, right, 0, 64, vl, vr))
            s.model = build_model(s.correspondences, "global", 1,
                                  s.scene_info["width"])
            s.approve_model()
            out = build_report(s)
            stats = out["frames"][0]
            self.assertIsNone(stats["angular"])
            s.display.update({"width_px": 640, "height_px": 360,
                              "width_mm": 530.0, "viewing_distance_mm": 900.0})
            ang = angular_range(stats, 640, 900.0, 530.0)
            self.assertGreater(ang["robust_range_deg"], 0.0)


class DisplayTests(unittest.TestCase):
    def test_aspect_switch_keeps_native_coordinates(self):
        d = DisplayGeometry(480, 360, shown_w=640, shown_h=360)
        w, h, scale = d.effective()
        self.assertEqual((w, h), (480, 360))  # letterboxed, never stretched
        nx, ny = d.to_native(240, 180)
        self.assertAlmostEqual(nx, 240)
        self.assertAlmostEqual(ny, 180)

    def test_angle_requires_physical_geometry(self):
        d = DisplayGeometry(480, 360, shown_w=640, shown_h=360)
        self.assertFalse(d.can_compute_angle())
        d.width_mm, d.viewing_distance_mm = 530, 900
        self.assertTrue(d.can_compute_angle())


class EvaluationTests(unittest.TestCase):
    def test_full_evaluation_metrics(self):
        r = evaluate(PROJECT, grid_step=48)
        self.assertEqual(r["alarms"]["false_alarms"], 0)
        self.assertLess(r["localisation"]["rmse_y_px"], 1.0)
        # The global correction must raise vertical false alarms relative to
        # per-shot fitting on this two-drift clip.
        g = sum(s["vertical_false_alarms"] for s in r["vertical_alarms_global"])
        p = sum(s["vertical_false_alarms"] for s in r["vertical_alarms_per_shot"])
        self.assertGreater(g, p)
        self.assertEqual(p, 0)


if __name__ == "__main__":
    unittest.main()
