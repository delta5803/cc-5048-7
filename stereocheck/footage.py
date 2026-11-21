"""Synthetic binocular footage with ground-truth geometry.

The clip deliberately alternates the three failure modes a reviewer must see:

* repeated texture   - periodic railing bars and a tiled wall section
* monocular regions  - a pillar that exists only in the right eye and a
                       notice board that exists only in the left eye, plus
                       disocclusion strips around the walking person
* real vertical drift- shot A is nearly level while shot B carries a true
                       offset plus a small tilt

Everything rendered here is accompanied by machine-readable ground truth so
the evaluation command can measure localisation error and false alarms rather
than eyeball them.  Original frames are written once and never overwritten.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H = 480, 360
FRAME_COUNT = 30
SHOT_SPLIT = 15  # shot A: 0..14, shot B: 15..29

# Layer identifiers and their true horizontal disparity (px, left minus right).
LBL_WALL, LBL_TILE, LBL_RAIL, LBL_PERSON, LBL_MONO = 0, 1, 2, 3, 4
LAYER_DISPARITY = {
    LBL_WALL: 5,
    LBL_TILE: 5,
    LBL_RAIL: 13,
    LBL_PERSON: 24,
    LBL_MONO: 0,
}
LAYER_NAMES = {
    LBL_WALL: "wall", LBL_TILE: "tile", LBL_RAIL: "railing",
    LBL_PERSON: "person", LBL_MONO: "monocular",
}

# True vertical drift of the right view: right_y = left_y + dy(x).
DRIFT = {
    0: {"a": 0.3, "b": 0.0},                 # shot A: negligible
    1: {"a": 2.2, "b": 0.003},               # shot B: offset + tilt
}


@dataclass
class FrameArtifacts:
    frame: int
    left_path: Path
    right_path: Path
    valid_left_path: Path
    valid_right_path: Path
    repeated_path: Path
    mono_path: Path
    gt_dx_path: Path
    gt_dy_path: Path


def true_dy(frame: int, x: float) -> float:
    shot = 0 if frame < SHOT_SPLIT else 1
    p = DRIFT[shot]
    return p["a"] + p["b"] * (x - W / 2.0)


def _wall_texture() -> np.ndarray:
    """Static wide texture canvas so horizontal crops produce natural parallax."""
    rng = np.random.default_rng(20261121)
    canvas_w = W + 40
    base = np.zeros((H, canvas_w, 3), dtype=np.uint8)
    grad = np.linspace(150, 205, H, dtype=np.uint8)[:, None]
    base[..., 0] = grad - 30
    base[..., 1] = grad
    base[..., 2] = grad + 20
    noise = rng.integers(-14, 14, size=(H, canvas_w, 1), dtype=np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return base


def _new_canvas() -> tuple[Image.Image, Image.Image]:
    rgb = Image.new("RGB", (W, H), (0, 0, 0))
    lbl = Image.new("L", (W, H), LBL_WALL)
    return rgb, lbl


def _paint_tile(draw_rgb: ImageDraw.ImageDraw, draw_lbl: ImageDraw.ImageDraw, ox: int):
    # Repeated diamond-tile section on the wall: intentionally self-similar.
    x0, x1, y0, y1 = 70 - ox, 210 - ox, 40, 320
    for k, x in enumerate(range(x0 - 24, x1 + 24, 24)):
        col = (186, 176, 150) if k % 2 == 0 else (176, 166, 140)
        draw_rgb.rectangle([x, y0, x + 22, y1], fill=col)
        draw_lbl.rectangle([x, y0, x + 22, y1], fill=LBL_TILE)
        for y in range(y0, y1, 24):
            draw_rgb.line([x, y, x + 22, y + 22], fill=(140, 130, 110), width=1)
            draw_rgb.line([x + 22, y, x, y + 22], fill=(140, 130, 110), width=1)


def _paint_railing(draw_rgb: ImageDraw.ImageDraw, draw_lbl: ImageDraw.ImageDraw, ox: int):
    top, bottom = 60, 340
    for x in range(26 - ox, W + 20, 34):
        draw_rgb.rectangle([x, top, x + 7, bottom], fill=(60, 66, 78))
        draw_lbl.rectangle([x, top, x + 7, bottom], fill=LBL_RAIL)
        draw_rgb.ellipse([x - 2, top - 6, x + 9, top + 4], fill=(80, 86, 98))
    for y in (top + 6, bottom - 14):
        draw_rgb.rectangle([0, y, W, y + 6], fill=(70, 76, 88))
        draw_lbl.rectangle([0, y, W, y + 6], fill=LBL_RAIL)


def _paint_person(draw_rgb, draw_lbl, px: int):
    top = 150
    draw_rgb.rounded_rectangle([px, top, px + 46, top + 150], radius=14,
                               fill=(178, 84, 64))
    draw_lbl.rounded_rectangle([px, top, px + 46, top + 150], radius=14,
                               fill=LBL_PERSON)
    draw_rgb.ellipse([px + 8, top - 40, px + 38, top - 10], fill=(224, 186, 150))
    draw_lbl.ellipse([px + 8, top - 40, px + 38, top - 10], fill=LBL_PERSON)


def _paint_scene(frame: int, eye: str, texture: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb_img, lbl_img = _new_canvas()
    if eye == "L":
        bg_ox, rail_ox, person_ox = 0, 0, 0
    else:
        bg_ox, rail_ox, person_ox = (
            LAYER_DISPARITY[LBL_WALL],
            LAYER_DISPARITY[LBL_RAIL],
            LAYER_DISPARITY[LBL_PERSON],
        )
    rgb_img.paste(Image.fromarray(texture[:, bg_ox:bg_ox + W]))
    dr, dl = ImageDraw.Draw(rgb_img), ImageDraw.Draw(lbl_img)
    _paint_tile(dr, dl, bg_ox)

    # Distinct round wall lamps give the reviewer an obvious reference for the
    # vertical drift (each lamp moves by dy(x) between the eyes).
    for lamp_x in (140, 300, 430):
        x = lamp_x - bg_ox
        dr.ellipse([x - 12, 18, x + 12, 42], fill=(238, 214, 140), outline=(150, 120, 70))
        dl.ellipse([x - 12, 18, x + 12, 42], fill=LBL_WALL)

    _paint_railing(dr, dl, rail_ox)
    # The walker passes IN FRONT of the railing.
    person_left = -60 + frame * 22 - person_ox
    _paint_person(dr, dl, person_left)

    if eye == "R":
        # Monocular pillar: genuinely absent from the left camera.
        dr.rectangle([0, 0, 24, H], fill=(44, 48, 58))
        dl.rectangle([0, 0, 24, H], fill=LBL_MONO)
    else:
        # Left-only notice board at the opposite edge.
        dr.rectangle([W - 22, 40, W, 150], fill=(96, 120, 150))
        dl.rectangle([W - 22, 40, W, 150], fill=LBL_MONO)
    return np.asarray(rgb_img), np.asarray(lbl_img)


def _warp_columns(img: np.ndarray, frame: int) -> np.ndarray:
    """Apply dy(x): output(x, y) samples source(x, y - dy(x))."""
    out = np.empty_like(img)
    ys = np.arange(H, dtype=np.float64)
    for x in range(W):
        shift = true_dy(frame, x)
        sample = np.clip(ys - shift, 0, H - 1)
        y0 = np.floor(sample).astype(int)
        y1 = np.clip(y0 + 1, 0, H - 1)
        t = (sample - y0)
        if img.ndim == 3:
            t = t[:, None]
        out[:, x] = (img[y0, x] * (1 - t) + img[y1, x] * t).astype(img.dtype)
    return out


def _build_masks(frame: int, left_lbl: np.ndarray, right_lbl: np.ndarray):
    """Validity masks: a pixel is valid when its true counterpart exists."""
    xs = np.arange(W)[None, :].repeat(H, axis=0).astype(np.float64)
    ys = np.arange(H)[:, None].repeat(W, axis=1).astype(np.float64)
    dx_map = np.vectorize(lambda l: LAYER_DISPARITY.get(int(l), 0))(left_lbl)
    dy_map = np.vectorize(lambda x: true_dy(frame, x))(xs)

    valid_left = np.zeros((H, W), dtype=np.uint8)
    rx = np.rint(xs - dx_map).astype(int)
    ry = np.rint(ys + dy_map).astype(int)
    in_img = (rx >= 0) & (rx < W) & (ry >= 0) & (ry < H)
    idx = in_img
    valid_left[idx] = (
        (left_lbl != LBL_MONO)[idx]
        & (right_lbl[ry[idx], rx[idx]] != LBL_MONO)
    )

    valid_right = np.zeros((H, W), dtype=np.uint8)
    back_dx = np.vectorize(lambda l: LAYER_DISPARITY.get(int(l), 0))(right_lbl)
    lx = np.rint(xs + back_dx).astype(int)
    back_dy = np.vectorize(lambda x: true_dy(frame, x))(xs)
    ly = np.rint(ys - back_dy).astype(int)
    in_img = (lx >= 0) & (lx < W) & (ly >= 0) & (ly < H)
    valid_right[in_img] = (
        (right_lbl != LBL_MONO)[in_img]
        & (left_lbl[ly[in_img], lx[in_img]] != LBL_MONO)
    )

    gt_dx = np.where(valid_left.astype(bool), dx_map.astype(np.float32), np.nan)
    gt_dy = np.where(valid_left.astype(bool), dy_map.astype(np.float32), np.nan)
    return valid_left, valid_right, gt_dx, gt_dy


def _repeated_mask(lbl: np.ndarray) -> np.ndarray:
    return np.isin(lbl, [LBL_TILE, LBL_RAIL]).astype(np.uint8) * 255


def _mono_mask(lbl: np.ndarray) -> np.ndarray:
    return (lbl == LBL_MONO).astype(np.uint8) * 255


def generate(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    frames_dir = out / "frames"
    masks_dir = out / "masks"
    gt_dir = out / "groundtruth"
    for d in (frames_dir, masks_dir, gt_dir):
        d.mkdir(parents=True, exist_ok=True)

    texture = _wall_texture()
    artifacts: list[dict] = []
    for f in range(FRAME_COUNT):
        left_rgb, left_lbl = _paint_scene(f, "L", texture)
        right_rgb, right_lbl = _paint_scene(f, "R", texture)
        right_rgb = _warp_columns(right_rgb, f)
        right_lbl = _warp_columns(right_lbl[..., None], f)[..., 0]

        valid_left, valid_right, gt_dx, gt_dy = _build_masks(f, left_lbl, right_lbl)
        repeated = _repeated_mask(left_lbl)
        mono = np.maximum(_mono_mask(left_lbl), _mono_mask(right_lbl))

        paths = {
            "left": frames_dir / f"f{f:03d}_L.png",
            "right": frames_dir / f"f{f:03d}_R.png",
            "valid_left": masks_dir / f"f{f:03d}_validL.png",
            "valid_right": masks_dir / f"f{f:03d}_validR.png",
            "repeated": masks_dir / f"f{f:03d}_repeated.png",
            "mono": masks_dir / f"f{f:03d}_mono.png",
            "gt_dx": gt_dir / f"f{f:03d}_dx.npy",
            "gt_dy": gt_dir / f"f{f:03d}_dy.npy",
        }
        Image.fromarray(left_rgb).save(paths["left"])
        Image.fromarray(right_rgb).save(paths["right"])
        Image.fromarray(valid_left * 255).save(paths["valid_left"])
        Image.fromarray(valid_right * 255).save(paths["valid_right"])
        Image.fromarray(repeated).save(paths["repeated"])
        Image.fromarray(mono).save(paths["mono"])
        np.save(paths["gt_dx"], gt_dx)
        np.save(paths["gt_dy"], gt_dy)
        artifacts.append({k: str(v.relative_to(out)) for k, v in paths.items()})

    info = {
        "width": W,
        "height": H,
        "frame_count": FRAME_COUNT,
        "shot_split": SHOT_SPLIT,
        "shots": [[0, SHOT_SPLIT - 1], [SHOT_SPLIT, FRAME_COUNT - 1]],
        "layer_disparity_px": {LAYER_NAMES[k]: v for k, v in LAYER_DISPARITY.items()},
        "true_drift": {
            "shot_A": {"frames": [0, SHOT_SPLIT - 1], **DRIFT[0]},
            "shot_B": {"frames": [SHOT_SPLIT, FRAME_COUNT - 1], **DRIFT[1]},
        },
        "convention": "right_x = left_x - dx_layer; right_y = left_y + dy(x)",
        "frames": artifacts,
    }
    (out / "scene_info.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "output/footage"
    info = generate(target)
    print(f"generated {info['frame_count']} stereo frame pairs in {target}")
