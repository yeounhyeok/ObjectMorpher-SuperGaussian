import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from editing.utils.layered_scene_utils import merge_layered_gaussians


def _read_vertices(path: Path) -> np.ndarray:
    return PlyData.read(str(path))["vertex"].data


def _write_vertices(path: Path, vertices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def _load_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _w2c_from_meta(meta: dict) -> np.ndarray:
    extrinsic = np.asarray(meta.get("extrinsic", np.eye(4)), dtype=np.float32)
    if extrinsic.size == 12:
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3] = extrinsic.reshape(3, 4)
        return w2c
    return extrinsic.reshape(4, 4)


def _mask_bbox(mask_path: Path, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    mask = Image.open(mask_path).convert("L").resize(image_size, Image.Resampling.NEAREST)
    mask_np = np.asarray(mask) > 127
    ys, xs = np.where(mask_np)
    if xs.size == 0:
        raise ValueError(f"Mask is empty: {mask_path}")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _estimate_mask_depth(reference_gs: Path, mask_path: Path, meta: dict) -> float:
    width, height = [int(v) for v in meta["image_size"]]
    mask = Image.open(mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)
    mask_np = np.asarray(mask) > 127
    intrinsic = np.asarray(meta["intrinsic"], dtype=np.float32).reshape(3, 3)
    w2c = _w2c_from_meta(meta)

    vertices = _read_vertices(reference_gs)
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    homog = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    camera_points = (w2c @ homog.T).T[:, :3]
    z = camera_points[:, 2]
    valid_z = z > 1e-6
    u = intrinsic[0, 0] * (camera_points[:, 0] / np.maximum(z, 1e-6)) + intrinsic[0, 2]
    v = intrinsic[1, 1] * (camera_points[:, 1] / np.maximum(z, 1e-6)) + intrinsic[1, 2]
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    in_bounds = valid_z & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    selected = in_bounds & mask_np[np.clip(vi, 0, height - 1), np.clip(ui, 0, width - 1)]
    if int(selected.sum()) < 100:
        raise ValueError(f"Could not estimate reliable mask depth from {reference_gs}; selected={int(selected.sum())}")
    return float(np.median(z[selected]))


def _copy_vertices(vertices: np.ndarray) -> np.ndarray:
    copied = np.empty(vertices.shape, dtype=vertices.dtype)
    for name in vertices.dtype.names or ():
        copied[name] = vertices[name]
    return copied


def _transform_trellis_object(
    object_ply: Path,
    output_ply: Path,
    *,
    meta: dict,
    mask_bbox: tuple[int, int, int, int],
    depth: float,
    scale_multiplier: float,
) -> tuple[Path, Path, dict]:
    vertices = _copy_vertices(_read_vertices(object_ply))
    names = vertices.dtype.names or ()
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)

    object_min = xyz.min(axis=0)
    object_max = xyz.max(axis=0)
    object_center = (object_min + object_max) * 0.5
    object_width = max(float(object_max[0] - object_min[0]), 1e-6)

    x0, y0, x1, y1 = mask_bbox
    cx_px = 0.5 * (x0 + x1)
    cy_px = 0.5 * (y0 + y1)
    mask_width_px = max(float(x1 - x0 + 1), 1.0)
    intrinsic = np.asarray(meta["intrinsic"], dtype=np.float32).reshape(3, 3)
    w2c = _w2c_from_meta(meta)
    c2w = np.linalg.inv(w2c)

    target_camera = np.asarray(
        [
            (cx_px - intrinsic[0, 2]) * depth / intrinsic[0, 0],
            (cy_px - intrinsic[1, 2]) * depth / intrinsic[1, 1],
            depth,
            1.0,
        ],
        dtype=np.float32,
    )
    target_world = (c2w @ target_camera)[:3]
    target_width_world = mask_width_px * depth / intrinsic[0, 0]
    scale = float(target_width_world / object_width) * float(scale_multiplier)

    transformed = (xyz - object_center[None, :]) * scale + target_world[None, :]
    vertices["x"] = transformed[:, 0]
    vertices["y"] = transformed[:, 1]
    vertices["z"] = transformed[:, 2]

    log_scale = math.log(max(scale, 1e-6))
    for name in names:
        if name.startswith("scale_"):
            vertices[name] = vertices[name] + log_scale

    _write_vertices(output_ply, vertices)
    edit_state = output_ply.with_suffix(".edit_state.npz")
    np.savez_compressed(edit_state, editable_mask=np.ones(vertices.shape[0], dtype=bool))
    summary = {
        "target_world": target_world.tolist(),
        "mask_bbox": list(mask_bbox),
        "depth": depth,
        "object_scale": scale,
        "object_input_bbox": {"min": object_min.tolist(), "max": object_max.tolist()},
    }
    return output_ply, edit_state, summary


def _scale_background(background_ply: Path, output_ply: Path, *, meta: dict, scale: float) -> Path:
    vertices = _copy_vertices(_read_vertices(background_ply))
    if abs(scale - 1.0) < 1e-6:
        _write_vertices(output_ply, vertices)
        return output_ply

    w2c = _w2c_from_meta(meta)
    c2w = np.linalg.inv(w2c)
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    homog = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    camera_points = (w2c @ homog.T).T
    camera_points[:, 0] *= scale
    camera_points[:, 1] *= scale
    world_points = (c2w @ camera_points.T).T[:, :3]
    vertices["x"] = world_points[:, 0]
    vertices["y"] = world_points[:, 1]
    vertices["z"] = world_points[:, 2]

    log_scale = math.log(max(scale, 1e-6))
    for name in vertices.dtype.names or ():
        if name.startswith("scale_"):
            vertices[name] = vertices[name] + log_scale
    _write_vertices(output_ply, vertices)
    return output_ply


def _look_at_w2c(camera_position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target.astype(np.float32) - camera_position.astype(np.float32)
    forward = forward / max(float(np.linalg.norm(forward)), 1e-6)
    down_hint = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(down_hint, forward)
    right = right / max(float(np.linalg.norm(right)), 1e-6)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=0)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = -rotation @ camera_position.astype(np.float32)
    return w2c


def _yaw_views(meta: dict, target_world: list[float], yaws: str) -> list[dict]:
    base_w2c = _w2c_from_meta(meta)
    base_c2w = np.linalg.inv(base_w2c)
    source_center = base_c2w[:3, 3]
    target = np.asarray(target_world, dtype=np.float32)
    intrinsic = meta["intrinsic"]
    image_size = meta["image_size"]
    views = [
        {
            "name": "source",
            "image_size": image_size,
            "intrinsic": intrinsic,
            "extrinsic": base_w2c.tolist(),
            "extrinsic_convention": "world_to_camera",
        }
    ]
    source_vec = source_center - target
    for raw_yaw in [item.strip() for item in yaws.split(",") if item.strip()]:
        yaw = float(raw_yaw)
        if abs(yaw) < 1e-6:
            continue
        theta = math.radians(yaw)
        rot_y = np.asarray(
            [
                [math.cos(theta), 0.0, math.sin(theta)],
                [0.0, 1.0, 0.0],
                [-math.sin(theta), 0.0, math.cos(theta)],
            ],
            dtype=np.float32,
        )
        camera_position = target + rot_y @ source_vec
        views.append(
            {
                "name": f"yaw_{yaw:+.0f}",
                "image_size": image_size,
                "intrinsic": intrinsic,
                "extrinsic": _look_at_w2c(camera_position, target).tolist(),
                "extrinsic_convention": "world_to_camera",
            }
        )
    return views


def compose(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _load_meta(Path(args.scene_meta))
    image_size = tuple(int(v) for v in meta["image_size"])
    bbox = _mask_bbox(Path(args.edit_mask), image_size)
    depth = float(args.depth) if args.depth > 0 else _estimate_mask_depth(Path(args.reference_gs), Path(args.edit_mask), meta)

    foreground_ply, foreground_state, fg_summary = _transform_trellis_object(
        Path(args.object_gs),
        out_dir / "trellis_object_aligned.ply",
        meta=meta,
        mask_bbox=bbox,
        depth=depth,
        scale_multiplier=args.object_scale,
    )
    background_ply = _scale_background(
        Path(args.background_gs),
        out_dir / f"background_scaled_{args.background_scale:.2f}.ply",
        meta=meta,
        scale=args.background_scale,
    )

    merged_ply = out_dir / "layered_trellis_background.ply"
    merged_meta = out_dir / "scene_meta.json"
    merged_state = out_dir / "layered_trellis_background.edit_state.npz"
    summary = merge_layered_gaussians(
        background_ply=background_ply,
        foreground_ply=foreground_ply,
        edit_state=foreground_state,
        output_ply=merged_ply,
        output_edit_state=merged_state,
        output_meta=merged_meta,
        background_scene_meta=Path(args.scene_meta),
        source_image=args.source_image,
    )

    composed_meta = {**meta, **summary}
    composed_meta["format"] = "pixelhacker_background_trellis_object_layered"
    composed_meta["source_image"] = args.source_image
    composed_meta["object_gs"] = str(Path(args.object_gs))
    composed_meta["object_alignment"] = fg_summary
    composed_meta["background_scale"] = args.background_scale
    composed_meta["views"] = _yaw_views(meta, fg_summary["target_world"], args.view_yaws)
    merged_meta.write_text(json.dumps(composed_meta, indent=2), encoding="utf-8")

    print(f"ALIGNED_OBJECT_PLY={foreground_ply}", flush=True)
    print(f"SCALED_BACKGROUND_PLY={background_ply}", flush=True)
    print(f"LAYERED_TRELLIS_PLY={merged_ply}", flush=True)
    print(f"LAYERED_TRELLIS_META={merged_meta}", flush=True)
    print(f"LAYERED_TRELLIS_EDIT_STATE={merged_state}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose PixelHacker background 3DGS with a TRELLIS object 3DGS")
    parser.add_argument("--background-gs", required=True)
    parser.add_argument("--object-gs", required=True)
    parser.add_argument("--scene-meta", required=True)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--edit-mask", required=True)
    parser.add_argument("--reference-gs", required=True, help="Original full-scene GS used only to estimate mask depth")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--depth", type=float, default=-1.0)
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--background-scale", type=float, default=1.0)
    parser.add_argument("--view-yaws", default="-30,-15,15,30")
    return parser.parse_args()


if __name__ == "__main__":
    compose(parse_args())
