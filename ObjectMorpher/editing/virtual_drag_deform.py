import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

EDITING_DIR = Path(__file__).resolve().parent
if str(EDITING_DIR) not in sys.path:
    sys.path.insert(0, str(EDITING_DIR))

from utils.scene_lifting_utils import (  # noqa: E402
    editable_mask_from_projection,
    load_scene_metadata,
    project_points_to_pixels,
)


def _load_gaussian_model():
    module_path = EDITING_DIR / "scene" / "gaussian_model.py"
    spec = importlib.util.spec_from_file_location("objectmorpher_virtual_drag_gaussian_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load GaussianModel from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GaussianModel


def _build_union_mask(
    original_mask_path: Path,
    edited_xyz: np.ndarray,
    editable_mask: np.ndarray,
    metadata: dict,
    output_path: Path,
    dilation: int,
) -> None:
    original = Image.open(original_mask_path).convert("L")
    width, height = original.size
    original_np = np.asarray(original) > 127
    pixels, valid = project_points_to_pixels(edited_xyz, metadata, (width, height))
    u = np.rint(pixels[:, 0]).astype(np.int64)
    v = np.rint(pixels[:, 1]).astype(np.int64)
    in_bounds = valid & editable_mask & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    union = original_np.copy()
    union[v[in_bounds], u[in_bounds]] = True
    union_image = Image.fromarray((union.astype(np.uint8) * 255), mode="L")
    if dilation > 1:
        if dilation % 2 == 0:
            dilation += 1
        union_image = union_image.filter(ImageFilter.MaxFilter(dilation))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    union_image.save(output_path)


def apply_virtual_drag(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("virtual_drag_deform requires CUDA because GaussianModel stores tensors on CUDA")

    GaussianModel = _load_gaussian_model()
    metadata = load_scene_metadata(args.scene_meta)

    gaussians = GaussianModel(0, with_motion_mask=False)
    gaussians.load_ply(args.gs_path)
    editable_mask = editable_mask_from_projection(
        gaussians.get_xyz,
        args.edit_mask,
        metadata,
        opacity=gaussians.get_opacity[:, 0],
    )
    editable_np = editable_mask.detach().cpu().numpy().astype(bool)
    if editable_np.sum() == 0:
        raise ValueError("Editable mask selected zero Gaussians")

    with Image.open(args.edit_mask).convert("L") as mask_image:
        width, height = mask_image.size
        mask_np = np.asarray(mask_image) > 127
    ys, xs = np.where(mask_np)
    if xs.size == 0:
        raise ValueError(f"Mask has no foreground pixels: {args.edit_mask}")

    bbox_min = np.array([xs.min(), ys.min()], dtype=np.float32)
    bbox_max = np.array([xs.max(), ys.max()], dtype=np.float32)
    bbox_size = bbox_max - bbox_min
    handle_uv = bbox_min + np.array([args.handle_x, args.handle_y], dtype=np.float32) * bbox_size
    drag_uv = np.array([args.drag_x, args.drag_y], dtype=np.float32)

    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
    pixels, valid_depth = project_points_to_pixels(xyz, metadata, (width, height))
    pixel_delta = pixels - handle_uv[None, :]
    dist2 = np.sum(pixel_delta * pixel_delta, axis=1)
    weights = np.exp(-0.5 * dist2 / max(args.radius_px * args.radius_px, 1e-6)).astype(np.float32)
    weights *= editable_np.astype(np.float32)
    weights *= valid_depth.astype(np.float32)

    intrinsic = np.asarray(metadata["intrinsic"], dtype=np.float32).reshape(3, 3).copy()
    meta_width, meta_height = metadata.get("image_size", [width, height])
    if int(meta_width) != width or int(meta_height) != height:
        intrinsic[0, :] *= width / float(meta_width)
        intrinsic[1, :] *= height / float(meta_height)

    extrinsic = np.asarray(metadata.get("extrinsic", np.eye(4)), dtype=np.float32)
    if extrinsic.size == 12:
        full = np.eye(4, dtype=np.float32)
        full[:3] = extrinsic.reshape(3, 4)
        extrinsic = full
    else:
        extrinsic = extrinsic.reshape(4, 4)

    homog = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    camera_xyz = (extrinsic @ homog.T).T[:, :3]
    z = np.maximum(camera_xyz[:, 2], 1e-6)
    camera_delta = np.zeros_like(xyz, dtype=np.float32)
    camera_delta[:, 0] = drag_uv[0] * z / max(float(intrinsic[0, 0]), 1e-6)
    camera_delta[:, 1] = drag_uv[1] * z / max(float(intrinsic[1, 1]), 1e-6)

    world_delta = camera_delta @ extrinsic[:3, :3]
    d_xyz = world_delta * weights[:, None]
    edited_xyz = xyz + d_xyz

    gaussians._xyz = torch.nn.Parameter(torch.from_numpy(edited_xyz).to(device="cuda", dtype=torch.float32))
    out_ply = Path(args.out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(out_ply))

    edit_state_path = Path(args.edit_state)
    edit_state_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        edit_state_path,
        editable_mask=editable_np.astype(np.bool_),
        initial_xyz=xyz,
        edited_xyz=edited_xyz,
        d_xyz=d_xyz,
        weights=weights,
        handle_uv=handle_uv,
        drag_uv=drag_uv,
        radius_px=np.asarray([args.radius_px], dtype=np.float32),
    )

    if args.out_mask:
        _build_union_mask(
            Path(args.edit_mask),
            edited_xyz,
            editable_np,
            metadata,
            Path(args.out_mask),
            args.mask_dilation,
        )

    moved = np.linalg.norm(d_xyz, axis=1)
    print(f"VIRTUAL_DRAG_PLY={out_ply}")
    print(f"EDIT_STATE={edit_state_path}")
    if args.out_mask:
        print(f"UNION_MASK={Path(args.out_mask)}")
    print(f"EDITABLE={int(editable_np.sum())}/{editable_np.size}")
    print(f"HANDLE_UV={handle_uv.tolist()} DRAG_UV={drag_uv.tolist()} RADIUS_PX={args.radius_px}")
    print(f"MAX_DISPLACEMENT={float(moved.max()):.6f} MEAN_EDITABLE_DISPLACEMENT={float(moved[editable_np].mean()):.6f}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a programmatic virtual drag to editable object Gaussians")
    parser.add_argument("--gs-path", required=True)
    parser.add_argument("--edit-mask", required=True)
    parser.add_argument("--scene-meta", required=True)
    parser.add_argument("--out-ply", required=True)
    parser.add_argument("--edit-state", required=True)
    parser.add_argument("--out-mask", default="")
    parser.add_argument("--handle-x", type=float, default=0.78, help="Handle x as a fraction of the 2D mask bbox")
    parser.add_argument("--handle-y", type=float, default=0.30, help="Handle y as a fraction of the 2D mask bbox")
    parser.add_argument("--drag-x", type=float, default=45.0, help="Virtual drag in source-view pixels")
    parser.add_argument("--drag-y", type=float, default=-32.0, help="Virtual drag in source-view pixels")
    parser.add_argument("--radius-px", type=float, default=135.0)
    parser.add_argument("--mask-dilation", type=int, default=35)
    return parser.parse_args(argv)


if __name__ == "__main__":
    apply_virtual_drag(parse_args())
