import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from editing.diffusion_prior_finetune import build_camera_set, ensure_runtime_imports, load_scene_metadata
import editing.diffusion_prior_finetune as ft


def load_mask(path: Path, size: tuple[int, int], dilation: int) -> np.ndarray:
    image = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    if dilation > 1:
        kernel = int(dilation)
        if kernel % 2 == 0:
            kernel += 1
        image = image.filter(ImageFilter.MaxFilter(kernel))
    return np.asarray(image, dtype=np.uint8) > 127


def choose_source_camera(cameras):
    for camera in cameras:
        if (camera.name or "").lower() == "source":
            return camera
    if not cameras:
        raise ValueError("scene metadata produced zero cameras")
    return cameras[0]


def inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-8), 1.0 - 1e-8)
    return float(np.log(value / (1.0 - value)))


def project_points(points: np.ndarray, w2c: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    homog = np.concatenate([points.astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    camera_points = (w2c @ homog.T).T[:, :3]
    z = camera_points[:, 2]
    valid = z > 1e-6
    pixels = np.zeros((points.shape[0], 2), dtype=np.float32)
    pixels[valid, 0] = intrinsic[0, 0] * (camera_points[valid, 0] / z[valid]) + intrinsic[0, 2]
    pixels[valid, 1] = intrinsic[1, 1] * (camera_points[valid, 1] / z[valid]) + intrinsic[1, 2]
    return pixels, valid


@torch.no_grad()
def prune(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("prune_object_by_source_mask requires CUDA because GaussianModel.load_ply uses CUDA tensors")
    ensure_runtime_imports()

    with Image.open(args.source_image) as source_image:
        width, height = source_image.size
    size = (width, height)
    metadata = load_scene_metadata(args.scene_meta)
    camera = choose_source_camera(build_camera_set(metadata, width, height, nearby_views=0))
    mask = load_mask(Path(args.edit_mask), size, args.mask_dilate)

    edit_state = np.load(args.edit_state)
    if "editable_mask" not in edit_state:
        raise KeyError(f"{args.edit_state} does not contain editable_mask")
    editable = edit_state["editable_mask"].astype(bool)

    gaussians = ft.GaussianModel(0, with_motion_mask=False)
    gaussians.load_ply(args.gs_path)
    if editable.shape[0] != gaussians.get_xyz.shape[0]:
        raise ValueError(f"editable_mask length {editable.shape[0]} does not match Gaussian count {gaussians.get_xyz.shape[0]}")

    points = gaussians.get_xyz.detach().cpu().numpy()
    pixels, valid = project_points(points, camera.w2c, camera.intrinsic)
    u = np.rint(pixels[:, 0]).astype(np.int64)
    v = np.rint(pixels[:, 1]).astype(np.int64)
    in_bounds = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    inside_mask = np.zeros(points.shape[0], dtype=bool)
    inside_mask[in_bounds] = mask[v[in_bounds], u[in_bounds]]

    prune_mask = editable & in_bounds & (~inside_mask)
    keep_editable = editable & (~prune_mask)
    opacity_logit = inverse_sigmoid(args.min_opacity)
    torch_prune = torch.from_numpy(prune_mask).cuda()
    gaussians._opacity[torch_prune, 0] = opacity_logit

    out_ply = Path(args.out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(out_ply))

    out_edit_state = Path(args.out_edit_state) if args.out_edit_state else out_ply.with_suffix(".edit_state.npz")
    out_edit_state.parent.mkdir(parents=True, exist_ok=True)
    state = {key: edit_state[key] for key in edit_state.files}
    state["editable_mask"] = keep_editable.astype(np.bool_)
    state["source_visible_mask"] = inside_mask.astype(np.bool_)
    state["source_pruned_mask"] = prune_mask.astype(np.bool_)
    np.savez_compressed(out_edit_state, **state)

    summary = {
        "input_ply": str(args.gs_path),
        "output_ply": str(out_ply),
        "input_edit_state": str(args.edit_state),
        "output_edit_state": str(out_edit_state),
        "source_image": str(args.source_image),
        "edit_mask": str(args.edit_mask),
        "total_gaussians": int(points.shape[0]),
        "input_editable": int(editable.sum()),
        "kept_editable": int(keep_editable.sum()),
        "pruned_editable": int(prune_mask.sum()),
        "min_opacity": float(args.min_opacity),
    }
    summary_path = out_ply.with_suffix(".prune_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return out_ply


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hide source-visible object Gaussians that fall outside a clean source-view mask")
    parser.add_argument("--gs-path", required=True)
    parser.add_argument("--edit-state", required=True)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--edit-mask", required=True)
    parser.add_argument("--scene-meta", required=True)
    parser.add_argument("--out-ply", required=True)
    parser.add_argument("--out-edit-state", default="")
    parser.add_argument("--mask-dilate", type=int, default=3)
    parser.add_argument("--min-opacity", type=float, default=1e-4)
    return parser.parse_args(argv)


if __name__ == "__main__":
    output = prune(parse_args())
    print(f"PRUNED_PLY={output}", flush=True)
