import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

EDITING_DIR = Path(__file__).resolve().parent
if str(EDITING_DIR) not in sys.path:
    sys.path.insert(0, str(EDITING_DIR))

from edit_gui import NodeDriver, farthest_point_sample, quaternion_multiply  # noqa: E402
from lap_deform import LapDeform  # noqa: E402
from utils.scene_lifting_utils import (  # noqa: E402
    editable_mask_from_projection,
    gate_deformation_values,
    load_scene_metadata,
    project_points_to_pixels,
)


def _load_gaussian_model():
    module_path = EDITING_DIR / "scene" / "gaussian_model.py"
    spec = importlib.util.spec_from_file_location("objectmorpher_programmatic_arap_gaussian_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load GaussianModel from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GaussianModel


def camera_pixel_delta_to_world(
    points: np.ndarray,
    metadata: dict,
    image_size: tuple[int, int],
    pixel_delta: np.ndarray,
) -> np.ndarray:
    width, height = image_size
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

    homog = np.concatenate([points.astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    camera_points = (extrinsic @ homog.T).T[:, :3]
    z = np.maximum(camera_points[:, 2], 1e-6)
    pixel_delta = np.asarray(pixel_delta, dtype=np.float32)
    if pixel_delta.ndim == 1:
        pixel_delta = np.repeat(pixel_delta[None], points.shape[0], axis=0)
    if pixel_delta.shape != (points.shape[0], 2):
        raise ValueError(f"Expected pixel_delta shape {(points.shape[0], 2)} or (2,), got {pixel_delta.shape}")
    camera_delta = np.zeros_like(points, dtype=np.float32)
    camera_delta[:, 0] = pixel_delta[:, 0] * z / max(float(intrinsic[0, 0]), 1e-6)
    camera_delta[:, 1] = pixel_delta[:, 1] * z / max(float(intrinsic[1, 1]), 1e-6)

    # Metadata stores world-to-camera. A camera-space displacement maps through R_wc^T.
    return camera_delta @ extrinsic[:3, :3]


def mask_bbox(mask_path: Path) -> tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray]:
    image = Image.open(mask_path).convert("L")
    mask = np.asarray(image) > 127
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError(f"Mask is empty: {mask_path}")
    bbox_min = np.array([xs.min(), ys.min()], dtype=np.float32)
    bbox_max = np.array([xs.max(), ys.max()], dtype=np.float32)
    return image, mask, bbox_min, bbox_max


def unique_tensor(items: Iterable[torch.Tensor], device: torch.device) -> torch.Tensor:
    tensors = [item.reshape(-1).to(device=device, dtype=torch.long) for item in items if item.numel() > 0]
    if not tensors:
        return torch.empty((0,), dtype=torch.long, device=device)
    return torch.unique(torch.cat(tensors, dim=0))


def bbox_from_projected_points(pixels: np.ndarray, valid: np.ndarray, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    in_bounds = (
        valid
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(width))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(height))
    )
    if not np.any(in_bounds):
        raise ValueError("Could not build projected bbox: no valid projected points")
    selected = pixels[in_bounds]
    return selected.min(axis=0).astype(np.float32), selected.max(axis=0).astype(np.float32)


def nearest_valid_node(
    node_pixels: np.ndarray,
    valid: np.ndarray,
    target_uv: np.ndarray,
    exclude: set[int] | None = None,
) -> int:
    exclude = exclude or set()
    candidates = np.where(valid)[0]
    if candidates.size == 0:
        candidates = np.arange(node_pixels.shape[0])
    if exclude:
        candidates = np.array([idx for idx in candidates.tolist() if idx not in exclude], dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("No available control node candidates")
    score = np.sum((node_pixels[candidates] - target_uv[None, :]) ** 2, axis=1)
    return int(candidates[int(np.argmin(score))])


def select_handle_groups(
    animate_tool: LapDeform,
    control_nodes: torch.Tensor,
    node_pixels: np.ndarray,
    node_valid: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    bbox_size = np.maximum(bbox_max - bbox_min, 1.0)
    tail_uv = bbox_min + np.array([args.tail_x, args.tail_y], dtype=np.float32) * bbox_size

    tail_region = (
        node_valid
        & (node_pixels[:, 0] < bbox_min[0] + args.tail_region_x * bbox_size[0])
        & (node_pixels[:, 1] > bbox_min[1] + args.tail_region_y * bbox_size[1])
    )
    tail_candidates = np.where(tail_region)[0]
    if tail_candidates.size == 0:
        tail_seed = nearest_valid_node(node_pixels, node_valid, tail_uv)
    else:
        score = np.sum((node_pixels[tail_candidates] - tail_uv[None, :]) ** 2, axis=1)
        tail_seed = int(tail_candidates[int(np.argmin(score))])

    tail_group = animate_tool.add_n_ring_nbs(torch.tensor([tail_seed], device=control_nodes.device), n=args.tail_n_rings)

    # Programmatic equivalent of adding several zero-delta keypoints in the GUI:
    # these keep the body/head/legs fixed while the selected tail keypoint moves.
    anchor_targets = [
        (0.30, 0.36),
        (0.42, 0.45),
        (0.58, 0.42),
        (0.76, 0.31),
        (0.86, 0.50),
        (0.50, 0.63),
        (0.27, 0.70),
    ]
    excluded = set(int(x) for x in tail_group.detach().cpu().numpy().tolist())
    anchor_groups = []
    anchor_seeds = []
    for rel_x, rel_y in anchor_targets[: args.anchor_seeds]:
        uv = bbox_min + np.array([rel_x, rel_y], dtype=np.float32) * bbox_size
        seed = nearest_valid_node(node_pixels, node_valid, uv, exclude=excluded)
        anchor_seeds.append(seed)
        group = animate_tool.add_n_ring_nbs(torch.tensor([seed], device=control_nodes.device), n=args.anchor_n_rings)
        anchor_groups.append(group)
        excluded.update(int(x) for x in group.detach().cpu().numpy().tolist())

    anchor_group = unique_tensor(anchor_groups, control_nodes.device)
    tail_group = torch.unique(tail_group.to(dtype=torch.long))
    anchor_group = anchor_group[~torch.isin(anchor_group, tail_group)]

    metadata = {
        "tail_seed": tail_seed,
        "tail_group": tail_group.detach().cpu().numpy().astype(int).tolist(),
        "anchor_seeds": [int(x) for x in anchor_seeds],
        "anchor_group": anchor_group.detach().cpu().numpy().astype(int).tolist(),
        "tail_uv": tail_uv.astype(float).tolist(),
    }
    return tail_group, anchor_group, metadata


def load_editable_mask_for_arap(args: argparse.Namespace, gaussians, metadata: dict) -> torch.Tensor:
    if args.input_edit_state:
        edit_state = np.load(args.input_edit_state)
        if "editable_mask" not in edit_state:
            raise KeyError(f"Input edit state has no editable_mask: {args.input_edit_state}")
        editable = torch.from_numpy(edit_state["editable_mask"].astype(np.bool_)).cuda()
        if editable.shape[0] != gaussians.get_xyz.shape[0]:
            raise ValueError(
                f"Input edit state editable_mask length {editable.shape[0]} "
                f"does not match Gaussian count {gaussians.get_xyz.shape[0]}"
            )
        return editable

    return editable_mask_from_projection(
        gaussians.get_xyz,
        args.edit_mask,
        metadata,
        opacity=gaussians.get_opacity[:, 0],
        min_opacity=args.min_opacity,
    )


def select_tip_up_handles(
    animate_tool: LapDeform,
    control_nodes: torch.Tensor,
    node_pixels: np.ndarray,
    node_valid: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    metadata: dict,
    image_size: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    bbox_size = np.maximum(bbox_max - bbox_min, 1.0)
    tail_x_min = bbox_min[0] + args.tail_tip_region_x_min * bbox_size[0]
    tail_x_max = bbox_min[0] + args.tail_tip_region_x_max * bbox_size[0]
    tail_y_min = bbox_min[1] + args.tail_tip_region_y_min * bbox_size[1]
    tail_region = (
        node_valid
        & (node_pixels[:, 0] >= tail_x_min)
        & (node_pixels[:, 0] <= tail_x_max)
        & (node_pixels[:, 1] >= tail_y_min)
    )
    tail_local = np.where(tail_region)[0]
    if tail_local.size < 4:
        raise ValueError(
            f"Could not find enough tail control nodes for tip-up pose; found {tail_local.size}. "
            "Adjust --tail-tip-region-* bounds."
        )

    tail_pixels = node_pixels[tail_local]
    base_uv = np.array(
        [
            np.median(tail_pixels[:, 0]),
            bbox_min[1] + args.tail_base_y * bbox_size[1],
        ],
        dtype=np.float32,
    )
    tip_current_y = float(tail_pixels[:, 1].max())
    tip_target_uv = bbox_min + np.array([args.tip_target_x, args.tip_target_y], dtype=np.float32) * bbox_size

    alpha = (tail_pixels[:, 1] - base_uv[1]) / max(tip_current_y - base_uv[1], 1.0)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    tail_move_local = tail_local[alpha >= args.tail_move_min_alpha]
    tail_move_alpha = alpha[alpha >= args.tail_move_min_alpha]
    if tail_move_local.size < 2:
        raise ValueError("Tip-up pose selected too few moving tail handles")

    # Base handles pin the attached part of the tail; body anchors pin the cat.
    base_local = tail_local[alpha < args.tail_move_min_alpha]
    if base_local.size == 0:
        base_seed = int(tail_local[int(np.argmin(np.abs(tail_pixels[:, 1] - base_uv[1])))])
        base_group = animate_tool.add_n_ring_nbs(torch.tensor([base_seed], device=control_nodes.device), n=args.anchor_n_rings)
    else:
        base_group = torch.from_numpy(base_local).to(device=control_nodes.device, dtype=torch.long)

    anchor_targets = [
        (0.30, 0.36),
        (0.42, 0.45),
        (0.58, 0.42),
        (0.76, 0.31),
        (0.86, 0.50),
        (0.50, 0.63),
    ]
    excluded = set(int(x) for x in tail_local.tolist())
    anchor_groups = [base_group]
    anchor_seeds = []
    for rel_x, rel_y in anchor_targets[: args.anchor_seeds]:
        uv = bbox_min + np.array([rel_x, rel_y], dtype=np.float32) * bbox_size
        seed = nearest_valid_node(node_pixels, node_valid, uv, exclude=excluded)
        anchor_seeds.append(seed)
        group = animate_tool.add_n_ring_nbs(torch.tensor([seed], device=control_nodes.device), n=args.anchor_n_rings)
        anchor_groups.append(group)
        excluded.update(int(x) for x in group.detach().cpu().numpy().tolist())

    moving_idx = torch.from_numpy(tail_move_local).to(device=control_nodes.device, dtype=torch.long)
    anchor_idx = unique_tensor(anchor_groups, control_nodes.device)
    anchor_idx = anchor_idx[~torch.isin(anchor_idx, moving_idx)]
    handle_idx = torch.unique(torch.cat([moving_idx, anchor_idx], dim=0))
    handle_pos = control_nodes[handle_idx].detach().clone()

    move_mask = torch.isin(handle_idx, moving_idx)
    moving_handle_idx = handle_idx[move_mask]
    moving_pixels = node_pixels[moving_handle_idx.detach().cpu().numpy()]
    moving_alpha = (moving_pixels[:, 1] - base_uv[1]) / max(tip_current_y - base_uv[1], 1.0)
    moving_alpha = np.clip(moving_alpha, 0.0, 1.0).astype(np.float32)
    eased_alpha = np.power(moving_alpha, args.tip_alpha_gamma).astype(np.float32)
    target_pixels = base_uv[None, :] + eased_alpha[:, None] * (tip_target_uv[None, :] - base_uv[None, :])
    if args.tip_curve_x != 0.0:
        target_pixels[:, 0] += args.tip_curve_x * bbox_size[0] * np.sin(np.pi * eased_alpha)
    deltas_uv = target_pixels - moving_pixels
    deltas_world = camera_pixel_delta_to_world(
        control_nodes[moving_handle_idx].detach().cpu().numpy(),
        metadata,
        image_size,
        deltas_uv,
    )
    handle_pos[move_mask] = handle_pos[move_mask] + torch.from_numpy(deltas_world).to(
        device=handle_pos.device,
        dtype=handle_pos.dtype,
    )

    debug = {
        "tail_pose": "tip_up",
        "tail_region_count": int(tail_local.size),
        "tail_move_count": int(tail_move_local.size),
        "tail_base_count": int(base_group.numel()),
        "tail_region": {
            "x_min": float(tail_x_min),
            "x_max": float(tail_x_max),
            "y_min": float(tail_y_min),
        },
        "base_uv": base_uv.astype(float).tolist(),
        "tip_current_y": tip_current_y,
        "tip_target_uv": tip_target_uv.astype(float).tolist(),
        "anchor_seeds": [int(x) for x in anchor_seeds],
        "tail_group": moving_idx.detach().cpu().numpy().astype(int).tolist(),
        "anchor_group": anchor_idx.detach().cpu().numpy().astype(int).tolist(),
    }
    return handle_idx, handle_pos, moving_idx, anchor_idx, debug


def select_tip_up_left_handles(
    animate_tool: LapDeform,
    control_nodes: torch.Tensor,
    node_pixels: np.ndarray,
    node_valid: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    metadata: dict,
    image_size: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    bbox_size = np.maximum(bbox_max - bbox_min, 1.0)
    tail_x_min = bbox_min[0] + args.tail_tip_region_x_min * bbox_size[0]
    tail_x_max = bbox_min[0] + args.tail_tip_region_x_max * bbox_size[0]
    tail_y_min = bbox_min[1] + args.tail_tip_region_y_min * bbox_size[1]
    tail_y_max = bbox_min[1] + args.tail_tip_region_y_max * bbox_size[1]
    tail_region = (
        node_valid
        & (node_pixels[:, 0] >= tail_x_min)
        & (node_pixels[:, 0] <= tail_x_max)
        & (node_pixels[:, 1] >= tail_y_min)
        & (node_pixels[:, 1] <= tail_y_max)
    )
    tail_local = np.where(tail_region)[0]
    if tail_local.size < 4:
        raise ValueError(
            f"Could not find enough tail control nodes for left-tip pose; found {tail_local.size}. "
            "Adjust --tail-tip-region-* bounds."
        )

    tail_pixels = node_pixels[tail_local]
    base_uv = np.array(
        [
            bbox_min[0] + args.tail_base_x * bbox_size[0],
            float(np.median(tail_pixels[:, 1])),
        ],
        dtype=np.float32,
    )
    tip_current_x = float(tail_pixels[:, 0].min())
    tip_target_uv = bbox_min + np.array([args.tip_target_x, args.tip_target_y], dtype=np.float32) * bbox_size

    alpha = (base_uv[0] - tail_pixels[:, 0]) / max(base_uv[0] - tip_current_x, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    tail_move_local = tail_local[alpha >= args.tail_move_min_alpha]
    if tail_move_local.size < 2:
        raise ValueError("Left-tip pose selected too few moving tail handles")

    base_local = tail_local[alpha < args.tail_move_min_alpha]
    if base_local.size == 0:
        base_seed = int(tail_local[int(np.argmin(np.abs(tail_pixels[:, 0] - base_uv[0])))])
        base_group = animate_tool.add_n_ring_nbs(torch.tensor([base_seed], device=control_nodes.device), n=args.anchor_n_rings)
    else:
        base_group = torch.from_numpy(base_local).to(device=control_nodes.device, dtype=torch.long)

    anchor_targets = [
        (0.30, 0.36),
        (0.42, 0.45),
        (0.58, 0.42),
        (0.76, 0.31),
        (0.86, 0.50),
        (0.50, 0.63),
    ]
    excluded = set(int(x) for x in tail_local.tolist())
    anchor_groups = [base_group]
    anchor_seeds = []
    for rel_x, rel_y in anchor_targets[: args.anchor_seeds]:
        uv = bbox_min + np.array([rel_x, rel_y], dtype=np.float32) * bbox_size
        seed = nearest_valid_node(node_pixels, node_valid, uv, exclude=excluded)
        anchor_seeds.append(seed)
        group = animate_tool.add_n_ring_nbs(torch.tensor([seed], device=control_nodes.device), n=args.anchor_n_rings)
        anchor_groups.append(group)
        excluded.update(int(x) for x in group.detach().cpu().numpy().tolist())

    moving_idx = torch.from_numpy(tail_move_local).to(device=control_nodes.device, dtype=torch.long)
    anchor_idx = unique_tensor(anchor_groups, control_nodes.device)
    anchor_idx = anchor_idx[~torch.isin(anchor_idx, moving_idx)]
    handle_idx = torch.unique(torch.cat([moving_idx, anchor_idx], dim=0))
    handle_pos = control_nodes[handle_idx].detach().clone()

    move_mask = torch.isin(handle_idx, moving_idx)
    moving_handle_idx = handle_idx[move_mask]
    moving_pixels = node_pixels[moving_handle_idx.detach().cpu().numpy()]
    moving_alpha = (base_uv[0] - moving_pixels[:, 0]) / max(base_uv[0] - tip_current_x, 1.0)
    moving_alpha = np.clip(moving_alpha, 0.0, 1.0).astype(np.float32)
    eased_alpha = np.power(moving_alpha, args.tip_alpha_gamma).astype(np.float32)
    target_pixels = base_uv[None, :] + eased_alpha[:, None] * (tip_target_uv[None, :] - base_uv[None, :])
    if args.tip_curve_x != 0.0:
        target_pixels[:, 1] -= args.tip_curve_x * bbox_size[1] * np.sin(np.pi * eased_alpha)
    deltas_uv = target_pixels - moving_pixels
    deltas_world = camera_pixel_delta_to_world(
        control_nodes[moving_handle_idx].detach().cpu().numpy(),
        metadata,
        image_size,
        deltas_uv,
    )
    handle_pos[move_mask] = handle_pos[move_mask] + torch.from_numpy(deltas_world).to(
        device=handle_pos.device,
        dtype=handle_pos.dtype,
    )

    debug = {
        "tail_pose": "tip_up_left",
        "tail_region_count": int(tail_local.size),
        "tail_move_count": int(tail_move_local.size),
        "tail_base_count": int(base_group.numel()),
        "tail_region": {
            "x_min": float(tail_x_min),
            "x_max": float(tail_x_max),
            "y_min": float(tail_y_min),
            "y_max": float(tail_y_max),
        },
        "base_uv": base_uv.astype(float).tolist(),
        "tip_current_x": tip_current_x,
        "tip_target_uv": tip_target_uv.astype(float).tolist(),
        "anchor_seeds": [int(x) for x in anchor_seeds],
        "tail_group": moving_idx.detach().cpu().numpy().astype(int).tolist(),
        "anchor_group": anchor_idx.detach().cpu().numpy().astype(int).tolist(),
    }
    return handle_idx, handle_pos, moving_idx, anchor_idx, debug


def select_tail_wave_handles(
    animate_tool: LapDeform,
    control_nodes: torch.Tensor,
    node_pixels: np.ndarray,
    node_valid: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    metadata: dict,
    image_size: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    bbox_size = np.maximum(bbox_max - bbox_min, 1.0)
    tail_x_min = bbox_min[0] + args.tail_tip_region_x_min * bbox_size[0]
    tail_x_max = bbox_min[0] + args.tail_tip_region_x_max * bbox_size[0]
    tail_y_min = bbox_min[1] + args.tail_tip_region_y_min * bbox_size[1]
    tail_region = (
        node_valid
        & (node_pixels[:, 0] >= tail_x_min)
        & (node_pixels[:, 0] <= tail_x_max)
        & (node_pixels[:, 1] >= tail_y_min)
    )
    tail_local = np.where(tail_region)[0]
    if tail_local.size < 4:
        raise ValueError(
            f"Could not find enough tail control nodes for wave pose; found {tail_local.size}. "
            "Adjust --tail-tip-region-* bounds."
        )

    tail_pixels = node_pixels[tail_local]
    base_uv = np.array(
        [
            np.median(tail_pixels[:, 0]),
            bbox_min[1] + args.tail_base_y * bbox_size[1],
        ],
        dtype=np.float32,
    )
    tip_current_y = float(tail_pixels[:, 1].max())
    alpha = (tail_pixels[:, 1] - base_uv[1]) / max(tip_current_y - base_uv[1], 1.0)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    endpoint_mask = (alpha <= args.tail_wave_anchor_alpha) | (alpha >= 1.0 - args.tail_wave_anchor_alpha)
    moving_mask = ~endpoint_mask
    moving_local = tail_local[moving_mask]
    if moving_local.size < 2:
        raise ValueError("Tail wave pose selected too few moving tail handles")

    anchor_groups = [
        torch.from_numpy(tail_local[endpoint_mask]).to(device=control_nodes.device, dtype=torch.long)
    ]
    anchor_targets = [
        (0.30, 0.36),
        (0.42, 0.45),
        (0.58, 0.42),
        (0.76, 0.31),
        (0.86, 0.50),
        (0.50, 0.63),
    ]
    excluded = set(int(x) for x in tail_local.tolist())
    anchor_seeds = []
    for rel_x, rel_y in anchor_targets[: args.anchor_seeds]:
        uv = bbox_min + np.array([rel_x, rel_y], dtype=np.float32) * bbox_size
        seed = nearest_valid_node(node_pixels, node_valid, uv, exclude=excluded)
        anchor_seeds.append(seed)
        group = animate_tool.add_n_ring_nbs(torch.tensor([seed], device=control_nodes.device), n=args.anchor_n_rings)
        anchor_groups.append(group)
        excluded.update(int(x) for x in group.detach().cpu().numpy().tolist())

    moving_idx = torch.from_numpy(moving_local).to(device=control_nodes.device, dtype=torch.long)
    anchor_idx = unique_tensor(anchor_groups, control_nodes.device)
    anchor_idx = anchor_idx[~torch.isin(anchor_idx, moving_idx)]
    handle_idx = torch.unique(torch.cat([moving_idx, anchor_idx], dim=0))
    handle_pos = control_nodes[handle_idx].detach().clone()

    move_mask = torch.isin(handle_idx, moving_idx)
    moving_handle_idx = handle_idx[move_mask]
    moving_pixels = node_pixels[moving_handle_idx.detach().cpu().numpy()]
    moving_alpha = (moving_pixels[:, 1] - base_uv[1]) / max(tip_current_y - base_uv[1], 1.0)
    moving_alpha = np.clip(moving_alpha, 0.0, 1.0).astype(np.float32)
    wave = np.sin(np.pi * moving_alpha)
    deltas_uv = np.zeros((moving_pixels.shape[0], 2), dtype=np.float32)
    deltas_uv[:, 0] = args.tail_wave_x * bbox_size[0] * wave
    deltas_uv[:, 1] = args.tail_wave_y * bbox_size[1] * wave
    deltas_world = camera_pixel_delta_to_world(
        control_nodes[moving_handle_idx].detach().cpu().numpy(),
        metadata,
        image_size,
        deltas_uv,
    )
    handle_pos[move_mask] = handle_pos[move_mask] + torch.from_numpy(deltas_world).to(
        device=handle_pos.device,
        dtype=handle_pos.dtype,
    )

    debug = {
        "tail_pose": "tail_wave",
        "tail_region_count": int(tail_local.size),
        "tail_move_count": int(moving_local.size),
        "tail_endpoint_count": int(endpoint_mask.sum()),
        "tail_region": {
            "x_min": float(tail_x_min),
            "x_max": float(tail_x_max),
            "y_min": float(tail_y_min),
        },
        "base_uv": base_uv.astype(float).tolist(),
        "tip_current_y": tip_current_y,
        "wave_uv": [float(args.tail_wave_x), float(args.tail_wave_y)],
        "anchor_seeds": [int(x) for x in anchor_seeds],
        "tail_group": moving_idx.detach().cpu().numpy().astype(int).tolist(),
        "anchor_group": anchor_idx.detach().cpu().numpy().astype(int).tolist(),
    }
    return handle_idx, handle_pos, moving_idx, anchor_idx, debug


def select_ear_flick_handles(
    animate_tool: LapDeform,
    control_nodes: torch.Tensor,
    node_pixels: np.ndarray,
    node_valid: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    metadata: dict,
    image_size: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    bbox_size = np.maximum(bbox_max - bbox_min, 1.0)
    x_min = bbox_min[0] + args.ear_region_x_min * bbox_size[0]
    x_max = bbox_min[0] + args.ear_region_x_max * bbox_size[0]
    y_max = bbox_min[1] + args.ear_region_y_max * bbox_size[1]
    ear_region = node_valid & (node_pixels[:, 0] >= x_min) & (node_pixels[:, 0] <= x_max) & (node_pixels[:, 1] <= y_max)
    ear_local = np.where(ear_region)[0]
    if ear_local.size < 3:
        raise ValueError(
            f"Could not find enough ear/head control nodes; found {ear_local.size}. "
            "Adjust --ear-region-* bounds."
        )

    ear_pixels = node_pixels[ear_local]
    y_cut = np.quantile(ear_pixels[:, 1], args.ear_tip_quantile)
    moving_local = ear_local[ear_pixels[:, 1] <= y_cut]
    if moving_local.size < 2:
        moving_local = ear_local[np.argsort(ear_pixels[:, 1])[:2]]
    moving_idx = torch.from_numpy(moving_local).to(device=control_nodes.device, dtype=torch.long)

    anchor_targets = [
        (0.32, 0.36),
        (0.45, 0.45),
        (0.58, 0.43),
        (0.76, 0.50),
        (0.86, 0.50),
        (0.50, 0.64),
        (0.24, 0.84),
    ]
    excluded = set(int(x) for x in moving_local.tolist())
    anchor_groups = []
    anchor_seeds = []
    for rel_x, rel_y in anchor_targets[: args.anchor_seeds]:
        uv = bbox_min + np.array([rel_x, rel_y], dtype=np.float32) * bbox_size
        seed = nearest_valid_node(node_pixels, node_valid, uv, exclude=excluded)
        anchor_seeds.append(seed)
        group = animate_tool.add_n_ring_nbs(torch.tensor([seed], device=control_nodes.device), n=args.anchor_n_rings)
        anchor_groups.append(group)
        excluded.update(int(x) for x in group.detach().cpu().numpy().tolist())

    anchor_idx = unique_tensor(anchor_groups, control_nodes.device)
    anchor_idx = anchor_idx[~torch.isin(anchor_idx, moving_idx)]
    handle_idx = torch.unique(torch.cat([moving_idx, anchor_idx], dim=0))
    handle_pos = control_nodes[handle_idx].detach().clone()

    move_mask = torch.isin(handle_idx, moving_idx)
    deltas_uv = np.repeat(
        np.array([[args.ear_drag_x, args.ear_drag_y]], dtype=np.float32),
        int(move_mask.sum().item()),
        axis=0,
    )
    deltas_world = camera_pixel_delta_to_world(
        control_nodes[handle_idx[move_mask]].detach().cpu().numpy(),
        metadata,
        image_size,
        deltas_uv,
    )
    if args.ear_drag_z != 0.0:
        extrinsic = np.asarray(metadata.get("extrinsic", np.eye(4)), dtype=np.float32)
        if extrinsic.size == 12:
            full = np.eye(4, dtype=np.float32)
            full[:3] = extrinsic.reshape(3, 4)
            extrinsic = full
        else:
            extrinsic = extrinsic.reshape(4, 4)
        camera_depth_delta = np.zeros_like(deltas_world, dtype=np.float32)
        camera_depth_delta[:, 2] = args.ear_drag_z
        deltas_world = deltas_world + camera_depth_delta @ extrinsic[:3, :3]
    handle_pos[move_mask] = handle_pos[move_mask] + torch.from_numpy(deltas_world).to(
        device=handle_pos.device,
        dtype=handle_pos.dtype,
    )

    debug = {
        "tail_pose": "ear_flick",
        "ear_region_count": int(ear_local.size),
        "ear_move_count": int(moving_local.size),
        "ear_region": {"x_min": float(x_min), "x_max": float(x_max), "y_max": float(y_max)},
        "ear_drag_uvz": [float(args.ear_drag_x), float(args.ear_drag_y), float(args.ear_drag_z)],
        "anchor_seeds": [int(x) for x in anchor_seeds],
        "tail_group": moving_idx.detach().cpu().numpy().astype(int).tolist(),
        "anchor_group": anchor_idx.detach().cpu().numpy().astype(int).tolist(),
    }
    return handle_idx, handle_pos, moving_idx, anchor_idx, debug


def normalize_rotation_bias(rotation_bias: torch.Tensor, row_count: int) -> torch.Tensor:
    if rotation_bias.ndim == 1 and rotation_bias.shape[0] == 4:
        rotation_bias = rotation_bias[None].repeat(row_count, 1)
    if rotation_bias.ndim != 2 or rotation_bias.shape[-1] != 4:
        raise ValueError(f"Invalid d_rotation_bias shape: {tuple(rotation_bias.shape)}")
    return torch.nn.functional.normalize(rotation_bias, dim=-1)


def mean_knn_distance(points: torch.Tensor, k: int) -> torch.Tensor:
    neighbor_count = min(max(int(k), 1) + 1, int(points.shape[0]))
    if neighbor_count <= 1:
        return torch.tensor(1e-6, dtype=points.dtype, device=points.device)
    dist = torch.cdist(points, points, p=2)
    nearest = torch.topk(dist, k=neighbor_count, dim=-1, largest=False, sorted=True).values[:, 1:]
    dist = torch.clamp(nearest, min=1e-6)
    return torch.clamp(dist.mean(), min=1e-6)


def build_union_mask(
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
    image = Image.fromarray((union.astype(np.uint8) * 255), mode="L")
    if dilation > 1:
        if dilation % 2 == 0:
            dilation += 1
        image = image.filter(ImageFilter.MaxFilter(dilation))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_handle_overlay(
    source_image: Path,
    output_path: Path,
    node_pixels: np.ndarray,
    tail_group: torch.Tensor,
    anchor_group: torch.Tensor,
    tail_uv: list[float],
) -> None:
    image = Image.open(source_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    for idx in anchor_group.detach().cpu().numpy().astype(int).tolist():
        x, y = node_pixels[idx]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(70, 170, 255), width=2)
    for idx in tail_group.detach().cpu().numpy().astype(int).tolist():
        x, y = node_pixels[idx]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 70, 70), width=2)
    x, y = tail_uv
    draw.line((x - 8, y, x + 8, y), fill=(255, 255, 0), width=2)
    draw.line((x, y - 8, x, y + 8), fill=(255, 255, 0), width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_deformation_mask(
    xyz: torch.Tensor,
    editable_mask: torch.Tensor,
    metadata: dict,
    image_size: tuple[int, int],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.deform_gate == "editable":
        return editable_mask

    if args.deform_gate == "tail-handles":
        raise ValueError("tail-handles deformation gate is built from NodeDriver nearest-control weights")

    if args.deform_gate != "tail-left":
        raise ValueError(f"Unsupported deform gate: {args.deform_gate}")

    bbox_size = np.maximum(bbox_max - bbox_min, 1.0)
    gate_x_min = args.deform_gate_x_min if args.deform_gate_x_min >= 0.0 else args.tail_tip_region_x_min
    gate_x_max = args.deform_gate_x_max if args.deform_gate_x_max >= 0.0 else args.tail_base_x
    gate_y_min = args.deform_gate_y_min if args.deform_gate_y_min >= 0.0 else args.tail_tip_region_y_min
    gate_y_max = args.deform_gate_y_max if args.deform_gate_y_max >= 0.0 else args.tail_tip_region_y_max
    x_min = bbox_min[0] + gate_x_min * bbox_size[0]
    x_max = bbox_min[0] + gate_x_max * bbox_size[0]
    y_min = bbox_min[1] + gate_y_min * bbox_size[1]
    y_max = bbox_min[1] + gate_y_max * bbox_size[1]

    pixels, valid = project_points_to_pixels(xyz.detach().cpu().numpy(), metadata, image_size)
    selected_np = (
        valid
        & (pixels[:, 0] >= x_min)
        & (pixels[:, 0] <= x_max)
        & (pixels[:, 1] >= y_min)
        & (pixels[:, 1] <= y_max)
    )
    selected = torch.from_numpy(selected_np).to(device=xyz.device, dtype=torch.bool)
    return editable_mask & selected


def build_tail_handle_deformation_mask(
    xyz: torch.Tensor,
    editable_mask: torch.Tensor,
    control_nodes: torch.Tensor,
    tail_group: torch.Tensor,
    radius: float,
    control_node_count: int,
) -> torch.Tensor:
    del control_node_count
    tail_nodes = control_nodes[tail_group.to(device=control_nodes.device, dtype=torch.long)]
    selected = torch.zeros_like(editable_mask, dtype=torch.bool)
    editable_idx = torch.where(editable_mask)[0]
    if editable_idx.numel() == 0 or tail_nodes.numel() == 0:
        return selected

    # Keep this chunked: scenes can have over a million Gaussians.
    chunk_size = 65536
    radius_sq = float(radius) ** 2
    for start in range(0, int(editable_idx.numel()), chunk_size):
        idx = editable_idx[start : start + chunk_size]
        dist_sq = torch.cdist(xyz[idx], tail_nodes, p=2).pow(2).min(dim=1).values
        selected[idx] = dist_sq <= radius_sq
    return selected


@torch.no_grad()
def run_objectmorpher_arap(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("ObjectMorpher ARAP deformation requires CUDA")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    GaussianModel = _load_gaussian_model()
    metadata = load_scene_metadata(args.scene_meta)
    gaussians = GaussianModel(0, with_motion_mask=False)
    gaussians.load_ply(args.gs_path)

    editable_mask = load_editable_mask_for_arap(args, gaussians, metadata)
    editable_count = int(editable_mask.sum().item())
    if editable_count < 4:
        raise ValueError(f"Need at least four editable Gaussians, got {editable_count}")

    opacity = gaussians.get_opacity[:, 0]
    control_mask = (opacity > args.control_min_opacity) & editable_mask
    if int(control_mask.sum().item()) < 4:
        control_mask = editable_mask
    control_candidates = gaussians.get_xyz[control_mask]
    control_count = min(args.node_count, control_candidates.shape[0])
    control_idx = farthest_point_sample(control_candidates[None], control_count)[0]
    control_nodes = control_candidates[control_idx].detach().clone()

    scale = torch.norm(control_nodes.max(0).values - control_nodes.min(0).values)
    mean_nn = mean_knn_distance(control_nodes, args.mean_nn_k)
    if args.node_radius_mode == "mean-nn":
        node_radius = torch.clamp(mean_nn * args.node_radius_scale, min=1e-6)
    else:
        node_radius = torch.clamp(scale / 20.0, min=1e-6)
    graph_radius = None
    if args.graph_radius_mode == "mean-nn":
        graph_radius = torch.clamp(mean_nn * args.graph_radius_scale, min=1e-6)
    animate_tool = LapDeform(
        init_pcl=control_nodes,
        K=4,
        trajectory=None,
        node_radius=node_radius,
        graph_radius=graph_radius,
        connectivity_mode=args.graph_mode,
        graph_k=args.graph_k,
        least_edge_num=args.least_edge_num,
    )
    animator = NodeDriver()

    mask_image, _, bbox_min, bbox_max = mask_bbox(Path(args.edit_mask))
    width, height = mask_image.size
    node_pixels, node_valid = project_points_to_pixels(control_nodes.detach().cpu().numpy(), metadata, (width, height))
    if args.selection_bbox == "projected":
        editable_points = gaussians.get_xyz[editable_mask].detach().cpu().numpy()
        editable_pixels, editable_valid = project_points_to_pixels(editable_points, metadata, (width, height))
        bbox_min, bbox_max = bbox_from_projected_points(editable_pixels, editable_valid, (width, height))
    if args.tail_pose == "tip-up":
        handle_idx, handle_pos, tail_group, anchor_group, selection_meta = select_tip_up_handles(
            animate_tool,
            control_nodes,
            node_pixels,
            node_valid,
            bbox_min,
            bbox_max,
            metadata,
            (width, height),
            args,
        )
    elif args.tail_pose == "tip-up-left":
        handle_idx, handle_pos, tail_group, anchor_group, selection_meta = select_tip_up_left_handles(
            animate_tool,
            control_nodes,
            node_pixels,
            node_valid,
            bbox_min,
            bbox_max,
            metadata,
            (width, height),
            args,
        )
    elif args.tail_pose == "tail-wave":
        handle_idx, handle_pos, tail_group, anchor_group, selection_meta = select_tail_wave_handles(
            animate_tool,
            control_nodes,
            node_pixels,
            node_valid,
            bbox_min,
            bbox_max,
            metadata,
            (width, height),
            args,
        )
    elif args.tail_pose == "ear-flick":
        handle_idx, handle_pos, tail_group, anchor_group, selection_meta = select_ear_flick_handles(
            animate_tool,
            control_nodes,
            node_pixels,
            node_valid,
            bbox_min,
            bbox_max,
            metadata,
            (width, height),
            args,
        )
    else:
        tail_group, anchor_group, selection_meta = select_handle_groups(
            animate_tool,
            control_nodes,
            node_pixels,
            node_valid,
            bbox_min,
            bbox_max,
            args,
        )

        handle_idx = torch.unique(torch.cat([tail_group, anchor_group], dim=0)).long()
        handle_pos = control_nodes[handle_idx].detach().clone()
        is_tail_handle = torch.isin(handle_idx, tail_group)
        drag_delta = camera_pixel_delta_to_world(
            control_nodes[handle_idx[is_tail_handle]].detach().cpu().numpy(),
            metadata,
            (width, height),
            np.array([args.drag_x, args.drag_y], dtype=np.float32),
        )
        handle_pos[is_tail_handle] = handle_pos[is_tail_handle] + torch.from_numpy(drag_delta).to(
            device=handle_pos.device,
            dtype=handle_pos.dtype,
        )

    animated_pcl, _, _ = animate_tool.deform_arap(
        handle_idx=handle_idx.detach().cpu().numpy().astype(np.int64).tolist(),
        handle_pos=handle_pos.detach().cpu().numpy().astype(np.float32),
        init_verts=None,
        return_R=True,
    )
    animation_trans_bias = animated_pcl - animate_tool.init_pcl
    d_values = animator(
        gaussians.get_xyz,
        control_nodes,
        animation_trans_bias,
        node_radius=float(node_radius.detach().cpu().item()),
    )
    d_xyz = d_values["d_xyz"]
    d_rotation_bias = normalize_rotation_bias(d_values["d_rotation_bias"], gaussians.get_xyz.shape[0])

    if args.deform_gate == "tail-handles":
        deformation_mask = build_tail_handle_deformation_mask(
            gaussians.get_xyz,
            editable_mask,
            control_nodes,
            tail_group,
            float(node_radius.detach().cpu().item()) * args.deform_gate_radius_scale,
            control_nodes.shape[0],
        )
    else:
        deformation_mask = build_deformation_mask(
            gaussians.get_xyz,
            editable_mask,
            metadata,
            (width, height),
            bbox_min,
            bbox_max,
            args,
        )
    gated = gate_deformation_values(
        editable_mask=deformation_mask,
        d_xyz=d_xyz,
        d_rotation=0.0,
        d_scaling=0.0,
        d_opacity=0.0,
        d_color=0.0,
        d_rotation_bias=d_rotation_bias,
    )
    d_xyz = gated["d_xyz"]
    d_rotation_bias = gated["d_rotation_bias"]
    edited_xyz = gaussians.get_xyz + d_xyz

    gaussians._xyz = torch.nn.Parameter(edited_xyz.detach().clone())
    gaussians._rotation = torch.nn.Parameter(quaternion_multiply(d_rotation_bias, gaussians.get_rotation).detach().clone())

    out_ply = Path(args.out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(out_ply))

    edit_state = Path(args.edit_state)
    edit_state.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        edit_state,
        editable_mask=editable_mask.detach().cpu().numpy().astype(np.bool_),
        deformation_mask=deformation_mask.detach().cpu().numpy().astype(np.bool_),
        initial_xyz=(edited_xyz - d_xyz).detach().cpu().numpy(),
        edited_xyz=edited_xyz.detach().cpu().numpy(),
        d_xyz=d_xyz.detach().cpu().numpy(),
        d_rotation_bias=d_rotation_bias.detach().cpu().numpy(),
        control_nodes=control_nodes.detach().cpu().numpy(),
        deformed_control_nodes=animated_pcl.detach().cpu().numpy(),
        handle_idx=handle_idx.detach().cpu().numpy(),
        tail_group=tail_group.detach().cpu().numpy(),
        anchor_group=anchor_group.detach().cpu().numpy(),
        drag_uv=np.array([args.drag_x, args.drag_y], dtype=np.float32),
    )

    if args.out_mask:
        build_union_mask(
            Path(args.edit_mask),
            edited_xyz.detach().cpu().numpy(),
            editable_mask.detach().cpu().numpy().astype(bool),
            metadata,
            Path(args.out_mask),
            args.mask_dilation,
        )

    debug = {
        **selection_meta,
        "gs_path": args.gs_path,
        "out_ply": str(out_ply),
        "edit_state": str(edit_state),
        "editable_count": editable_count,
        "deformation_count": int(deformation_mask.sum().item()),
        "gaussian_count": int(editable_mask.numel()),
        "control_nodes": int(control_nodes.shape[0]),
        "node_radius": float(node_radius.detach().cpu().item()),
        "mean_nn": float(mean_nn.detach().cpu().item()),
        "graph_radius": None if graph_radius is None else float(graph_radius.detach().cpu().item()),
        "graph_mode": args.graph_mode,
        "graph_k": int(args.graph_k),
        "least_edge_num": int(args.least_edge_num),
        "drag_uv": [float(args.drag_x), float(args.drag_y)],
        "seed": int(args.seed),
        "input_edit_state": args.input_edit_state,
        "handle_count": int(handle_idx.numel()),
        "tail_handle_count": int(tail_group.numel()),
        "anchor_handle_count": int(anchor_group.numel()),
        "max_displacement": float(torch.linalg.norm(d_xyz, dim=1).max().detach().cpu().item()),
        "mean_editable_displacement": float(torch.linalg.norm(d_xyz[editable_mask], dim=1).mean().detach().cpu().item()),
    }
    if args.debug_json:
        debug_path = Path(args.debug_json)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
    if args.handle_overlay:
        source = Path(args.source_image) if args.source_image else Path(metadata.get("source_image", ""))
        if not source.is_absolute():
            source = Path.cwd() / source
        overlay_uv = selection_meta.get("tail_uv", selection_meta.get("tip_target_uv", [0.0, 0.0]))
        save_handle_overlay(source, Path(args.handle_overlay), node_pixels, tail_group, anchor_group, overlay_uv)

    print(f"ARAP_PLY={out_ply}")
    print(f"EDIT_STATE={edit_state}")
    if args.out_mask:
        print(f"UNION_MASK={Path(args.out_mask)}")
    if args.handle_overlay:
        print(f"HANDLE_OVERLAY={Path(args.handle_overlay)}")
    print(
        "OBJECTMORPHER_ARAP "
        f"editable={editable_count}/{editable_mask.numel()} "
        f"nodes={control_nodes.shape[0]} handles={handle_idx.numel()} "
        f"tail={tail_group.numel()} anchors={anchor_group.numel()} "
        f"max_disp={debug['max_displacement']:.6f} mean_editable_disp={debug['mean_editable_displacement']:.6f}"
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ObjectMorpher local-global ARAP without opening the GUI")
    parser.add_argument("--gs-path", required=True)
    parser.add_argument("--source-image", default="")
    parser.add_argument("--edit-mask", required=True)
    parser.add_argument("--scene-meta", required=True)
    parser.add_argument("--out-ply", required=True)
    parser.add_argument("--edit-state", required=True)
    parser.add_argument("--input-edit-state", default="", help="Optional existing edit_state.npz used to select editable rows")
    parser.add_argument("--out-mask", default="")
    parser.add_argument("--debug-json", default="")
    parser.add_argument("--handle-overlay", default="")
    parser.add_argument("--node-count", type=int, default=512)
    parser.add_argument("--min-opacity", type=float, default=0.05)
    parser.add_argument("--control-min-opacity", type=float, default=0.9)
    parser.add_argument("--graph-mode", choices=("nn", "floyd"), default="nn")
    parser.add_argument("--graph-k", type=int, default=4, help="K used to build the Floyd geodesic graph between control nodes")
    parser.add_argument("--least-edge-num", type=int, default=3, help="Always keep at least this many graph edges even outside radius")
    parser.add_argument("--mean-nn-k", type=int, default=4, help="K used to estimate mean nearest-neighbor control-node spacing")
    parser.add_argument("--node-radius-mode", choices=("bbox", "mean-nn"), default="bbox")
    parser.add_argument("--node-radius-scale", type=float, default=2.0)
    parser.add_argument("--graph-radius-mode", choices=("bbox", "mean-nn"), default="bbox")
    parser.add_argument("--graph-radius-scale", type=float, default=4.0)
    parser.add_argument(
        "--deform-gate",
        choices=("editable", "tail-left", "tail-handles"),
        default="editable",
        help="Rows that receive the final skinned Gaussian displacement; tail modes freeze the body.",
    )
    parser.add_argument("--deform-gate-x-min", type=float, default=-1.0)
    parser.add_argument("--deform-gate-x-max", type=float, default=-1.0)
    parser.add_argument("--deform-gate-y-min", type=float, default=-1.0)
    parser.add_argument("--deform-gate-y-max", type=float, default=-1.0)
    parser.add_argument("--deform-gate-radius-scale", type=float, default=2.5)
    parser.add_argument(
        "--selection-bbox",
        choices=("mask", "projected"),
        default="mask",
        help="Use the original 2D mask bbox or the projected editable-Gaussian bbox for handle/gate coordinates.",
    )
    parser.add_argument("--tail-x", type=float, default=0.19)
    parser.add_argument("--tail-y", type=float, default=0.86)
    parser.add_argument("--tail-region-x", type=float, default=0.36)
    parser.add_argument("--tail-region-y", type=float, default=0.56)
    parser.add_argument("--tail-n-rings", type=int, default=1)
    parser.add_argument("--anchor-n-rings", type=int, default=1)
    parser.add_argument("--anchor-seeds", type=int, default=7)
    parser.add_argument("--drag-x", type=float, default=12.0)
    parser.add_argument("--drag-y", type=float, default=-85.0)
    parser.add_argument("--mask-dilation", type=int, default=35)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tail-pose", choices=("drag", "tip-up", "tip-up-left", "tail-wave", "ear-flick"), default="drag")
    parser.add_argument("--tail-tip-region-x-min", type=float, default=0.10)
    parser.add_argument("--tail-tip-region-x-max", type=float, default=0.34)
    parser.add_argument("--tail-tip-region-y-min", type=float, default=0.50)
    parser.add_argument("--tail-tip-region-y-max", type=float, default=1.00)
    parser.add_argument("--tail-base-x", type=float, default=0.34)
    parser.add_argument("--tail-base-y", type=float, default=0.56)
    parser.add_argument("--tail-move-min-alpha", type=float, default=0.22)
    parser.add_argument("--tip-target-x", type=float, default=0.24)
    parser.add_argument("--tip-target-y", type=float, default=0.12)
    parser.add_argument("--tip-alpha-gamma", type=float, default=0.85)
    parser.add_argument("--tip-curve-x", type=float, default=0.03)
    parser.add_argument("--tail-wave-x", type=float, default=0.18)
    parser.add_argument("--tail-wave-y", type=float, default=-0.04)
    parser.add_argument("--tail-wave-anchor-alpha", type=float, default=0.18)
    parser.add_argument("--ear-region-x-min", type=float, default=0.55)
    parser.add_argument("--ear-region-x-max", type=float, default=0.84)
    parser.add_argument("--ear-region-y-max", type=float, default=0.36)
    parser.add_argument("--ear-tip-quantile", type=float, default=0.45)
    parser.add_argument("--ear-drag-x", type=float, default=14.0)
    parser.add_argument("--ear-drag-y", type=float, default=-48.0)
    parser.add_argument("--ear-drag-z", type=float, default=0.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_objectmorpher_arap(parse_args())
