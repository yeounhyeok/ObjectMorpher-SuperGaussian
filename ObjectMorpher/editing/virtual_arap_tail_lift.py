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
    spec = importlib.util.spec_from_file_location("objectmorpher_virtual_arap_gaussian_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load GaussianModel from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GaussianModel


def farthest_point_sample(points: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=points.device)
    generator.manual_seed(seed)
    count = min(count, points.shape[0])
    selected = torch.empty((count,), dtype=torch.long, device=points.device)
    distance = torch.full((points.shape[0],), 1e10, dtype=points.dtype, device=points.device)
    farthest = torch.randint(points.shape[0], (1,), generator=generator, device=points.device).item()
    for i in range(count):
        selected[i] = farthest
        dist = torch.sum((points - points[farthest]) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.argmax(distance).item()
    return selected


def build_knn_graph(nodes: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    dist2 = torch.cdist(nodes, nodes).square()
    knn_dist2, knn_idx = torch.topk(dist2, k=k + 1, largest=False, dim=1)
    knn_dist2 = knn_dist2[:, 1:]
    knn_idx = knn_idx[:, 1:]
    finite = knn_dist2.reshape(-1)
    scale = torch.clamp(finite[finite > 0].mean(), min=1e-8)
    weights = torch.exp(-knn_dist2 / scale)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return knn_idx, weights


def build_laplacian(knn_idx: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    n, k = knn_idx.shape
    laplacian = torch.eye(n, dtype=weights.dtype, device=weights.device)
    rows = torch.arange(n, device=weights.device)[:, None].expand(n, k)
    laplacian[rows.reshape(-1), knn_idx.reshape(-1)] = -weights.reshape(-1)
    return laplacian


def solve_with_handles(laplacian: torch.Tensor, rhs: torch.Tensor, handle_idx: torch.Tensor, handle_pos: torch.Tensor) -> torch.Tensor:
    n = laplacian.shape[0]
    handle_mask = torch.zeros((n,), dtype=torch.bool, device=laplacian.device)
    handle_mask[handle_idx] = True
    unknown_mask = ~handle_mask
    adjusted_rhs = rhs - laplacian[:, handle_idx] @ handle_pos
    solution_unknown = torch.linalg.lstsq(laplacian[:, unknown_mask], adjusted_rhs).solution
    solution = torch.empty_like(rhs)
    solution[handle_idx] = handle_pos
    solution[unknown_mask] = solution_unknown
    return solution


def estimate_rotations(nodes: torch.Tensor, deformed: torch.Tensor, knn_idx: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    n, k = knn_idx.shape
    source_edges = nodes[:, None, :] - nodes[knn_idx]
    target_edges = deformed[:, None, :] - deformed[knn_idx]
    covariance = torch.einsum("nka,nk,nkb->nab", source_edges, weights, target_edges)
    u, s, vh = torch.linalg.svd(covariance)
    rotations = vh.transpose(-2, -1) @ u.transpose(-2, -1)
    det = torch.det(rotations)
    bad = det < 0
    if bad.any():
        vh_fixed = vh.clone()
        min_col = torch.argmin(s[bad], dim=1)
        bad_idx = torch.where(bad)[0]
        vh_fixed[bad_idx, min_col, :] *= -1
        rotations[bad] = vh_fixed[bad].transpose(-2, -1) @ u[bad].transpose(-2, -1)
    return rotations


def deform_nodes_arap(
    nodes: torch.Tensor,
    handle_idx: torch.Tensor,
    handle_pos: torch.Tensor,
    *,
    graph_k: int,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    knn_idx, weights = build_knn_graph(nodes, graph_k)
    laplacian = build_laplacian(knn_idx, weights)
    deformed = solve_with_handles(laplacian, laplacian @ nodes, handle_idx, handle_pos)
    rotations = torch.eye(3, dtype=nodes.dtype, device=nodes.device)[None].repeat(nodes.shape[0], 1, 1)

    for _ in range(iterations):
        rotations = estimate_rotations(nodes, deformed, knn_idx, weights)
        edge = nodes[:, None, :] - nodes[knn_idx]
        neighbor_rot = rotations[knn_idx]
        blended = 0.5 * (
            torch.einsum("nab,nkb->nka", rotations, edge)
            + torch.einsum("nkab,nkb->nka", neighbor_rot, edge)
        )
        rhs = (weights[..., None] * blended).sum(dim=1)
        deformed = solve_with_handles(laplacian, rhs, handle_idx, handle_pos)

    return deformed, knn_idx, weights


def camera_pixel_delta_to_world(points: np.ndarray, metadata: dict, image_size: tuple[int, int], pixel_delta: np.ndarray) -> np.ndarray:
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
    camera_delta = np.zeros_like(points, dtype=np.float32)
    camera_delta[:, 0] = pixel_delta[0] * z / max(float(intrinsic[0, 0]), 1e-6)
    camera_delta[:, 1] = pixel_delta[1] * z / max(float(intrinsic[1, 1]), 1e-6)
    return camera_delta @ extrinsic[:3, :3]


def skin_gaussians(
    xyz: torch.Tensor,
    editable_mask: torch.Tensor,
    nodes: torch.Tensor,
    node_delta: torch.Tensor,
    *,
    k: int,
    chunk_size: int,
) -> torch.Tensor:
    output = torch.zeros_like(xyz)
    scale = torch.clamp(torch.cdist(nodes, nodes).square().topk(k=2, largest=False, dim=1).values[:, 1].mean(), min=1e-8)
    for start in range(0, xyz.shape[0], chunk_size):
        end = min(start + chunk_size, xyz.shape[0])
        chunk = xyz[start:end]
        dist2 = torch.cdist(chunk, nodes).square()
        nn_dist2, nn_idx = torch.topk(dist2, k=k, largest=False, dim=1)
        weights = torch.exp(-nn_dist2 / scale)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        output[start:end] = (node_delta[nn_idx] * weights[..., None]).sum(dim=1)
    output = torch.where(editable_mask[:, None], output, torch.zeros_like(output))
    return output


def build_union_mask(original_mask_path: Path, edited_xyz: np.ndarray, editable_mask: np.ndarray, metadata: dict, output_path: Path, dilation: int) -> None:
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


def apply_tail_lift(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("virtual_arap_tail_lift requires CUDA")
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
    editable_count = int(editable_mask.sum().item())
    if editable_count < args.node_count:
        raise ValueError(f"Only {editable_count} editable Gaussians found")

    with Image.open(args.edit_mask).convert("L") as mask_image:
        width, height = mask_image.size
        mask_np = np.asarray(mask_image) > 127
    ys, xs = np.where(mask_np)
    bbox_min = np.array([xs.min(), ys.min()], dtype=np.float32)
    bbox_max = np.array([xs.max(), ys.max()], dtype=np.float32)
    bbox_size = bbox_max - bbox_min
    tail_uv = bbox_min + np.array([args.tail_x, args.tail_y], dtype=np.float32) * bbox_size

    xyz = gaussians.get_xyz.detach()
    candidates = torch.where(editable_mask)[0]
    opacity = gaussians.get_opacity[candidates, 0]
    opaque_candidates = candidates[opacity > args.min_opacity]
    if opaque_candidates.shape[0] >= args.node_count:
        candidates = opaque_candidates
    sample_idx = farthest_point_sample(xyz[candidates], args.node_count, args.seed)
    node_world_idx = candidates[sample_idx]
    nodes = xyz[node_world_idx].detach().clone()

    node_pixels, node_valid = project_points_to_pixels(nodes.detach().cpu().numpy(), metadata, (width, height))
    tail_score = np.sum((node_pixels - tail_uv[None, :]) ** 2, axis=1)
    tail_region = (
        node_valid
        & (node_pixels[:, 0] < bbox_min[0] + args.tail_region_x * bbox_size[0])
        & (node_pixels[:, 1] > bbox_min[1] + args.tail_region_y * bbox_size[1])
    )
    if tail_region.sum() >= args.tail_handles:
        tail_pool = np.where(tail_region)[0]
        tail_handle_local = tail_pool[np.argsort(tail_score[tail_pool])[: args.tail_handles]]
    else:
        tail_handle_local = np.argsort(tail_score)[: args.tail_handles]

    anchor_region = (
        node_valid
        & (
            (node_pixels[:, 0] > bbox_min[0] + args.anchor_region_x * bbox_size[0])
            | (node_pixels[:, 1] < bbox_min[1] + args.anchor_region_y * bbox_size[1])
        )
    )
    anchor_pool = np.setdiff1d(np.where(anchor_region)[0], tail_handle_local)
    if anchor_pool.shape[0] < args.anchor_count:
        anchor_pool = np.setdiff1d(np.arange(nodes.shape[0]), tail_handle_local)
    anchor_score = np.sum((node_pixels[anchor_pool] - node_pixels[anchor_pool].mean(axis=0, keepdims=True)) ** 2, axis=1)
    anchor_local = anchor_pool[np.argsort(anchor_score)[-args.anchor_count:]]

    tail_delta = camera_pixel_delta_to_world(
        nodes[tail_handle_local].detach().cpu().numpy(),
        metadata,
        (width, height),
        np.array([args.drag_x, args.drag_y], dtype=np.float32),
    )
    tail_target = nodes[tail_handle_local] + torch.from_numpy(tail_delta).to(device=nodes.device, dtype=nodes.dtype)
    anchor_target = nodes[anchor_local]

    handle_local = torch.from_numpy(np.concatenate([tail_handle_local, anchor_local])).long().cuda()
    handle_target = torch.cat([tail_target, anchor_target], dim=0)
    unique_handles, inverse = torch.unique(handle_local, return_inverse=True)
    unique_target = torch.zeros((unique_handles.shape[0], 3), dtype=handle_target.dtype, device=handle_target.device)
    unique_target.index_add_(0, inverse, handle_target)
    counts = torch.zeros((unique_handles.shape[0], 1), dtype=handle_target.dtype, device=handle_target.device)
    counts.index_add_(0, inverse, torch.ones((handle_target.shape[0], 1), dtype=handle_target.dtype, device=handle_target.device))
    unique_target = unique_target / counts.clamp_min(1)

    deformed_nodes, _, _ = deform_nodes_arap(
        nodes,
        unique_handles,
        unique_target,
        graph_k=args.graph_k,
        iterations=args.arap_iterations,
    )
    node_delta = deformed_nodes - nodes
    d_xyz = skin_gaussians(
        xyz,
        editable_mask,
        nodes,
        node_delta,
        k=args.skin_k,
        chunk_size=args.chunk_size,
    )
    edited_xyz = xyz + d_xyz

    gaussians._xyz = torch.nn.Parameter(edited_xyz.detach().clone())
    out_ply = Path(args.out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(out_ply))

    edit_state = Path(args.edit_state)
    edit_state.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        edit_state,
        editable_mask=editable_mask.detach().cpu().numpy().astype(np.bool_),
        initial_xyz=xyz.detach().cpu().numpy(),
        edited_xyz=edited_xyz.detach().cpu().numpy(),
        d_xyz=d_xyz.detach().cpu().numpy(),
        node_world_idx=node_world_idx.detach().cpu().numpy(),
        nodes=nodes.detach().cpu().numpy(),
        deformed_nodes=deformed_nodes.detach().cpu().numpy(),
        tail_handle_local=tail_handle_local.astype(np.int64),
        anchor_local=anchor_local.astype(np.int64),
        tail_uv=tail_uv,
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

    moved = torch.linalg.norm(d_xyz, dim=1)
    print(f"ARAP_TAIL_PLY={out_ply}")
    print(f"EDIT_STATE={edit_state}")
    if args.out_mask:
        print(f"UNION_MASK={Path(args.out_mask)}")
    print(f"EDITABLE={editable_count}/{editable_mask.numel()} NODES={nodes.shape[0]}")
    print(f"TAIL_UV={tail_uv.tolist()} DRAG_UV={[args.drag_x, args.drag_y]}")
    print(f"TAIL_HANDLES={tail_handle_local.tolist()} ANCHORS={len(anchor_local)}")
    print(f"MAX_DISPLACEMENT={float(moved.max()):.6f} MEAN_EDITABLE_DISPLACEMENT={float(moved[editable_mask].mean()):.6f}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Programmatic ARAP tail lift for a masked full-scene 3DGS")
    parser.add_argument("--gs-path", required=True)
    parser.add_argument("--edit-mask", required=True)
    parser.add_argument("--scene-meta", required=True)
    parser.add_argument("--out-ply", required=True)
    parser.add_argument("--edit-state", required=True)
    parser.add_argument("--out-mask", default="")
    parser.add_argument("--node-count", type=int, default=512)
    parser.add_argument("--graph-k", type=int, default=12)
    parser.add_argument("--skin-k", type=int, default=4)
    parser.add_argument("--arap-iterations", type=int, default=6)
    parser.add_argument("--tail-handles", type=int, default=10)
    parser.add_argument("--anchor-count", type=int, default=110)
    parser.add_argument("--tail-x", type=float, default=0.18)
    parser.add_argument("--tail-y", type=float, default=0.84)
    parser.add_argument("--tail-region-x", type=float, default=0.38)
    parser.add_argument("--tail-region-y", type=float, default=0.55)
    parser.add_argument("--anchor-region-x", type=float, default=0.38)
    parser.add_argument("--anchor-region-y", type=float, default=0.45)
    parser.add_argument("--drag-x", type=float, default=18.0)
    parser.add_argument("--drag-y", type=float, default=-85.0)
    parser.add_argument("--min-opacity", type=float, default=0.05)
    parser.add_argument("--mask-dilation", type=int, default=35)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


if __name__ == "__main__":
    apply_tail_lift(parse_args())
