"""Programmatic ARAP from a saved ``deform_kpt.pickle``.

Loads the kpt pickle saved by edit_gui's ``save_deform_kpt`` button,
overrides the delta of the currently *selected* kpts with a user-specified
3-vector (defaults to +Y lift), then runs the same ARAP + NodeDriver
skinning path that ``edit_gui``'s save button uses, and writes ``edit_<N>.ply``
in the repo root (so it matches the GUI's output layout).

This bypasses the dpg viewport entirely, so it works in any environment
where the OM machinery imports — no display needed.
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# Install the pytorch3d shim before any OM import.
from .run_edit_gui import _install_pytorch3d_shim  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
EDITING_DIR = REPO_ROOT / "ObjectMorpher" / "editing"


def _setup_om_imports() -> None:
    _install_pytorch3d_shim()
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    if str(EDITING_DIR) not in sys.path:
        sys.path.insert(0, str(EDITING_DIR))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply ARAP using a saved deform_kpt.pickle (headless).")
    p.add_argument("--gs-path", default=str(REPO_ROOT / "runs" / "coarse.ply"))
    p.add_argument("--kpt-pickle", default=str(REPO_ROOT / "deform_kpt.pickle"))
    p.add_argument("--out-dir", default=str(REPO_ROOT))
    p.add_argument("--lift-x", type=float, default=0.0)
    p.add_argument("--lift-y", type=float, default=-0.3,
                   help="World-units lift for selected kpts. NOTE: TRELLIS coarse PLY uses -Y as up "
                        "(adapter render_frames uses up_world=(0,-1,0)), so negative values lift visually upward. "
                        "Magnitude ~0.3 ≈ a clear 60° tail raise; smaller values are subtle.")
    p.add_argument("--lift-z", type=float, default=0.0)
    p.add_argument("--node-count", type=int, default=512)
    p.add_argument("--mean-nn-k", type=int, default=4)
    p.add_argument("--node-radius-mode", choices=("bbox", "mean-nn"), default="bbox")
    p.add_argument("--node-radius-scale", type=float, default=2.0)
    p.add_argument("--graph-radius-mode", choices=("bbox", "mean-nn"), default="bbox")
    p.add_argument("--graph-radius-scale", type=float, default=4.0)
    p.add_argument("--graph-mode", default="nn")
    p.add_argument("--graph-k", type=int, default=12)
    p.add_argument("--least-edge-num", type=int, default=3)
    p.add_argument("--control-min-opacity", type=float, default=0.05)
    p.add_argument("--local-deform-radius-scale", type=float, default=3.0,
                   help="Gate gaussians: only those within node_radius * this scale of a moving (selected) node get displaced.")
    p.add_argument("--local-deform-node-rings", type=int, default=1,
                   help="Expand the moving-node set by this many ring neighbors before gating.")
    p.add_argument("--anchor-exclude-radius-scale", type=float, default=4.0,
                   help="Drop saved-anchor kpts whose 3D position lies within node_radius * this scale of any moving kpt. "
                        "Prevents nearby anchored handles from tearing the moving region.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    _setup_om_imports()

    # Imports that need the shim + sys.path
    from scene.gaussian_model import GaussianModel
    from edit_gui import sample_control_nodes, mean_knn_distance, NodeDriver
    from lap_deform import LapDeform

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load Gaussians
    gaussians = GaussianModel(3)
    gaussians.load_ply(args.gs_path)
    device = gaussians.get_xyz.device

    # 2) Build control-node graph (reproduce edit_gui.animation_initialize)
    opacity = gaussians.get_opacity[:, 0]
    mask = opacity > args.control_min_opacity
    pcl = gaussians.get_xyz[mask]
    cls = argparse.Namespace(
        node_count=args.node_count, mean_nn_k=args.mean_nn_k,
        node_radius_mode=args.node_radius_mode, node_radius_scale=args.node_radius_scale,
        graph_radius_mode=args.graph_radius_mode, graph_radius_scale=args.graph_radius_scale,
        graph_mode=args.graph_mode, graph_k=args.graph_k, least_edge_num=args.least_edge_num,
    )
    pcl = sample_control_nodes(pcl, cls)
    scale = torch.norm(pcl.max(0).values - pcl.min(0).values)
    mean_nn = mean_knn_distance(pcl, args.mean_nn_k)
    if args.node_radius_mode == "mean-nn":
        node_radius = torch.clamp(mean_nn * args.node_radius_scale, min=1e-6)
    else:
        node_radius = torch.clamp(scale / 20, min=1e-6)
    graph_radius = torch.clamp(mean_nn * args.graph_radius_scale, min=1e-6) if args.graph_radius_mode == "mean-nn" else None
    print(f"[graph] N={pcl.shape[0]}  node_radius={float(node_radius):.5f}  mean_nn={float(mean_nn):.5f}")

    animate_tool = LapDeform(
        init_pcl=pcl, K=4, trajectory=None,
        node_radius=node_radius, graph_radius=graph_radius,
        connectivity_mode=args.graph_mode, graph_k=args.graph_k,
        least_edge_num=args.least_edge_num,
    )

    # 3) Load saved kpts (need editing dir on sys.path for unpickling)
    with open(args.kpt_pickle, "rb") as f:
        dk = pickle.load(f)

    keypoints_idx = list(dk.keypoints_idx_list)            # node-graph indices, len=K
    keypoints_3d = [np.asarray(v, dtype=np.float32) for v in dk.keypoints3d_list]
    delta = [np.asarray(v, dtype=np.float32).copy() for v in dk.keypoints3d_delta_list]
    sel = list(dk.selective_keypoints_idx_list)            # positions in the kpt list (0..K-1)
    print(f"[kpt] total={len(keypoints_idx)}  selected={len(sel)}")
    if not sel:
        raise SystemExit("No selective_keypoints_idx_list — nothing to deform. Save a selection in GUI first.")

    # 4) Set lift on selected kpts
    lift = np.array([args.lift_x, args.lift_y, args.lift_z], dtype=np.float32)
    for i in sel:
        delta[i] = lift
    print(f"[lift] vec={lift}  applied to {len(sel)} selected kpts")

    # 5) Map saved 3D kpt positions → nearest node in OUR pcl
    #    (necessary because edit_gui's FPS uses a random start so saved node
    #    indices don't transfer across runs; saved 3D positions do.)
    src_pos = np.stack(keypoints_3d).astype(np.float32)  # (K, 3)
    target = src_pos + np.stack(delta).astype(np.float32)
    pcl_np = pcl.detach().cpu().numpy()
    # NN by squared distance
    dist_sq = ((src_pos[:, None, :] - pcl_np[None, :, :]) ** 2).sum(-1)
    mapped_node_idx = dist_sq.argmin(axis=1)
    mapping_err = np.sqrt(dist_sq[np.arange(len(src_pos)), mapped_node_idx])
    print(f"[map] saved-pos → our-node NN err: mean={mapping_err.mean():.5f} max={mapping_err.max():.5f}")

    # 5b) Drop saved-anchor kpts within R of any moving kpt — they tear the moving region.
    moving_pos = src_pos[sel]
    excl_r = float(node_radius) * args.anchor_exclude_radius_scale
    keep_mask = np.ones(len(src_pos), dtype=bool)
    if excl_r > 0:
        dist_to_mover = ((src_pos[:, None, :] - moving_pos[None, :, :]) ** 2).sum(-1)  # (K, |sel|)
        too_close = dist_to_mover.min(axis=1) < (excl_r * excl_r)
        # Keep selected (movers) always; drop too-close non-selected anchors
        sel_mask = np.zeros(len(src_pos), dtype=bool); sel_mask[sel] = True
        drop = too_close & (~sel_mask)
        keep_mask &= ~drop
        print(f"[anchors] R={excl_r:.4f}  dropped {int(drop.sum())} near-tail anchors (kept {int(keep_mask.sum())}/{len(src_pos)})")
    src_pos = src_pos[keep_mask]
    target = target[keep_mask]
    mapped_node_idx = mapped_node_idx[keep_mask]
    # Re-index selected positions within the filtered list (movers stay at their indices)
    orig_to_new = -np.ones(len(keep_mask), dtype=int); orig_to_new[keep_mask] = np.arange(int(keep_mask.sum()))
    sel = [int(orig_to_new[i]) for i in sel if keep_mask[i]]

    # Aggregate duplicates (same target node hit by multiple saved kpts → average target)
    from collections import defaultdict
    agg = defaultdict(list)
    for nidx, pos in zip(mapped_node_idx, target):
        agg[int(nidx)].append(pos)
    uniq_idx = sorted(agg.keys())
    uniq_pos = np.stack([np.mean(agg[k], axis=0) for k in uniq_idx]).astype(np.float32)
    print(f"[handles] unique nodes after NN-merge={len(uniq_idx)} (was {len(src_pos)})")

    handle_idx_t = torch.tensor(uniq_idx, dtype=torch.long, device=device)
    handle_pos_t = torch.from_numpy(uniq_pos).to(device)

    # 6) Run ARAP
    with torch.no_grad():
        animated_pcl, _quat, _scaling = animate_tool.deform_arap(
            handle_idx=handle_idx_t, handle_pos=handle_pos_t,
            init_verts=None, return_R=True,
        )
        trans_bias = animated_pcl - animate_tool.init_pcl
        moved = float(trans_bias.norm(dim=-1).mean())
        print(f"[arap] mean node trans = {moved:.5f}")

        # 7) Skinning → d_xyz, d_rotation_bias for every gaussian
        driver = NodeDriver()
        d_values = driver(gaussians.get_xyz, pcl, trans_bias, node_radius=float(node_radius))
        d_xyz = d_values["d_xyz"]
        d_rotation_bias = d_values.get("d_rotation_bias")

        # 7b) Local-deform gate: only apply displacement to gaussians within R of a
        #     *moving* (selected) handle node. Mirrors edit_gui's --local-deform-gate.
        sel_node_idx = sorted({int(mapped_node_idx[i]) for i in sel})
        moving_t = torch.tensor(sel_node_idx, device=device, dtype=torch.long)
        if args.local_deform_node_rings > 0:
            moving_t = animate_tool.add_n_ring_nbs(moving_t, n=args.local_deform_node_rings)
        moving_t = torch.unique(moving_t.long())
        radius = float(node_radius) * args.local_deform_radius_scale
        node_pos = pcl[moving_t]
        gs_xyz = gaussians.get_xyz
        dist_sq_min = torch.cdist(gs_xyz, node_pos, p=2).pow(2).min(dim=1).values
        local_mask = dist_sq_min <= (radius * radius)
        kept = int(local_mask.sum().item())
        print(f"[gate] moving nodes (incl. rings)={moving_t.numel()}  gated gaussians={kept}/{gs_xyz.shape[0]}  radius={radius:.4f}")
        d_xyz = torch.where(local_mask[:, None], d_xyz, torch.zeros_like(d_xyz))
        if torch.is_tensor(d_rotation_bias):
            identity_q = torch.zeros_like(d_rotation_bias); identity_q[:, 0] = 1.0
            d_rotation_bias = torch.where(local_mask[:, None], d_rotation_bias, identity_q)

        # 8) Apply to a copy and save
        gaussian_new = copy.deepcopy(gaussians)
        gaussian_new._xyz = torch.nn.Parameter((gaussian_new.get_xyz + d_xyz).detach().clone())
        if torch.is_tensor(d_rotation_bias):
            from edit_gui import quaternion_multiply
            d_rotation_bias = torch.nn.functional.normalize(d_rotation_bias, dim=-1)
            gaussian_new._rotation = torch.nn.Parameter(
                quaternion_multiply(d_rotation_bias, gaussian_new.get_rotation).detach().clone()
            )

        existing = sorted(p for p in os.listdir(out_dir) if p.startswith("edit") and p.endswith(".ply"))
        new_id = len(existing)
        save_path = out_dir / f"edit_{new_id}.ply"
        gaussian_new.save_ply(str(save_path))
        print(f"[save] {save_path}")


if __name__ == "__main__":
    main()
