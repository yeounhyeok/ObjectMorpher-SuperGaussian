import argparse
from pathlib import Path

import numpy as np
import torch

from editing.diffusion_prior_finetune import ensure_runtime_imports
import editing.diffusion_prior_finetune as ft


@torch.no_grad()
def extract(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("extract_editable_gaussians requires CUDA because GaussianModel.load_ply uses CUDA tensors")
    ensure_runtime_imports()

    state = np.load(args.edit_state)
    if args.mask_key not in state:
        raise KeyError(f"{args.edit_state} does not contain {args.mask_key}")
    keep_np = state[args.mask_key].astype(bool).reshape(-1)

    gaussians = ft.GaussianModel(0, with_motion_mask=False)
    gaussians.load_ply(args.gs_path)
    if keep_np.shape[0] != gaussians.get_xyz.shape[0]:
        raise ValueError(f"Mask length {keep_np.shape[0]} does not match Gaussian count {gaussians.get_xyz.shape[0]}")

    keep = torch.from_numpy(keep_np).to(device=gaussians.get_xyz.device, dtype=torch.bool)
    if int(keep.sum().item()) < 1:
        raise ValueError("Editable mask selected zero Gaussians")

    gaussians._xyz = torch.nn.Parameter(gaussians._xyz[keep].detach().clone())
    gaussians._features_dc = torch.nn.Parameter(gaussians._features_dc[keep].detach().clone())
    gaussians._features_rest = torch.nn.Parameter(gaussians._features_rest[keep].detach().clone())
    gaussians._opacity = torch.nn.Parameter(gaussians._opacity[keep].detach().clone())
    gaussians._scaling = torch.nn.Parameter(gaussians._scaling[keep].detach().clone())
    gaussians._rotation = torch.nn.Parameter(gaussians._rotation[keep].detach().clone())
    if getattr(gaussians, "fea_dim", 0) > 0:
        gaussians.feature = torch.nn.Parameter(gaussians.feature[keep].detach().clone())

    out_ply = Path(args.out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(out_ply))

    out_edit_state = Path(args.out_edit_state) if args.out_edit_state else out_ply.with_suffix(".edit_state.npz")
    out_edit_state.parent.mkdir(parents=True, exist_ok=True)
    source_indices = np.flatnonzero(keep_np).astype(np.int64)
    np.savez_compressed(
        out_edit_state,
        editable_mask=np.ones((source_indices.shape[0],), dtype=np.bool_),
        source_indices=source_indices,
        source_gs_path=str(args.gs_path),
        source_edit_state=str(args.edit_state),
    )

    print(f"EXTRACTED_PLY={out_ply}", flush=True)
    print(f"EXTRACTED_EDIT_STATE={out_edit_state}", flush=True)
    print(f"EXTRACTED_COUNT={source_indices.shape[0]}/{keep_np.shape[0]}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract editable/object Gaussian rows into a standalone PLY")
    parser.add_argument("--gs-path", required=True)
    parser.add_argument("--edit-state", required=True)
    parser.add_argument("--out-ply", required=True)
    parser.add_argument("--out-edit-state", default="")
    parser.add_argument("--mask-key", default="editable_mask")
    return parser.parse_args(argv)


if __name__ == "__main__":
    extract(parse_args())
