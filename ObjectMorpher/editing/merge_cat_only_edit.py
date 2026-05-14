import argparse
from pathlib import Path

import numpy as np
import torch

from editing.diffusion_prior_finetune import ensure_runtime_imports
import editing.diffusion_prior_finetune as ft


GAUSSIAN_ROW_ATTRS = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_opacity",
    "_scaling",
    "_rotation",
)


def _load_gaussians(path: str):
    gaussians = ft.GaussianModel(0, with_motion_mask=False)
    gaussians.load_ply(path)
    return gaussians


def _replace_rows(full, cat, source_indices: torch.Tensor) -> None:
    for attr_name in GAUSSIAN_ROW_ATTRS:
        full_tensor = getattr(full, attr_name)
        cat_tensor = getattr(cat, attr_name)
        if full_tensor.shape[1:] != cat_tensor.shape[1:]:
            raise ValueError(
                f"{attr_name} shape mismatch: full {tuple(full_tensor.shape)} vs cat {tuple(cat_tensor.shape)}"
            )
        merged = full_tensor.detach().clone()
        merged[source_indices] = cat_tensor.detach()
        setattr(full, attr_name, torch.nn.Parameter(merged.requires_grad_(True)))

    if getattr(full, "fea_dim", 0) > 0:
        if full.feature.shape[1:] != cat.feature.shape[1:]:
            raise ValueError(f"feature shape mismatch: full {tuple(full.feature.shape)} vs cat {tuple(cat.feature.shape)}")
        merged_feature = full.feature.detach().clone()
        merged_feature[source_indices] = cat.feature.detach()
        full.feature = torch.nn.Parameter(merged_feature.requires_grad_(True))


@torch.no_grad()
def merge(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("merge_cat_only_edit requires CUDA because GaussianModel.load_ply uses CUDA tensors")
    ensure_runtime_imports()

    source_state = np.load(args.cat_source_state)
    if "source_indices" not in source_state:
        raise KeyError(f"{args.cat_source_state} does not contain source_indices")
    source_indices_np = source_state["source_indices"].astype(np.int64).reshape(-1)

    full = _load_gaussians(args.full_gs_path)
    cat = _load_gaussians(args.cat_gs_path)
    full_count = int(full.get_xyz.shape[0])
    cat_count = int(cat.get_xyz.shape[0])
    if source_indices_np.shape[0] != cat_count:
        raise ValueError(f"source_indices length {source_indices_np.shape[0]} does not match cat Gaussian count {cat_count}")
    if source_indices_np.min(initial=0) < 0 or source_indices_np.max(initial=-1) >= full_count:
        raise ValueError(f"source_indices are out of bounds for full Gaussian count {full_count}")

    source_indices = torch.from_numpy(source_indices_np).to(device=full.get_xyz.device, dtype=torch.long)
    initial_xyz_full = full.get_xyz.detach().cpu().numpy().astype(np.float32)
    _replace_rows(full, cat, source_indices)
    edited_xyz_full = full.get_xyz.detach().cpu().numpy().astype(np.float32)

    out_ply = Path(args.out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    full.save_ply(str(out_ply))

    editable_mask = np.zeros((full_count,), dtype=np.bool_)
    editable_mask[source_indices_np] = True
    d_xyz = np.zeros((full_count, 3), dtype=np.float32)
    d_xyz[source_indices_np] = edited_xyz_full[source_indices_np] - initial_xyz_full[source_indices_np]
    d_rotation_bias = np.zeros((full_count, 4), dtype=np.float32)
    d_rotation_bias[:, 0] = 1.0

    if args.cat_edit_state:
        cat_edit_state = np.load(args.cat_edit_state)
        if "d_xyz" in cat_edit_state:
            cat_d_xyz = cat_edit_state["d_xyz"].astype(np.float32)
            if cat_d_xyz.shape == (cat_count, 3):
                d_xyz[source_indices_np] = cat_d_xyz
        if "d_rotation_bias" in cat_edit_state:
            cat_d_rotation = cat_edit_state["d_rotation_bias"].astype(np.float32)
            if cat_d_rotation.shape == (cat_count, 4):
                d_rotation_bias[source_indices_np] = cat_d_rotation

    out_edit_state = Path(args.out_edit_state) if args.out_edit_state else out_ply.with_suffix(".edit_state.npz")
    out_edit_state.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_edit_state,
        editable_mask=editable_mask,
        initial_xyz=initial_xyz_full,
        edited_xyz=edited_xyz_full,
        d_xyz=d_xyz,
        d_rotation_bias=d_rotation_bias,
        source_indices=source_indices_np,
        full_gs_path=str(args.full_gs_path),
        cat_gs_path=str(args.cat_gs_path),
        cat_source_state=str(args.cat_source_state),
        cat_edit_state=str(args.cat_edit_state or ""),
    )

    bg_mask = ~editable_mask
    bg_max_delta = float(np.abs(d_xyz[bg_mask]).max(initial=0.0))
    obj_max_delta = float(np.linalg.norm(d_xyz[editable_mask], axis=1).max(initial=0.0))
    print(f"MERGED_PLY={out_ply}", flush=True)
    print(f"MERGED_EDIT_STATE={out_edit_state}", flush=True)
    print(f"FULL_COUNT={full_count} EDITABLE_COUNT={int(editable_mask.sum())}", flush=True)
    print(f"BACKGROUND_MAX_DXYZ={bg_max_delta:.8f} OBJECT_MAX_DXYZ={obj_max_delta:.8f}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a cat-only/object-only edited PLY back into its full-scene source rows")
    parser.add_argument("--full-gs-path", required=True)
    parser.add_argument("--cat-gs-path", required=True)
    parser.add_argument("--cat-source-state", required=True, help="Extraction edit_state containing source_indices")
    parser.add_argument("--cat-edit-state", default="", help="Optional object-only edit_state with d_xyz/d_rotation_bias")
    parser.add_argument("--out-ply", required=True)
    parser.add_argument("--out-edit-state", default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    merge(parse_args())
