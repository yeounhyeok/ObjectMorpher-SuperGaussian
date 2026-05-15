"""From-scratch 3DGS fit on HR pseudo-GT frames (SuperGS-style).

- Random-init N gaussians within a user-specified bbox
- Optimize means / scales / quats / opacities / colors (no SH, no densification)
- L1 + (1-SSIM) loss vs HR pseudo-GT frames
- Save as INRIA-format PLY with SH degree 0

Defaults match SuperGaussian paper: N=131072 fixed gaussians, 2000 iters.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import trange

SH_C0 = 0.28209479177387814


def load_transforms(transforms_path: Path, image_dir: Path, train_size: int):
    """Load NerfStudio-style transforms.json + matching HR images, resize to train_size."""
    meta = json.loads(transforms_path.read_text())
    src_w = int(meta.get("w", 256)); src_h = int(meta.get("h", 256))
    fl_x = float(meta["fl_x"]); fl_y = float(meta["fl_y"])
    cx = float(meta["cx"]); cy = float(meta["cy"])

    # Find HR image dims by reading one frame
    sample = sorted(image_dir.glob("*.png"))[0]
    hr_w, hr_h = Image.open(sample).size
    sx = hr_w / src_w; sy = hr_h / src_h
    # Now scale to train_size
    rx = train_size / hr_w; ry = train_size / hr_h
    fl_x_t = fl_x * sx * rx; fl_y_t = fl_y * sy * ry
    cx_t = cx * sx * rx;   cy_t = cy * sy * ry

    imgs, c2ws = [], []
    for fr in meta["frames"]:
        fpath = image_dir / Path(fr["file_path"]).name
        img = Image.open(fpath).convert("RGB").resize((train_size, train_size), Image.LANCZOS)
        imgs.append(np.asarray(img, dtype=np.float32) / 255.0)
        c2ws.append(np.asarray(fr["transform_matrix"], dtype=np.float32))
    imgs = np.stack(imgs)                 # (N, H, W, 3)
    c2ws = np.stack(c2ws)                 # (N, 4, 4)
    K = np.array([[fl_x_t, 0, cx_t], [0, fl_y_t, cy_t], [0, 0, 1]], dtype=np.float32)
    return imgs, c2ws, K, train_size


def init_gaussians(num: int, bbox: np.ndarray, device: torch.device, init_scale: float):
    """Random init within axis-aligned bbox = (min, max) of shape (2, 3)."""
    lo = torch.tensor(bbox[0], device=device, dtype=torch.float32)
    hi = torch.tensor(bbox[1], device=device, dtype=torch.float32)
    means = (lo + (hi - lo) * torch.rand(num, 3, device=device)).requires_grad_(True)
    scales_param = torch.full((num, 3), math.log(init_scale), device=device, requires_grad=True)
    rots = torch.zeros(num, 4, device=device); rots[:, 0] = 1.0
    rots.requires_grad_(True)
    opac_param = torch.full((num, 1), float(np.log(0.1 / 0.9)), device=device, requires_grad=True)  # sigmoid -> 0.1
    colors = torch.full((num, 3), 0.5, device=device, requires_grad=True)
    return means, scales_param, rots, opac_param, colors


def init_gaussians_from_ply(ply_path: Path, device: torch.device):
    """Load INRIA-format PLY and unpack into trainable params (SH dropped to DC only).

    Returns the same 5-tuple as ``init_gaussians`` but warm-started from a real model.
    """
    v = PlyData.read(str(ply_path))["vertex"]
    n = len(v["x"])
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)                          # (N, 3)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)          # (N, 3)
    color = f_dc * SH_C0 + 0.5                                            # convert SH DC -> linear RGB-ish 0..1
    color = np.clip(color, 0.0, 1.0)
    opa = np.asarray(v["opacity"]).reshape(-1, 1)                         # logit, keep as is
    scl = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)         # log-scale, keep as is
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1)   # wxyz

    means = torch.from_numpy(xyz.astype(np.float32)).to(device).requires_grad_(True)
    scales_param = torch.from_numpy(scl.astype(np.float32)).to(device).requires_grad_(True)
    rots = torch.from_numpy(rot.astype(np.float32)).to(device).requires_grad_(True)
    opac_param = torch.from_numpy(opa.astype(np.float32)).to(device).requires_grad_(True)
    colors = torch.from_numpy(color.astype(np.float32)).to(device).requires_grad_(True)
    print(f"[init] from {ply_path.name}: N={n}  (SH degrees>0 dropped)")
    return means, scales_param, rots, opac_param, colors


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - SSIM (3-ch, simple gaussian-window). pred/target: (1, 3, H, W) in [0,1]."""
    kernel_size = 11; sigma = 1.5
    coords = torch.arange(kernel_size, device=pred.device, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    window = (g[:, None] * g[None, :])[None, None].expand(3, 1, kernel_size, kernel_size)
    def cmean(x): return F.conv2d(x, window, padding=kernel_size // 2, groups=3)
    mu_x = cmean(pred); mu_y = cmean(target)
    mu_xy = mu_x * mu_y; mu_xx = mu_x * mu_x; mu_yy = mu_y * mu_y
    sigma_x = cmean(pred * pred) - mu_xx
    sigma_y = cmean(target * target) - mu_yy
    sigma_xy = cmean(pred * target) - mu_xy
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_xx + mu_yy + C1) * (sigma_x + sigma_y + C2))
    return 1.0 - ssim.mean()


def save_inria_ply(path: Path, means, scales_param, rots, opac_param, colors):
    """Save with SH degree 0 (only f_dc). Inverse activations: opacity = logit, scale = log.

    f_dc encoded so that SH_C0 * f_dc + 0.5 = color  =>  f_dc = (color - 0.5) / SH_C0.
    """
    n = means.shape[0]
    xyz = means.detach().cpu().numpy()
    nrm = np.zeros_like(xyz)
    f_dc = ((colors.detach() - 0.5) / SH_C0).cpu().numpy()   # (N, 3)
    opa = opac_param.detach().cpu().numpy()                 # (N, 1)
    scl = scales_param.detach().cpu().numpy()               # (N, 3)  log-scale
    rot = F.normalize(rots.detach(), dim=-1).cpu().numpy()  # (N, 4)  wxyz

    attrs = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
             "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.empty(n, dtype=[(a, "f4") for a in attrs])
    data = np.concatenate([xyz, nrm, f_dc, opa, scl, rot], axis=1)
    arr[:] = list(map(tuple, data))
    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(str(path))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="From-scratch 3DGS fit (SuperGS-style).")
    p.add_argument("--frames-dir", required=True, type=Path, help="HR PNG directory")
    p.add_argument("--transforms", required=True, type=Path, help="transforms.json (LR cameras; intrinsics scale to HR)")
    p.add_argument("--bbox-ply", type=Path, default=None,
                   help="PLY to derive bbox from (e.g. edit_0.ply). If unset uses [-0.5,0.5]^3.")
    p.add_argument("--init-ply", type=Path, default=None,
                   help="If set, warm-start gaussians from this PLY instead of random init. "
                        "SG variant: keeps the input model's structure and only refines toward SR HR views.")
    p.add_argument("--out-ply", required=True, type=Path)
    p.add_argument("--num-gaussians", type=int, default=131072)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--init-scale", type=float, default=0.02, help="Initial gaussian scale (world units)")
    p.add_argument("--lr-means", type=float, default=1.6e-4)
    p.add_argument("--lr-scales", type=float, default=5e-3)
    p.add_argument("--lr-rots", type=float, default=1e-3)
    p.add_argument("--lr-opacities", type=float, default=5e-2)
    p.add_argument("--lr-colors", type=float, default=2.5e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    # Load data
    imgs, c2ws, K_np, S = load_transforms(args.transforms, args.frames_dir, args.train_size)
    N_views = imgs.shape[0]
    imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)              # (N, 3, S, S)
    viewmats_all = torch.from_numpy(np.linalg.inv(c2ws)).to(device, torch.float32)  # (N, 4, 4)
    Ks_one = torch.from_numpy(K_np).to(device).unsqueeze(0)                     # (1, 3, 3)
    print(f"[data] {N_views} HR views @ {S}x{S}  fl=({K_np[0,0]:.1f},{K_np[1,1]:.1f})")

    # bbox from init ply (or default)
    if args.bbox_ply is not None and args.bbox_ply.exists():
        v = PlyData.read(str(args.bbox_ply))["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], 1)
        bbox = np.stack([xyz.min(0) - 0.05, xyz.max(0) + 0.05])
        print(f"[bbox] from {args.bbox_ply.name}: min={bbox[0]} max={bbox[1]}")
    else:
        bbox = np.array([[-0.5,-0.5,-0.5], [0.5,0.5,0.5]], np.float32)
        print(f"[bbox] default: {bbox}")

    if args.init_ply is not None:
        means, scales_param, rots, opac_param, colors = init_gaussians_from_ply(args.init_ply, device)
    else:
        means, scales_param, rots, opac_param, colors = init_gaussians(args.num_gaussians, bbox, device, args.init_scale)
    opt = torch.optim.Adam([
        {"params": [means],         "lr": args.lr_means},
        {"params": [scales_param],  "lr": args.lr_scales},
        {"params": [rots],          "lr": args.lr_rots},
        {"params": [opac_param],    "lr": args.lr_opacities},
        {"params": [colors],        "lr": args.lr_colors},
    ])

    from gsplat import rasterization
    # gsplat requires backgrounds with shape (C,) or None; we use solid black per-image.
    bg = None
    pbar = trange(args.iters, desc="fit")
    losses = []
    for it in pbar:
        idx = int(torch.randint(0, N_views, (1,)).item())
        viewmat = viewmats_all[idx:idx+1]                       # (1, 4, 4)
        target = imgs_t[idx:idx+1]                              # (1, 3, S, S)
        scales = scales_param.exp().clamp(max=0.5)
        opac = opac_param.sigmoid().squeeze(-1)
        out, _, _ = rasterization(
            means=means, quats=rots, scales=scales, opacities=opac,
            colors=colors.clamp(0.0, 1.0),
            viewmats=viewmat, Ks=Ks_one, width=S, height=S,
        )
        pred = out[0].permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
        l1 = F.l1_loss(pred, target)
        if args.ssim_weight > 0:
            sl = ssim_loss(pred, target)
            loss = (1 - args.ssim_weight) * l1 + args.ssim_weight * sl
        else:
            loss = l1
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(loss.item())
        if it % 50 == 0 or it == args.iters - 1:
            pbar.set_postfix(loss=f"{loss.item():.4f}", l1=f"{l1.item():.4f}")

    print(f"[done] final loss = {np.mean(losses[-100:]):.4f}")
    args.out_ply.parent.mkdir(parents=True, exist_ok=True)
    save_inria_ply(args.out_ply, means, scales_param, rots, opac_param, colors)
    print(f"[save] {args.out_ply}  ({args.num_gaussians} gaussians, sh_degree=0)")


if __name__ == "__main__":
    main()
