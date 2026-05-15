"""Frame-wise Real-ESRGAN x4 SR on a directory of orbit renders.

Inputs: directory of LR PNGs (e.g. om_baseline_frames at 256x256).
Output: directory of HR PNGs at 4x.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Real-ESRGAN x4 SR on a frame directory.")
    p.add_argument("--in-dir", required=True, type=Path, help="Directory of LR PNGs")
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory for HR PNGs")
    p.add_argument("--ckpt", default="ckpts/RealESRGAN_x4plus.pth",
                   type=Path, help="Real-ESRGAN x4plus weights")
    p.add_argument("--tile", type=int, default=0, help="Tile size (0 = no tiling)")
    p.add_argument("--device", default="cuda")
    return p


def main() -> None:
    args = build_parser().parse_args()
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4, model_path=str(args.ckpt), model=model,
        tile=args.tile, tile_pad=10, pre_pad=0, half=True, device=args.device,
    )

    frames = sorted(args.in_dir.glob("*.png"))
    if not frames:
        raise SystemExit(f"no PNG frames found in {args.in_dir}")
    print(f"[sr] {len(frames)} frames  {args.in_dir} -> {args.out_dir}")

    for f in frames:
        img = np.asarray(Image.open(f).convert("RGB"))
        with torch.no_grad():
            out, _ = upsampler.enhance(img[..., ::-1], outscale=4)  # BGR in/out
        Image.fromarray(out[..., ::-1]).save(args.out_dir / f.name)
    print(f"[sr] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
