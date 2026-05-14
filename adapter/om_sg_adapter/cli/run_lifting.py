from __future__ import annotations

import argparse
from pathlib import Path

from ..om_bridge.lift_2d import lift_image_to_ply


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TRELLIS 2D->3DGS lifting on an object crop.")
    parser.add_argument("--input-crop", required=True, type=Path)
    parser.add_argument("--output-ply", required=True, type=Path)
    parser.add_argument("--output-glb", default=None, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--model-id", default="microsoft/TRELLIS-image-large")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lift_image_to_ply(
        repo_root=Path(args.repo_root),
        image_path=args.input_crop,
        output_ply=args.output_ply,
        seed=args.seed,
        output_glb=args.output_glb,
        model_id=args.model_id,
    )
    print(f"Saved PLY -> {args.output_ply}")
    if args.output_glb:
        print(f"Saved GLB -> {args.output_glb}")


if __name__ == "__main__":
    main()
