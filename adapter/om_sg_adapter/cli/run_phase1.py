from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import AdapterConfig, SuperGaussianConfig, TrajectoryConfig
from ..pipelines.phase1_render_then_upsample import run_phase1_render


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 1 adapter scaffold steps.")
    parser.add_argument("--run-name", default="sample-phase1")
    parser.add_argument("--objectmorpher-ply", default="ObjectMorpher/sample.ply")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--fovy-degrees", type=float, default=60.0)
    parser.add_argument("--trajectory-frames", type=int, default=24)
    parser.add_argument("--trajectory-radius", type=float, default=2.0)
    parser.add_argument("--trajectory-elevation", type=float, default=15.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AdapterConfig(
        run_name=args.run_name,
        objectmorpher_ply=Path(args.objectmorpher_ply),
        image_width=args.image_width,
        image_height=args.image_height,
        fovy_degrees=args.fovy_degrees,
        trajectory=TrajectoryConfig(
            frames=args.trajectory_frames,
            radius=args.trajectory_radius,
            elevation_degrees=args.trajectory_elevation,
        ),
        supergaussian=SuperGaussianConfig(),
    )
    result = run_phase1_render(config, Path(args.repo_root))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
