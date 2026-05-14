from __future__ import annotations

from pathlib import Path

from ..config import AdapterConfig
from ..paths import build_run_paths, ensure_run_dirs
from ..cameras.trajectory import OrbitTrajectorySpec
from ..om_bridge.render_frames import export_headless_frames


def run_phase1_render(config: AdapterConfig, repo_root: Path) -> dict:
    paths = build_run_paths(repo_root, config.run_name, config.output_root)
    ensure_run_dirs(paths)
    ply_path = config.objectmorpher_ply
    if not ply_path.is_absolute():
        ply_path = repo_root / ply_path
    result = export_headless_frames(
        repo_root=repo_root,
        ply_path=ply_path,
        output_dir=paths.om_baseline_frames,
        spec=OrbitTrajectorySpec(
            frames=config.trajectory.frames,
            radius=config.trajectory.radius,
            elevation_degrees=config.trajectory.elevation_degrees,
        ),
        width=config.image_width,
        height=config.image_height,
        fovy_degrees=config.fovy_degrees,
    )
    return {
        "run_root": str(paths.root),
        "image_dir": str(result.image_dir),
        "transforms_path": str(result.transforms_path),
        "rendered_frames": result.rendered_frames,
        "blocked_reason": result.blocked_reason,
    }
