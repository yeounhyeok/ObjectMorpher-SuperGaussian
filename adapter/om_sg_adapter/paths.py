from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunPaths:
    root: Path
    om_baseline_frames: Path
    sg_pseudo_gt: Path
    optimization: Path
    reports: Path


def build_run_paths(repo_root: Path, run_name: str, output_root: Path | str = "runs") -> RunPaths:
    base = repo_root / Path(output_root) / run_name
    return RunPaths(
        root=base,
        om_baseline_frames=base / "om_baseline_frames",
        sg_pseudo_gt=base / "sg_pseudo_gt",
        optimization=base / "optimization",
        reports=base / "reports",
    )


def ensure_run_dirs(paths: RunPaths) -> None:
    for path in (paths.root, paths.om_baseline_frames, paths.sg_pseudo_gt, paths.optimization, paths.reports):
        path.mkdir(parents=True, exist_ok=True)
