from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..cameras.trajectory import OrbitTrajectorySpec, build_orbit_cameras


@dataclass(slots=True)
class RenderExportResult:
    image_dir: Path
    transforms_path: Path
    rendered_frames: int
    blocked_reason: str | None = None


def export_headless_frames(repo_root: Path, ply_path: Path, output_dir: Path, spec: OrbitTrajectorySpec) -> RenderExportResult:
    """Phase-1 scaffold for headless rendering.

    This intentionally stops short of forcing a risky deep integration. The next
    step is to construct ObjectMorpher-compatible MiniCam instances from the
    orbit poses and call `editing.gaussian_renderer.render` without the GUI.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _ = build_orbit_cameras(spec)
    transforms_path = output_dir / "transforms.json"
    blocked = (
        "Headless export wrapper scaffold created, but actual rendering is left "
        "for the next step after validating ObjectMorpher import/runtime contracts."
    )
    return RenderExportResult(
        image_dir=output_dir,
        transforms_path=transforms_path,
        rendered_frames=0,
        blocked_reason=blocked,
    )
