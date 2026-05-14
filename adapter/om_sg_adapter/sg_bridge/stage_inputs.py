from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass(slots=True)
class StageInputsResult:
    scene_root: Path
    low_res_dir: Path
    transforms_path: Path


def stage_supergaussian_inputs(source_frames_dir: Path, transforms_path: Path, scene_root: Path, low_res_dir_name: str = "lr_64x64") -> StageInputsResult:
    scene_root.mkdir(parents=True, exist_ok=True)
    low_res_dir = scene_root / low_res_dir_name
    low_res_dir.mkdir(parents=True, exist_ok=True)
    for frame in sorted(source_frames_dir.glob("*.png")):
        shutil.copy2(frame, low_res_dir / frame.name)
    shutil.copy2(transforms_path, scene_root / "transforms.json")
    return StageInputsResult(scene_root=scene_root, low_res_dir=low_res_dir, transforms_path=scene_root / "transforms.json")
