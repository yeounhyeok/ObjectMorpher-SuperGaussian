from __future__ import annotations

from pathlib import Path

from ..cameras.conventions import opencv_c2w_to_supergaussian_c2w
from ..cameras.transforms_io import TransformFrame, TransformsDocument, write_transforms_json


def write_supergaussian_transforms(path: Path, poses: list, width: int, height: int, focal_x: float, focal_y: float) -> None:
    frames = [
        TransformFrame(
            file_path=f"{idx:04d}.png",
            transform_matrix=opencv_c2w_to_supergaussian_c2w(pose),
        )
        for idx, pose in enumerate(poses)
    ]
    write_transforms_json(
        path,
        TransformsDocument(
            fl_x=focal_x,
            fl_y=focal_y,
            cx=width / 2.0,
            cy=height / 2.0,
            w=width,
            h=height,
            frames=frames,
        ),
    )
