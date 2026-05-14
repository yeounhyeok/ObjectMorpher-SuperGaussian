"""Build a scene_meta JSON expected by modded ObjectMorpher editing tools.

The modded `virtual_arap_tail_lift.py` (and related programmatic ARAP
modules) consume a JSON file describing the camera that observed the
source image, in OpenCV convention:

    {
      "image_size": [width, height],
      "intrinsic":  [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "extrinsic":  [[..4x4..]]   # world-to-camera (w2c)
    }

For our TRELLIS image-to-3D pipeline the source camera is implicit (the
input image is treated as a frontal view of the object), so we assume
an OpenCV camera placed at (0, 0, -distance) looking toward +z. The
object PLY is taken as object-centric, so the extrinsic is a simple
translation along z.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(slots=True)
class FrontViewCamera:
    image_size: Tuple[int, int]
    fovy_degrees: float = 50.0
    distance: float = 2.0  # world units; assumes object roughly within a unit box

    def intrinsic(self) -> list[list[float]]:
        w, h = self.image_size
        fovy = math.radians(self.fovy_degrees)
        fy = 0.5 * h / math.tan(fovy / 2.0)
        if w == h:
            fx = fy
        else:
            fovx = 2.0 * math.atan(w / (2.0 * fy))
            fx = 0.5 * w / math.tan(fovx / 2.0)
        cx, cy = w / 2.0, h / 2.0
        return [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]]

    def extrinsic_w2c(self) -> list[list[float]]:
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, float(self.distance)],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def to_dict(self) -> dict:
        w, h = self.image_size
        return {
            "image_size": [int(w), int(h)],
            "intrinsic": self.intrinsic(),
            "extrinsic": self.extrinsic_w2c(),
            "fovy_degrees": float(self.fovy_degrees),
            "distance": float(self.distance),
            "convention": "opencv",
        }


def write_scene_meta(
    output_path: Path,
    image_size: Tuple[int, int],
    fovy_degrees: float = 50.0,
    distance: float = 2.0,
) -> Path:
    cam = FrontViewCamera(image_size=image_size, fovy_degrees=fovy_degrees, distance=distance)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cam.to_dict(), indent=2), encoding="utf-8")
    return output_path
