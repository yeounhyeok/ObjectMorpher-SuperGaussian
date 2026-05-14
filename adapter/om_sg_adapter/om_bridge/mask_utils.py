"""Helpers to derive a binary edit-mask from a SAM object crop (RGBA).

The modded ARAP entrypoints expect a grayscale mask whose `> 127` region
marks the editable area. Our SAM stage produces an RGBA crop where the
alpha channel encodes object membership, so we just promote the alpha
band into a single-channel mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def sam_crop_alpha_to_mask(crop_path: Path, output_path: Path, threshold: int = 127) -> Path:
    img = Image.open(crop_path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = np.asarray(img)[..., 3]
    mask = (alpha > threshold).astype(np.uint8) * 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path)
    return output_path
