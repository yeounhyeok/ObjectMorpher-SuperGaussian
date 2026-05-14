from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class FittingRequest:
    scene_root: Path
    output_root: Path


# Phase 1 intentionally defers actual SuperGaussian 3DGS refitting.
