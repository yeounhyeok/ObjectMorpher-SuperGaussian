from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class TrajectoryConfig:
    kind: str = "orbit"
    frames: int = 24
    radius: float = 2.0
    elevation_degrees: float = 15.0


@dataclass(slots=True)
class SuperGaussianConfig:
    prior: str = "realbasicvsr"
    low_res_size: int = 64
    high_res_size: int = 256


@dataclass(slots=True)
class AdapterConfig:
    run_name: str
    objectmorpher_ply: Path
    output_root: Path = Path("runs")
    image_width: int = 256
    image_height: int = 256
    fovy_degrees: float = 60.0
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    supergaussian: SuperGaussianConfig = field(default_factory=SuperGaussianConfig)
