from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TransformFrame:
    file_path: str
    transform_matrix: list[list[float]]


@dataclass(slots=True)
class TransformsDocument:
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    w: int
    h: int
    frames: list[TransformFrame]


def write_transforms_json(path: Path, document: TransformsDocument) -> None:
    payload: dict[str, Any] = asdict(document)
    path.write_text(json.dumps(payload, indent=2))
