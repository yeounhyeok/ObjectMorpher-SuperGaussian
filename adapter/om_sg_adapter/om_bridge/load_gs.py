from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType


def _editing_root(repo_root: Path) -> Path:
    return repo_root / "ObjectMorpher" / "editing"


def add_objectmorpher_editing_to_syspath(repo_root: Path) -> Path:
    editing_root = _editing_root(repo_root)
    path_str = str(editing_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return editing_root


def load_gaussian_model(repo_root: Path, ply_path: Path, sh_degree: int = 3) -> ModuleType:
    add_objectmorpher_editing_to_syspath(repo_root)
    from scene.gaussian_model import GaussianModel  # type: ignore

    model = GaussianModel(sh_degree)
    model.load_ply(str(ply_path))
    return model
