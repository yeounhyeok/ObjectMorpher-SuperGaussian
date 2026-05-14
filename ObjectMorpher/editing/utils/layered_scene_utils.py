import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
from plyfile import PlyData, PlyElement


CORE_PREFIXES = ("f_dc_", "f_rest_", "scale_", "rot_", "fea_")
REQUIRED_GAUSSIAN_FIELDS = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity")


def _field_index(name: str, prefix: str) -> int:
    try:
        return int(name[len(prefix) :])
    except ValueError:
        return -1


def _sorted_prefixed(names: Iterable[str], prefix: str) -> list[str]:
    return sorted([name for name in names if name.startswith(prefix)], key=lambda value: _field_index(value, prefix))


def _read_vertices(path: str | Path) -> np.ndarray:
    plydata = PlyData.read(str(path))
    if not plydata.elements or plydata.elements[0].name != "vertex":
        raise ValueError(f"PLY does not start with a vertex element: {path}")
    vertices = plydata.elements[0].data
    names = vertices.dtype.names or ()
    missing = [name for name in REQUIRED_GAUSSIAN_FIELDS if name not in names]
    if missing:
        raise ValueError(f"PLY is missing required Gaussian fields {missing}: {path}")
    return vertices


def _canonical_output_fields(background: np.ndarray, foreground: np.ndarray) -> list[str]:
    bg_names = set(background.dtype.names or ())
    fg_names = set(foreground.dtype.names or ())
    all_names = bg_names | fg_names

    fields = ["x", "y", "z", "nx", "ny", "nz"]
    fields.extend(_sorted_prefixed(all_names, "f_dc_"))
    fields.extend(_sorted_prefixed(all_names, "f_rest_"))
    fields.append("opacity")
    fields.extend(_sorted_prefixed(all_names, "scale_"))
    fields.extend(_sorted_prefixed(all_names, "rot_"))
    fields.extend(_sorted_prefixed(all_names, "fea_"))

    seen = set()
    unique_fields = []
    for field in fields:
        if field in seen:
            continue
        seen.add(field)
        unique_fields.append(field)
    return unique_fields


def _copy_rows(dst: np.ndarray, start: int, src: np.ndarray, fields: list[str]) -> None:
    count = src.shape[0]
    src_names = set(src.dtype.names or ())
    for field in fields:
        if field in src_names:
            dst[field][start : start + count] = src[field]
        elif field == "nx" or field == "ny" or field == "nz" or any(field.startswith(prefix) for prefix in ("f_rest_", "fea_")):
            dst[field][start : start + count] = 0.0
        else:
            raise ValueError(f"Cannot synthesize missing Gaussian field '{field}'")


def load_editable_mask(edit_state_path: str | Path, *, key: str = "editable_mask", expected_count: Optional[int] = None) -> np.ndarray:
    state = np.load(str(edit_state_path))
    if key not in state:
        raise KeyError(f"Edit state has no '{key}' array: {edit_state_path}")
    mask = np.asarray(state[key]).astype(bool).reshape(-1)
    if expected_count is not None and mask.shape[0] != expected_count:
        raise ValueError(
            f"Editable mask length {mask.shape[0]} does not match foreground Gaussian count {expected_count}: {edit_state_path}"
        )
    return mask


def merge_layered_gaussians(
    *,
    background_ply: str | Path,
    foreground_ply: str | Path,
    edit_state: str | Path,
    output_ply: str | Path,
    output_edit_state: str | Path | None = None,
    output_meta: str | Path | None = None,
    background_scene_meta: str | Path | None = None,
    source_image: str | Path | None = None,
    mask_key: str = "editable_mask",
) -> Dict[str, Any]:
    """Merge full-scene background splats with only the editable foreground splats.

    The intended layer order is:
    1. all Gaussians from an inpainted-background full-scene lifting
    2. editable object Gaussians from the deformed foreground scene

    Background rows from the foreground scene are intentionally discarded so empty
    regions exposed by deformation come from the lifted inpainted background.
    """
    background_ply = Path(background_ply)
    foreground_ply = Path(foreground_ply)
    output_ply = Path(output_ply)

    background = _read_vertices(background_ply)
    foreground = _read_vertices(foreground_ply)
    editable_mask = load_editable_mask(edit_state, key=mask_key, expected_count=foreground.shape[0])
    foreground_selected = foreground[editable_mask]

    fields = _canonical_output_fields(background, foreground)
    dtype = [(field, "f4") for field in fields]
    merged_count = int(background.shape[0] + foreground_selected.shape[0])
    merged = np.zeros(merged_count, dtype=dtype)
    _copy_rows(merged, 0, background, fields)
    _copy_rows(merged, int(background.shape[0]), foreground_selected, fields)

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(merged, "vertex")]).write(str(output_ply))

    merged_editable_mask = np.zeros(merged_count, dtype=bool)
    merged_editable_mask[int(background.shape[0]) :] = True

    output_edit_state_path = Path(output_edit_state) if output_edit_state is not None else output_ply.with_suffix(".edit_state.npz")
    output_edit_state_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_edit_state_path,
        editable_mask=merged_editable_mask,
        foreground_editable_mask=editable_mask,
        background_count=np.asarray([background.shape[0]], dtype=np.int64),
        foreground_count=np.asarray([foreground.shape[0]], dtype=np.int64),
        foreground_selected_count=np.asarray([foreground_selected.shape[0]], dtype=np.int64),
    )

    summary: Dict[str, Any] = {
        "format": "layered_full_scene",
        "background_ply": str(background_ply),
        "foreground_ply": str(foreground_ply),
        "edit_state": str(edit_state),
        "output_ply": str(output_ply),
        "output_edit_state": str(output_edit_state_path),
        "source_image": str(source_image) if source_image is not None else None,
        "gaussian_counts": {
            "background": int(background.shape[0]),
            "foreground_total": int(foreground.shape[0]),
            "foreground_selected": int(foreground_selected.shape[0]),
            "merged": merged_count,
        },
    }

    if background_scene_meta is not None:
        background_scene_meta = Path(background_scene_meta)
        if background_scene_meta.exists():
            base_meta = json.loads(background_scene_meta.read_text(encoding="utf-8"))
            summary = {**base_meta, **summary, "background_scene_meta": str(background_scene_meta)}

    if output_meta is not None:
        output_meta_path = Path(output_meta)
        output_meta_path.parent.mkdir(parents=True, exist_ok=True)
        output_meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["output_meta"] = str(output_meta_path)

    return summary
