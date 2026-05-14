import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


def _resolve_path(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _image_size(path: Path) -> Optional[list[int]]:
    if not path.exists():
        return None
    with Image.open(path) as image:
        return [int(image.width), int(image.height)]


def _intrinsic_from_values(values: Dict[str, Any], image_size: list[int]) -> np.ndarray:
    width, height = int(image_size[0]), int(image_size[1])
    if values.get("intrinsic") is not None:
        return np.asarray(values["intrinsic"], dtype=np.float32).reshape(3, 3)

    fl_x = values.get("fl_x") or values.get("fx")
    fl_y = values.get("fl_y") or values.get("fy")
    if fl_x is None and values.get("camera_angle_x") is not None:
        fl_x = 0.5 * width / math.tan(0.5 * float(values["camera_angle_x"]))
    if fl_y is None and values.get("camera_angle_y") is not None:
        fl_y = 0.5 * height / math.tan(0.5 * float(values["camera_angle_y"]))
    if fl_x is None and fl_y is None:
        focal = 0.8 * max(width, height)
        fl_x = focal
        fl_y = focal
    elif fl_x is None:
        fl_x = fl_y
    elif fl_y is None:
        fl_y = fl_x

    cx = values.get("cx", width * 0.5)
    cy = values.get("cy", height * 0.5)
    return np.asarray([[fl_x, 0.0, cx], [0.0, fl_y, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _extrinsic_from_frame(frame: Dict[str, Any]) -> np.ndarray:
    if "extrinsic" in frame:
        values = np.asarray(frame["extrinsic"], dtype=np.float32)
        if values.size == 12:
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3] = values.reshape(3, 4)
            return matrix
        return values.reshape(4, 4)
    if "w2c" in frame:
        return np.asarray(frame["w2c"], dtype=np.float32).reshape(4, 4)
    if "world_to_camera" in frame:
        return np.asarray(frame["world_to_camera"], dtype=np.float32).reshape(4, 4)
    if "transform_matrix" in frame:
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float32).reshape(4, 4)
        return np.linalg.inv(c2w).astype(np.float32)
    raise ValueError("Frame has no extrinsic, w2c, world_to_camera, or transform_matrix")


def _normalize_view(view: Dict[str, Any], root: Path, defaults: Dict[str, Any], index: int) -> Dict[str, Any]:
    image_value = view.get("image_path") or view.get("file_path") or view.get("path")
    if not image_value:
        raise ValueError(f"View {index} does not define image_path/file_path")
    image_path = _resolve_path(image_value, root)
    if image_path.suffix == "":
        for suffix in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = image_path.with_suffix(suffix)
            if candidate.exists():
                image_path = candidate
                break

    image_size = view.get("image_size") or [view.get("w") or defaults.get("w"), view.get("h") or defaults.get("h")]
    if not image_size or image_size[0] is None or image_size[1] is None:
        detected = _image_size(image_path)
        if detected is None:
            raise FileNotFoundError(f"Cannot determine image size for view {index}: {image_path}")
        image_size = detected
    image_size = [int(image_size[0]), int(image_size[1])]

    values = dict(defaults)
    values.update(view)
    intrinsic = _intrinsic_from_values(values, image_size)
    extrinsic = _extrinsic_from_frame(view)

    return {
        "name": str(view.get("name") or image_path.stem or f"view_{index:03d}"),
        "image_path": str(image_path),
        "image_size": image_size,
        "intrinsic": intrinsic.tolist(),
        "extrinsic": extrinsic.tolist(),
        "extrinsic_convention": "world_to_camera",
        "role": str(view.get("role") or ("source" if index == 0 else "generated")),
    }


def _normalize_existing_scene_meta(scene_meta_path: Path, output_path: Path) -> Dict[str, Any]:
    root = scene_meta_path.parent
    metadata = json.loads(scene_meta_path.read_text(encoding="utf-8"))
    views = metadata.get("views") or metadata.get("frames") or []
    if views:
        defaults = {
            "w": metadata.get("w") or (metadata.get("image_size") or [None, None])[0],
            "h": metadata.get("h") or (metadata.get("image_size") or [None, None])[1],
            "fl_x": metadata.get("fl_x"),
            "fl_y": metadata.get("fl_y"),
            "cx": metadata.get("cx"),
            "cy": metadata.get("cy"),
            "camera_angle_x": metadata.get("camera_angle_x"),
            "camera_angle_y": metadata.get("camera_angle_y"),
            "intrinsic": metadata.get("intrinsic"),
        }
        normalized_views = [_normalize_view(view, root, defaults, idx) for idx, view in enumerate(views)]
        metadata["views"] = normalized_views
        first = normalized_views[0]
        metadata["image_size"] = first["image_size"]
        metadata["intrinsic"] = first["intrinsic"]
        metadata["extrinsic"] = first["extrinsic"]
    metadata["format"] = metadata.get("format") or "pseudo_multiview"
    metadata["scene_meta_source"] = str(scene_meta_path)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def normalize_pseudo_multiview_scene(
    run_dir: str | Path,
    output_path: str | Path | None = None,
    source_image: str | Path | None = None,
    ply_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Normalize ExScene/CAT3D-style outputs into ObjectMorpher's scene_meta.json.

    Accepted inputs:
    - scene_meta.json with `views`
    - transforms.json / transforms_train.json / cameras.json with NeRF-style `frames`
    Each view must define an image path plus either world-to-camera extrinsic or
    a NeRF-style camera-to-world `transform_matrix`.
    """
    run_dir = Path(run_dir)
    output_path = Path(output_path) if output_path is not None else run_dir / "scene_meta.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = [
        run_dir / "scene_meta.json",
        run_dir / "transforms.json",
        run_dir / "transforms_train.json",
        run_dir / "cameras.json",
    ]
    meta_source = next((path for path in candidates if path.exists()), None)
    if meta_source is None:
        raise FileNotFoundError(
            f"No scene_meta.json, transforms.json, transforms_train.json, or cameras.json found under {run_dir}"
        )

    metadata = _normalize_existing_scene_meta(meta_source, output_path)
    if source_image is not None:
        metadata["source_image"] = str(Path(source_image))
    if ply_path is not None:
        metadata["source_ply"] = str(Path(ply_path))
    metadata["format"] = "pseudo_multiview"
    metadata["coordinate_convention"] = metadata.get("coordinate_convention", "opencv_x_right_y_down_z_forward")
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def find_ply(run_dir: str | Path, explicit_ply: str | Path | None = None) -> Path:
    if explicit_ply:
        path = Path(explicit_ply)
        if not path.exists():
            raise FileNotFoundError(f"Pseudo-multiview PLY not found: {path}")
        return path
    run_dir = Path(run_dir)
    candidates = sorted(run_dir.rglob("*.ply"))
    if not candidates:
        raise FileNotFoundError(f"No PLY found under pseudo-multiview output dir: {run_dir}")
    return candidates[0]


def run_template_command(command_template: str, *, image_path: Path, run_dir: Path, scene_meta: Path, ply_path: Path | None = None) -> None:
    command = command_template.format(
        input=str(image_path),
        source_image=str(image_path),
        output_dir=str(run_dir),
        scene_meta=str(scene_meta),
        ply=str(ply_path or run_dir / "scene.ply"),
    )
    completed = subprocess.run(command, shell=True, text=True, capture_output=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        if len(detail) > 4000:
            detail = detail[-4000:]
        raise RuntimeError(f"Pseudo-multiview command failed: {detail}")


def prepare_pseudo_multiview_lifting(
    *,
    image_path: str | Path,
    output_root: str | Path,
    generator_command: str = "",
    reconstruction_command: str = "",
    import_dir: str | Path | None = None,
    explicit_ply: str | Path | None = None,
) -> Dict[str, Path]:
    image_path = Path(image_path)
    output_root = Path(output_root)
    if import_dir:
        run_dir = Path(import_dir)
    else:
        from datetime import datetime

        run_dir = output_root / f"{image_path.stem}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if not generator_command:
            raise ValueError("--pseudo-mv-command is required unless --pseudo-mv-import-dir is provided")
        run_template_command(generator_command, image_path=image_path, run_dir=run_dir, scene_meta=run_dir / "scene_meta.json")

    scene_meta = run_dir / "scene_meta.json"
    try:
        ply_path = find_ply(run_dir, explicit_ply)
    except FileNotFoundError:
        if not reconstruction_command:
            raise
        run_template_command(reconstruction_command, image_path=image_path, run_dir=run_dir, scene_meta=scene_meta)
        ply_path = find_ply(run_dir, explicit_ply)

    normalize_pseudo_multiview_scene(run_dir, scene_meta, image_path, ply_path)
    copied_source = run_dir / image_path.name
    if not copied_source.exists() and image_path.exists() and not import_dir:
        shutil.copy2(image_path, copied_source)
    return {"directory": run_dir, "ply": ply_path, "scene_meta": scene_meta}
