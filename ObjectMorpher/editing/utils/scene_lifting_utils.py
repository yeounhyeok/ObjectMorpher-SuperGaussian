import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image


SHARP_COORDINATE_CONVENTION = "opencv_x_right_y_down_z_forward"


def _read_single_property_element(plydata: Any, element_name: str, property_name: str) -> Optional[list]:
    for element in plydata.elements:
        if element.name == element_name and property_name in element.data.dtype.names:
            values = np.asarray(element.data[property_name])
            if values.shape[0] == 1:
                values = values[0]
            return np.asarray(values).reshape(-1).tolist()
    return None


def extract_sharp_scene_metadata(
    ply_path: str | Path,
    output_path: str | Path,
    source_image: str | Path | None = None,
) -> Dict[str, Any]:
    """Extract SHARP's non-vertex PLY camera metadata into a JSON sidecar."""
    from plyfile import PlyData

    ply_path = Path(ply_path)
    output_path = Path(output_path)
    plydata = PlyData.read(str(ply_path))

    image_size = _read_single_property_element(plydata, "image_size", "image_size")
    intrinsic = _read_single_property_element(plydata, "intrinsic", "intrinsic")
    extrinsic = _read_single_property_element(plydata, "extrinsic", "extrinsic")
    frame = _read_single_property_element(plydata, "frame", "frame")
    disparity = _read_single_property_element(plydata, "disparity", "disparity")
    color_space = _read_single_property_element(plydata, "color_space", "color_space")
    version = _read_single_property_element(plydata, "version", "version")

    source_size = None
    if source_image is not None:
        with Image.open(source_image) as image:
            source_size = [int(image.width), int(image.height)]

    if image_size is None:
        if source_size is None:
            raise ValueError("SHARP PLY has no image_size element and no source image was provided")
        image_size = source_size

    if len(image_size) != 2:
        raise ValueError(f"Expected image_size to contain [width, height], got {image_size}")

    width, height = int(image_size[0]), int(image_size[1])
    if intrinsic is None:
        focal = 0.8 * max(width, height)
        intrinsic_matrix = np.array(
            [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    else:
        intrinsic_matrix = np.asarray(intrinsic, dtype=np.float32).reshape(3, 3)

    if extrinsic is None:
        extrinsic_matrix = np.eye(4, dtype=np.float32)
    else:
        extrinsic_values = np.asarray(extrinsic, dtype=np.float32)
        if extrinsic_values.size == 12:
            extrinsic_matrix = np.eye(4, dtype=np.float32)
            extrinsic_matrix[:3] = extrinsic_values.reshape(3, 4)
        elif extrinsic_values.size == 16:
            extrinsic_matrix = extrinsic_values.reshape(4, 4)
        else:
            raise ValueError(f"Unrecognized extrinsic element length: {extrinsic_values.size}")

    metadata: Dict[str, Any] = {
        "format": "sharp",
        "source_ply": str(ply_path),
        "source_image": str(source_image) if source_image is not None else None,
        "image_size": [width, height],
        "source_image_size": source_size,
        "intrinsic": intrinsic_matrix.tolist(),
        "extrinsic": extrinsic_matrix.tolist(),
        "extrinsic_convention": "world_to_camera",
        "coordinate_convention": SHARP_COORDINATE_CONVENTION,
        "frame": frame,
        "disparity": disparity,
        "color_space": color_space,
        "version": version,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_scene_metadata(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scene metadata file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def project_points_to_pixels(points: np.ndarray, metadata: Dict[str, Any], image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points to pixel coordinates using OpenCV-style camera metadata."""
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")

    width, height = image_size
    meta_width, meta_height = metadata.get("image_size", [width, height])
    intrinsic = np.asarray(metadata.get("intrinsic", np.eye(3)), dtype=np.float32).reshape(3, 3).copy()
    if meta_width and meta_height and (int(meta_width) != width or int(meta_height) != height):
        sx = width / float(meta_width)
        sy = height / float(meta_height)
        intrinsic[0, :] *= sx
        intrinsic[1, :] *= sy

    extrinsic = np.asarray(metadata.get("extrinsic", np.eye(4)), dtype=np.float32)
    if extrinsic.size == 12:
        full = np.eye(4, dtype=np.float32)
        full[:3] = extrinsic.reshape(3, 4)
        extrinsic = full
    else:
        extrinsic = extrinsic.reshape(4, 4)

    homog = np.concatenate([points.astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    camera_points = (extrinsic @ homog.T).T[:, :3]
    z = camera_points[:, 2]
    valid_depth = z > 1e-6

    pixels = np.full((points.shape[0], 2), -1.0, dtype=np.float32)
    pixels[valid_depth, 0] = intrinsic[0, 0] * (camera_points[valid_depth, 0] / z[valid_depth]) + intrinsic[0, 2]
    pixels[valid_depth, 1] = intrinsic[1, 1] * (camera_points[valid_depth, 1] / z[valid_depth]) + intrinsic[1, 2]
    return pixels, valid_depth


def editable_mask_from_projection(
    xyz: torch.Tensor,
    mask_path: str | Path,
    metadata: Dict[str, Any],
    *,
    opacity: torch.Tensor | None = None,
    min_opacity: float = 0.05,
) -> torch.Tensor:
    """Return a boolean mask selecting Gaussians whose projected centers land in a 2D edit mask."""
    mask_image = Image.open(mask_path).convert("L")
    mask_np = np.asarray(mask_image) > 127
    height, width = mask_np.shape

    points = xyz.detach().cpu().numpy()
    pixels, valid_depth = project_points_to_pixels(points, metadata, (width, height))
    u = np.rint(pixels[:, 0]).astype(np.int64)
    v = np.rint(pixels[:, 1]).astype(np.int64)
    in_bounds = valid_depth & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    selected = np.zeros(points.shape[0], dtype=bool)
    selected[in_bounds] = mask_np[v[in_bounds], u[in_bounds]]

    if opacity is not None:
        opacity_np = opacity.detach().cpu().reshape(-1).numpy()
        selected &= opacity_np > min_opacity

    return torch.from_numpy(selected).to(device=xyz.device)


def gate_deformation_value(value, editable_mask: torch.Tensor, *, background_value=0.0):
    """Keep deformation/update tensors only on editable Gaussian rows."""
    if not torch.is_tensor(value):
        return value
    if value.shape[0] != editable_mask.shape[0]:
        raise ValueError(f"Cannot gate tensor with first dimension {value.shape[0]} by mask of length {editable_mask.shape[0]}")

    view_shape = [editable_mask.shape[0]] + [1] * (value.ndim - 1)
    mask = editable_mask.to(device=value.device, dtype=torch.bool).reshape(view_shape)
    if torch.is_tensor(background_value):
        background = background_value.to(device=value.device, dtype=value.dtype)
        while background.ndim < value.ndim:
            background = background.unsqueeze(0)
        background = background.expand_as(value)
    else:
        background = torch.full_like(value, float(background_value))
    return torch.where(mask, value, background)


def gate_deformation_values(
    *,
    editable_mask: torch.Tensor,
    d_xyz,
    d_rotation,
    d_scaling,
    d_opacity=None,
    d_color=None,
    d_rotation_bias=None,
) -> Dict[str, Any]:
    """Gate ARAP deformation tensors so background splats remain unchanged."""
    gated: Dict[str, Any] = {
        "d_xyz": gate_deformation_value(d_xyz, editable_mask, background_value=0.0),
        "d_rotation": gate_deformation_value(d_rotation, editable_mask, background_value=0.0),
        "d_scaling": gate_deformation_value(d_scaling, editable_mask, background_value=0.0),
        "d_opacity": gate_deformation_value(d_opacity, editable_mask, background_value=0.0),
        "d_color": gate_deformation_value(d_color, editable_mask, background_value=0.0),
        "d_rotation_bias": d_rotation_bias,
    }

    if torch.is_tensor(d_rotation_bias):
        identity = torch.zeros((d_rotation_bias.shape[-1],), dtype=d_rotation_bias.dtype, device=d_rotation_bias.device)
        if d_rotation_bias.shape[-1] == 4:
            identity[0] = 1.0
        gated["d_rotation_bias"] = gate_deformation_value(d_rotation_bias, editable_mask, background_value=identity)
    return gated
