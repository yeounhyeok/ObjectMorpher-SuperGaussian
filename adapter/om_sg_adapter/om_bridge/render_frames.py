from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

from ..cameras.trajectory import OrbitTrajectorySpec
from ..cameras.transforms_io import TransformFrame, TransformsDocument, write_transforms_json


SH_C0 = 0.28209479177387814


@dataclass(slots=True)
class RenderExportResult:
    image_dir: Path
    transforms_path: Path
    rendered_frames: int
    blocked_reason: str | None = None


def _load_inria_gaussians(ply_path: Path):
    plydata = PlyData.read(str(ply_path))
    v = plydata["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)
    opacity = v["opacity"].astype(np.float32)
    scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)
    means = torch.from_numpy(xyz)
    scales = torch.from_numpy(np.exp(scale))
    opacities = torch.from_numpy(1.0 / (1.0 + np.exp(-opacity)))
    quats = torch.from_numpy(rot)  # wxyz, INRIA convention
    colors = torch.from_numpy(np.clip(SH_C0 * f_dc + 0.5, 0.0, 1.0))
    return means, quats, scales, opacities, colors


def _build_opencv_c2w(cam_pos: np.ndarray, target: np.ndarray, up_world: np.ndarray) -> np.ndarray:
    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up_world)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(forward, right)
    up = up / (np.linalg.norm(up) + 1e-8)
    rot = np.stack([right, up, forward], axis=1)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rot
    c2w[:3, 3] = cam_pos
    return c2w


def _intrinsics(width: int, height: int, fovy_deg: float) -> tuple[float, float, float, float]:
    fovy = math.radians(fovy_deg)
    fy = 0.5 * height / math.tan(fovy / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    if width == height:
        fx = fy
    else:
        fovx = 2.0 * math.atan(width / (2.0 * fy))
        fx = 0.5 * width / math.tan(fovx / 2.0)
    return float(fx), float(fy), float(cx), float(cy)


def export_headless_frames(
    repo_root: Path,
    ply_path: Path,
    output_dir: Path,
    spec: OrbitTrajectorySpec,
    width: int = 256,
    height: int = 256,
    fovy_degrees: float = 50.0,
    device: str = "cuda",
) -> RenderExportResult:
    from gsplat import rasterization

    output_dir.mkdir(parents=True, exist_ok=True)

    means, quats, scales, opacities, colors = _load_inria_gaussians(ply_path)
    means = means.to(device)
    quats = quats.to(device)
    scales = scales.to(device)
    opacities = opacities.to(device)
    colors = colors.to(device)

    target = np.array(spec.target, dtype=np.float32)
    elevation = math.radians(spec.elevation_degrees)
    up_world = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    fx, fy, cx, cy = _intrinsics(width, height, fovy_degrees)
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )[None]

    frames: list[TransformFrame] = []
    for idx in range(spec.frames):
        azimuth = 2.0 * math.pi * idx / spec.frames
        cam_pos = np.array(
            [
                spec.radius * math.cos(elevation) * math.sin(azimuth) + target[0],
                -spec.radius * math.sin(elevation) + target[1],
                spec.radius * math.cos(elevation) * math.cos(azimuth) + target[2],
            ],
            dtype=np.float32,
        )
        c2w_np = _build_opencv_c2w(cam_pos, target, up_world)
        viewmat = torch.from_numpy(np.linalg.inv(c2w_np))[None].to(device, dtype=torch.float32)

        out = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
        )
        img = out[0][0]
        img_np = np.clip(img.detach().cpu().numpy() * 255.0, 0.0, 255.0).astype(np.uint8)
        fname = f"{idx:04d}.png"
        Image.fromarray(img_np).save(output_dir / fname)
        frames.append(TransformFrame(file_path=fname, transform_matrix=c2w_np.tolist()))

    transforms_path = output_dir / "transforms.json"
    write_transforms_json(
        transforms_path,
        TransformsDocument(
            fl_x=fx,
            fl_y=fy,
            cx=cx,
            cy=cy,
            w=width,
            h=height,
            frames=frames,
        ),
    )

    return RenderExportResult(
        image_dir=output_dir,
        transforms_path=transforms_path,
        rendered_frames=spec.frames,
        blocked_reason=None,
    )
