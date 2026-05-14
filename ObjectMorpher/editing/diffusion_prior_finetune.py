import argparse
import atexit
import importlib.util
import json
import math
import select
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

EDITING_DIR = Path(__file__).resolve().parent
if str(EDITING_DIR) not in sys.path:
    sys.path.insert(0, str(EDITING_DIR))

from utils.loss_utils import l1_loss, ssim  # noqa: E402
from utils.scene_lifting_utils import editable_mask_from_projection, load_scene_metadata  # noqa: E402

render = None
GaussianModel = None
stable_sr_worker_process = None
stable_sr_worker_log = None


def ensure_runtime_imports() -> None:
    global render, GaussianModel
    if render is None:
        from gaussian_renderer import render as renderer_render

        render = renderer_render
    if GaussianModel is None:
        module_path = EDITING_DIR / "scene" / "gaussian_model.py"
        spec = importlib.util.spec_from_file_location("objectmorpher_gaussian_model", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load GaussianModel from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        GaussianModel = module.GaussianModel


def get_projection_matrix(znear: float, zfar: float, fovx: float, fovy: float) -> torch.Tensor:
    tan_half_fovy = math.tan(fovy / 2)
    tan_half_fovx = math.tan(fovx / 2)
    projection = torch.zeros(4, 4)
    projection[0, 0] = 1 / tan_half_fovx
    projection[1, 1] = 1 / tan_half_fovy
    projection[3, 2] = 1.0
    projection[2, 2] = zfar / (zfar - znear)
    projection[2, 3] = -(zfar * znear) / (zfar - znear)
    return projection


class MetadataCamera:
    def __init__(
        self,
        w2c: np.ndarray,
        intrinsic: np.ndarray,
        width: int,
        height: int,
        fid: int = 0,
        image_path: str | None = None,
        name: str | None = None,
    ):
        self.image_width = width
        self.image_height = height
        self.znear = 0.01
        self.zfar = 100.0
        self.index = int(fid)
        self.fid = torch.tensor(fid, device="cuda").float()
        self.w2c = np.asarray(w2c, dtype=np.float32).reshape(4, 4)
        self.intrinsic = np.asarray(intrinsic, dtype=np.float32).reshape(3, 3)
        self.image_path = image_path
        self.name = name or f"view_{fid:03d}"

        fx = float(self.intrinsic[0, 0])
        fy = float(self.intrinsic[1, 1])
        self.FoVx = 2.0 * math.atan(width / max(2.0 * fx, 1e-6))
        self.FoVy = 2.0 * math.atan(height / max(2.0 * fy, 1e-6))

        c2w = np.linalg.inv(self.w2c)
        renderer_w2c = np.array(self.w2c, dtype=np.float32, copy=True)

        self.c2w = c2w
        self.world_view_transform = torch.tensor(renderer_w2c).transpose(0, 1).cuda().float()
        self.projection_matrix = get_projection_matrix(self.znear, self.zfar, self.FoVx, self.FoVy).transpose(0, 1).cuda().float()
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def _scaled_intrinsic(metadata: dict, width: int, height: int) -> np.ndarray:
    intrinsic = np.asarray(metadata.get("intrinsic", np.eye(3)), dtype=np.float32).reshape(3, 3).copy()
    meta_width, meta_height = metadata.get("image_size", [width, height])
    if int(meta_width) != width or int(meta_height) != height:
        intrinsic[0, :] *= width / float(meta_width)
        intrinsic[1, :] *= height / float(meta_height)
    return intrinsic


def _view_scaled_intrinsic(view: dict, width: int, height: int) -> np.ndarray:
    intrinsic = np.asarray(view.get("intrinsic", np.eye(3)), dtype=np.float32).reshape(3, 3).copy()
    meta_width, meta_height = view.get("image_size", [width, height])
    if int(meta_width) != width or int(meta_height) != height:
        intrinsic[0, :] *= width / float(meta_width)
        intrinsic[1, :] *= height / float(meta_height)
    return intrinsic


def _view_w2c(view: dict) -> np.ndarray:
    extrinsic = np.asarray(view.get("extrinsic", np.eye(4)), dtype=np.float32)
    if extrinsic.size == 12:
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3] = extrinsic.reshape(3, 4)
        return w2c
    return extrinsic.reshape(4, 4)


def build_camera_set(metadata: dict, width: int, height: int, nearby_views: int) -> List[MetadataCamera]:
    views = metadata.get("views") or []
    if views:
        cameras = []
        for idx, view in enumerate(views):
            cameras.append(
                MetadataCamera(
                    _view_w2c(view),
                    _view_scaled_intrinsic(view, width, height),
                    width,
                    height,
                    fid=idx,
                    image_path=view.get("image_path"),
                    name=view.get("name"),
                )
            )
        return cameras

    extrinsic = np.asarray(metadata.get("extrinsic", np.eye(4)), dtype=np.float32)
    if extrinsic.size == 12:
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3] = extrinsic.reshape(3, 4)
    else:
        w2c = extrinsic.reshape(4, 4)
    intrinsic = _scaled_intrinsic(metadata, width, height)

    cameras = [MetadataCamera(w2c, intrinsic, width, height, fid=0, image_path=metadata.get("source_image"), name="source")]
    if nearby_views <= 0:
        return cameras

    offsets = []
    for i in range(nearby_views):
        direction = -1.0 if i % 2 == 0 else 1.0
        magnitude = 0.015 * (1 + i // 2)
        offsets.append(direction * magnitude)

    for idx, offset in enumerate(offsets, start=1):
        jittered = w2c.copy()
        jittered[0, 3] += offset
        cameras.append(MetadataCamera(jittered, intrinsic, width, height, fid=idx, name=f"nearby_{idx}"))
    return cameras


def load_rgb_tensor(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).cuda()


def load_mask_tensor(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    array = (np.asarray(image, dtype=np.float32) > 127).astype(np.float32)
    return torch.from_numpy(array)[None].cuda()


def load_optional_camera_target(camera: MetadataCamera, source_image: Path, size: Tuple[int, int]) -> torch.Tensor | None:
    if camera.image_path:
        path = Path(camera.image_path)
        if path.exists():
            return load_rgb_tensor(path, size)
    if camera.index == 0:
        return load_rgb_tensor(source_image, size)
    return None


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((image * 255.0).round().astype(np.uint8)).save(path)


def project_editable_gaussians_to_mask(
    camera: MetadataCamera,
    xyz: torch.Tensor,
    editable_mask: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    selected = editable_mask.detach().bool().cpu().numpy()
    points = xyz.detach().cpu().numpy()
    if selected.shape[0] != points.shape[0]:
        raise ValueError(f"Editable mask length {selected.shape[0]} does not match xyz length {points.shape[0]}")

    homog = np.concatenate([points[selected].astype(np.float32), np.ones((int(selected.sum()), 1), dtype=np.float32)], axis=1)
    mask = np.zeros((camera.image_height, camera.image_width), dtype=np.uint8)
    if homog.shape[0] > 0:
        camera_points = (camera.w2c @ homog.T).T[:, :3]
        z = camera_points[:, 2]
        valid = z > 1e-6
        pixels = np.zeros((camera_points.shape[0], 2), dtype=np.float32)
        pixels[valid, 0] = camera.intrinsic[0, 0] * (camera_points[valid, 0] / z[valid]) + camera.intrinsic[0, 2]
        pixels[valid, 1] = camera.intrinsic[1, 1] * (camera_points[valid, 1] / z[valid]) + camera.intrinsic[1, 2]
        u = np.rint(pixels[:, 0]).astype(np.int64)
        v = np.rint(pixels[:, 1]).astype(np.int64)
        in_bounds = valid & (u >= 0) & (u < camera.image_width) & (v >= 0) & (v < camera.image_height)
        mask[v[in_bounds], u[in_bounds]] = 255

    if dilation > 1 and mask.any():
        kernel = int(dilation)
        if kernel % 2 == 0:
            kernel += 1
        mask_image = Image.fromarray(mask, mode="L").filter(ImageFilter.MaxFilter(kernel))
        mask = np.asarray(mask_image, dtype=np.uint8)
    return torch.from_numpy((mask > 0).astype(np.float32))[None].cuda()


def _read_worker_response(process: subprocess.Popen, output_path: Path) -> dict:
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"StableSR worker exited with code {process.returncode}")
        readable, _, _ = select.select([process.stdout], [], [], 15)
        if not readable:
            print(f"[DiffusionPrior] waiting for persistent StableSR worker output={output_path}", flush=True)
            continue
        line = process.stdout.readline()
        if not line:
            continue
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            print(f"[DiffusionPrior] StableSR worker log: {line}", flush=True)


def _shutdown_stable_sr_worker() -> None:
    global stable_sr_worker_process, stable_sr_worker_log
    process = stable_sr_worker_process
    if process is not None and process.poll() is None:
        try:
            process.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
            process.stdin.flush()
        except Exception:
            process.terminate()
    if stable_sr_worker_log is not None:
        stable_sr_worker_log.close()
    stable_sr_worker_process = None
    stable_sr_worker_log = None


def _get_stable_sr_worker(args: argparse.Namespace) -> subprocess.Popen:
    global stable_sr_worker_process, stable_sr_worker_log
    if stable_sr_worker_process is not None and stable_sr_worker_process.poll() is None:
        return stable_sr_worker_process
    if not args.stable_sr_worker_command:
        raise RuntimeError("StableSR worker command is empty")

    log_path = Path(args.out_dir) / "stablesr_worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stable_sr_worker_log = log_path.open("w", encoding="utf-8")
    print(f"[DiffusionPrior] starting persistent StableSR worker: {args.stable_sr_worker_command}", flush=True)
    stable_sr_worker_process = subprocess.Popen(
        args.stable_sr_worker_command,
        shell=True,
        cwd=str(args.stable_sr_root or Path.cwd()),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stable_sr_worker_log,
        text=True,
        bufsize=1,
    )
    atexit.register(_shutdown_stable_sr_worker)
    response = _read_worker_response(stable_sr_worker_process, Path(args.out_dir) / "worker_ready")
    if response.get("status") != "ready":
        raise RuntimeError(f"StableSR worker failed to start: {response}")
    return stable_sr_worker_process


def run_stable_sr_worker(input_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    process = _get_stable_sr_worker(args)
    request = {"input": str(input_path), "output": str(output_path)}
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    response = _read_worker_response(process, output_path)
    if response.get("status") != "ok":
        raise RuntimeError(f"StableSR worker failed: {response}")
    if not output_path.exists():
        raise FileNotFoundError(f"StableSR worker did not create {output_path}")


def run_stable_sr(input_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.reuse_stable_sr_output and output_path.exists():
        print(f"[DiffusionPrior] reuse StableSR output: {output_path}", flush=True)
        return
    if args.stable_sr_stub:
        Image.open(input_path).convert("RGB").save(output_path)
        return
    if args.stable_sr_worker_command:
        run_stable_sr_worker(input_path, output_path, args)
        return

    if args.stable_sr_command:
        command = args.stable_sr_command.format(
            input=str(input_path),
            output=str(output_path),
            root=str(args.stable_sr_root),
            env=str(args.stable_sr_env),
        )
        log_path = output_path.with_suffix(output_path.suffix + ".log")
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=str(args.stable_sr_root or Path.cwd()),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while process.poll() is None:
                print(f"[DiffusionPrior] waiting for StableSR pid={process.pid} output={output_path}", flush=True)
                time.sleep(15)
        if process.returncode != 0:
            detail = log_path.read_text(encoding="utf-8", errors="replace").strip() or f"exit code {process.returncode}"
            if len(detail) > 4000:
                detail = detail[-4000:]
            raise RuntimeError(f"StableSR command failed: {detail}")
        if not output_path.exists():
            raise FileNotFoundError(f"StableSR command did not create {output_path}")
        return

    stable_sr_root = Path(args.stable_sr_root) if args.stable_sr_root else None
    if stable_sr_root is None or not stable_sr_root.exists():
        raise RuntimeError("StableSR is not configured. Pass --stable-sr-command or use --stable-sr-stub for a dry run.")

    raise RuntimeError(
        "StableSR root was provided, but this repository does not expose a stable default CLI. "
        "Pass --stable-sr-command with {input}, {output}, and optionally {root} placeholders."
    )


def load_editable_mask(args: argparse.Namespace, gaussians, metadata: dict) -> torch.Tensor:
    if args.optimization_scope == "all":
        return torch.ones((gaussians.get_xyz.shape[0],), dtype=torch.bool, device="cuda")

    if args.edit_state:
        edit_state = np.load(args.edit_state)
        if "editable_mask" in edit_state:
            mask = torch.from_numpy(edit_state["editable_mask"].astype(np.bool_)).cuda()
            if mask.shape[0] == gaussians.get_xyz.shape[0]:
                return mask
            raise ValueError(f"edit_state editable_mask length {mask.shape[0]} does not match Gaussian count {gaussians.get_xyz.shape[0]}")

    return editable_mask_from_projection(
        gaussians.get_xyz,
        args.edit_mask,
        metadata,
        opacity=gaussians.get_opacity[:, 0],
    )


def restrict_mask_to_affected_rows(args: argparse.Namespace, editable_mask: torch.Tensor, gaussian_count: int) -> torch.Tensor:
    if not args.affected_mask_from_dxyz:
        return editable_mask
    if not args.edit_state:
        raise ValueError("--affected-mask-from-dxyz requires --edit-state")

    edit_state = np.load(args.edit_state)
    if "d_xyz" not in edit_state:
        raise KeyError(f"{args.edit_state} does not contain d_xyz")
    d_xyz = edit_state["d_xyz"].astype(np.float32)
    if d_xyz.shape != (gaussian_count, 3):
        raise ValueError(f"edit_state d_xyz shape {d_xyz.shape} does not match Gaussian count {gaussian_count}")

    moved_np = np.linalg.norm(d_xyz, axis=1) > args.affected_dxyz_threshold
    moved = torch.from_numpy(moved_np).to(device=editable_mask.device, dtype=torch.bool)
    affected_mask = editable_mask.bool() & moved
    affected_count = int(affected_mask.sum().item())
    if affected_count < args.min_affected_count:
        print(
            f"[DiffusionPrior] affected d_xyz mask selected only {affected_count} rows; "
            f"falling back to editable mask count={int(editable_mask.sum().item())}",
            flush=True,
        )
        return editable_mask.bool()
    print(
        f"[DiffusionPrior] affected_mask_from_dxyz rows={affected_count} "
        f"threshold={args.affected_dxyz_threshold}",
        flush=True,
    )
    return affected_mask


def freeze_and_build_optimizer(gaussians, editable_mask: torch.Tensor, args: argparse.Namespace) -> torch.optim.Optimizer:
    optimize_xyz = args.lr_xyz > 0
    optimize_rotation = args.lr_rotation > 0
    optimize_features = args.lr_feature > 0
    optimize_opacity = args.lr_opacity > 0
    optimize_scaling = args.lr_scaling > 0
    gaussians._xyz.requires_grad_(optimize_xyz)
    gaussians._rotation.requires_grad_(optimize_rotation)
    gaussians._features_dc.requires_grad_(optimize_features)
    gaussians._opacity.requires_grad_(optimize_opacity)
    gaussians._scaling.requires_grad_(optimize_scaling)
    if gaussians._features_rest.numel() > 0:
        gaussians._features_rest.requires_grad_(optimize_features)

    editable_mask = editable_mask.bool()
    params = []
    row_mask = editable_mask[:, None].float()
    if optimize_xyz:
        gaussians._xyz.register_hook(lambda grad: grad * row_mask)
        params.append({"params": [gaussians._xyz], "lr": args.lr_xyz, "name": "xyz"})
    if optimize_rotation:
        gaussians._rotation.register_hook(lambda grad: grad * row_mask)
        params.append({"params": [gaussians._rotation], "lr": args.lr_rotation, "name": "rotation"})
    if optimize_features:
        feature_mask = editable_mask[:, None, None].float()
        gaussians._features_dc.register_hook(lambda grad: grad * feature_mask)
        params.append({"params": [gaussians._features_dc], "lr": args.lr_feature, "name": "f_dc"})
    if optimize_opacity:
        gaussians._opacity.register_hook(lambda grad: grad * row_mask)
        params.append({"params": [gaussians._opacity], "lr": args.lr_opacity, "name": "opacity"})
    if optimize_scaling:
        gaussians._scaling.register_hook(lambda grad: grad * row_mask)
        params.append({"params": [gaussians._scaling], "lr": args.lr_scaling, "name": "scaling"})
    if optimize_features and gaussians._features_rest.numel() > 0:
        rest_mask = editable_mask[:, None, None].float()
        gaussians._features_rest.register_hook(lambda grad: grad * rest_mask)
        params.append({"params": [gaussians._features_rest], "lr": args.lr_feature / 20.0, "name": "f_rest"})
    if not params:
        raise ValueError("At least one optimization learning rate must be > 0")
    return torch.optim.Adam(params, lr=0.0, eps=1e-15)


def build_camera_masks(
    args: argparse.Namespace,
    cameras: List[MetadataCamera],
    gaussians,
    editable_mask: torch.Tensor,
    edit_mask: torch.Tensor,
) -> List[torch.Tensor]:
    if args.target_mask_mode == "all":
        return [torch.ones((1, camera.image_height, camera.image_width), dtype=torch.float32, device="cuda") for camera in cameras]

    if args.target_mask_mode == "projected":
        return [
            project_editable_gaussians_to_mask(camera, gaussians.get_xyz, editable_mask, args.view_mask_dilate)
            for camera in cameras
        ]

    camera_masks = []
    for camera in cameras:
        if camera.index == 0:
            camera_masks.append(edit_mask)
        else:
            camera_masks.append(project_editable_gaussians_to_mask(camera, gaussians.get_xyz, editable_mask, args.view_mask_dilate))
    return camera_masks


def box_blur_tensor(tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return tensor
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    image = tensor[None]
    image = F.pad(image, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(image, kernel_size=kernel_size, stride=1)[0]


def build_idu_target(rendered: torch.Tensor, sr_tensor: torch.Tensor, baseline: torch.Tensor, mask: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.target_mode == "direct":
        sr_target = sr_tensor
    elif args.target_mode == "highpass-residual":
        residual = sr_tensor - rendered
        residual = residual - box_blur_tensor(residual, args.highpass_kernel)
        if args.highpass_clip > 0:
            residual = residual.clamp(-args.highpass_clip, args.highpass_clip)
        sr_target = (rendered + args.sr_blend_alpha * residual).clamp(0, 1)
    else:
        raise ValueError(f"Unsupported target mode: {args.target_mode}")
    return mask * sr_target + (1.0 - mask) * baseline


@torch.no_grad()
def render_current(camera: MetadataCamera, gaussians, pipe, background: torch.Tensor) -> torch.Tensor:
    out = render(
        viewpoint_camera=camera,
        pc=gaussians,
        pipe=pipe,
        bg_color=background,
        d_xyz=0.0,
        d_rotation=0.0,
        d_scaling=0.0,
        d_opacity=0.0,
        d_color=0.0,
    )
    return out["render"].detach().clamp(0, 1)


def update_idu_target(
    iteration: int,
    camera: MetadataCamera,
    gaussians: GaussianModel,
    pipe,
    background: torch.Tensor,
    baseline_target: torch.Tensor | None,
    mask: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    rendered = render_current(camera, gaussians, pipe, background)
    input_path = Path(args.out_dir) / "idu" / f"iter_{iteration:05d}_render.png"
    sr_path = Path(args.out_dir) / "idu" / f"iter_{iteration:05d}_sr.png"
    save_tensor_image(rendered, input_path)
    torch.cuda.empty_cache()
    run_stable_sr(input_path, sr_path, args)
    torch.cuda.empty_cache()
    sr_tensor = load_rgb_tensor(sr_path, (camera.image_width, camera.image_height))
    sr_mean = float(sr_tensor.mean().item())
    sr_std = float(sr_tensor.std().item())
    if sr_mean < args.min_stable_sr_mean or sr_std < args.min_stable_sr_std:
        print(
            f"[DiffusionPrior] reject degenerate StableSR output at iter={iteration}: "
            f"mean={sr_mean:.6f} std={sr_std:.6f}; using renderer target",
            flush=True,
        )
        sr_tensor = rendered
    baseline = baseline_target if baseline_target is not None else rendered
    target = build_idu_target(rendered, sr_tensor, baseline, mask, args)
    if args.save_idu_targets:
        target_path = Path(args.out_dir) / "idu" / f"iter_{iteration:05d}_target.png"
        save_tensor_image(target, target_path)
    return target


def fine_tune(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("diffusion_prior_finetune requires CUDA because the 3DGS renderer is CUDA-only")
    ensure_runtime_imports()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(args.source_image) as source_image:
        source_width, source_height = source_image.size
    width = args.width or source_width
    height = args.height or source_height

    metadata = load_scene_metadata(args.scene_meta)
    cameras = build_camera_set(metadata, width, height, args.nearby_views)
    camera_targets = [load_optional_camera_target(camera, Path(args.source_image), (width, height)) for camera in cameras]
    edit_mask = load_mask_tensor(Path(args.edit_mask), (width, height))

    gaussians = GaussianModel(0, with_motion_mask=False)
    gaussians.load_ply(args.gs_path)
    editable_mask = load_editable_mask(args, gaussians, metadata)
    editable_mask = restrict_mask_to_affected_rows(args, editable_mask, int(gaussians.get_xyz.shape[0]))
    editable_count = int(editable_mask.sum().item())
    if editable_count == 0:
        raise ValueError("Editable mask selected zero Gaussians; cannot fine-tune appearance")
    initial_features_dc = gaussians._features_dc.detach().clone() if args.color_reg_weight > 0 else None
    initial_features_rest = gaussians._features_rest.detach().clone() if args.color_reg_weight > 0 and gaussians._features_rest.numel() > 0 else None

    optimizer = freeze_and_build_optimizer(gaussians, editable_mask, args)
    camera_masks = build_camera_masks(args, cameras, gaussians, editable_mask, edit_mask)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    source_anchor_target = None
    if args.source_anchor_weight > 0:
        source_anchor_target = render_current(cameras[0], gaussians, pipe, background)
    print(
        f"[DiffusionPrior] target_mode={args.target_mode} sr_blend_alpha={args.sr_blend_alpha} "
        f"color_reg_weight={args.color_reg_weight} source_anchor_weight={args.source_anchor_weight} "
        f"editable={editable_count}",
        flush=True,
    )

    fixed_camera_index = args.fixed_camera_index
    if fixed_camera_index is not None:
        if fixed_camera_index < 0 or fixed_camera_index >= len(cameras):
            raise ValueError(f"--fixed-camera-index {fixed_camera_index} is out of range for {len(cameras)} cameras")
        active_camera = cameras[fixed_camera_index]
    else:
        active_camera = cameras[0]
    active_index = cameras.index(active_camera)
    target = update_idu_target(
        0,
        active_camera,
        gaussians,
        pipe,
        background,
        camera_targets[active_index],
        camera_masks[active_index],
        args,
    )

    for iteration in range(1, args.iterations + 1):
        if fixed_camera_index is None and iteration > 1 and (iteration - 1) % args.idu_interval == 0:
            camera_idx = ((iteration - 1) // args.idu_interval) % len(cameras)
            active_camera = cameras[camera_idx]
            target = update_idu_target(
                iteration,
                active_camera,
                gaussians,
                pipe,
                background,
                camera_targets[camera_idx],
                camera_masks[camera_idx],
                args,
            )

        out = render(
            viewpoint_camera=active_camera,
            pc=gaussians,
            pipe=pipe,
            bg_color=background,
            d_xyz=0.0,
            d_rotation=0.0,
            d_scaling=0.0,
            d_opacity=0.0,
            d_color=0.0,
        )
        image = out["render"].clamp(0, 1)
        loss_l1 = l1_loss(image, target)
        loss_ssim = 1.0 - ssim(image[None], target[None])
        loss = (1.0 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim
        loss_color_reg = torch.tensor(0.0, device=image.device)
        if args.color_reg_weight > 0:
            feature_mask = editable_mask[:, None, None].float()
            loss_color_reg = ((gaussians._features_dc - initial_features_dc).square() * feature_mask).sum() / feature_mask.sum().clamp_min(1.0)
            if initial_features_rest is not None:
                loss_color_reg = loss_color_reg + (
                    (gaussians._features_rest - initial_features_rest).square() * feature_mask
                ).sum() / feature_mask.sum().clamp_min(1.0)
            loss = loss + args.color_reg_weight * loss_color_reg

        loss_source_anchor = torch.tensor(0.0, device=image.device)
        if source_anchor_target is not None:
            if active_camera is cameras[0]:
                source_render = image
            else:
                source_out = render(
                    viewpoint_camera=cameras[0],
                    pc=gaussians,
                    pipe=pipe,
                    bg_color=background,
                    d_xyz=0.0,
                    d_rotation=0.0,
                    d_scaling=0.0,
                    d_opacity=0.0,
                    d_color=0.0,
                )
                source_render = source_out["render"].clamp(0, 1)
            source_mask = camera_masks[0]
            source_diff = (source_render - source_anchor_target).abs() * source_mask
            loss_source_anchor = source_diff.sum() / (source_mask.sum() * source_render.shape[0]).clamp_min(1.0)
            loss = loss + args.source_anchor_weight * loss_source_anchor

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iteration == 1 or iteration % args.log_interval == 0 or iteration == args.iterations:
            print(
                f"[DiffusionPrior] iter={iteration} loss={loss.item():.6f} "
                f"l1={loss_l1.item():.6f} ssim={loss_ssim.item():.6f} "
                f"color_reg={loss_color_reg.item():.6f} source_anchor={loss_source_anchor.item():.6f} "
                f"editable={editable_count}",
                flush=True,
            )
        if args.save_interval > 0 and iteration % args.save_interval == 0 and iteration != args.iterations:
            checkpoint_path = out_dir / f"{Path(args.gs_path).stem}_iter_{iteration:04d}.ply"
            gaussians.save_ply(str(checkpoint_path))
            print(f"[DiffusionPrior] checkpoint_ply={checkpoint_path}", flush=True)

    refined_path = out_dir / f"{Path(args.gs_path).stem}_diffusion_prior.ply"
    gaussians.save_ply(str(refined_path))
    return refined_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARAP-GS-style diffusion-prior fine-tuning for ObjectMorpher full-scene 3DGS")
    parser.add_argument("--gs-path", required=True, help="Edited full-scene 3DGS PLY")
    parser.add_argument("--source-image", required=True, help="Original RGB input image")
    parser.add_argument("--edit-mask", required=True, help="SAM object mask used to select editable Gaussians")
    parser.add_argument("--scene-meta", required=True, help="SHARP or pseudo-multiview scene_meta.json")
    parser.add_argument("--edit-state", default=None, help="Optional edit_state.npz saved by edit_gui.py")
    parser.add_argument("--out-dir", required=True, help="Output directory for refined PLY and IDU images")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--idu-interval", type=int, default=10)
    parser.add_argument("--lambda-dssim", type=float, default=0.2)
    parser.add_argument("--target-mode", choices=("direct", "highpass-residual"), default="direct")
    parser.add_argument("--sr-blend-alpha", type=float, default=0.1, help="Blend strength for --target-mode highpass-residual")
    parser.add_argument("--highpass-kernel", type=int, default=31, help="Box blur kernel used to remove low-frequency SR tone shifts")
    parser.add_argument("--highpass-clip", type=float, default=0.25, help="Clamp high-pass residual magnitude before blending; <=0 disables")
    parser.add_argument("--color-reg-weight", type=float, default=0.0, help="L2 regularization to pre-IDU Gaussian color attributes")
    parser.add_argument("--source-anchor-weight", type=float, default=0.0, help="L1 anchor to the pre-IDU source-view render")
    parser.add_argument("--affected-mask-from-dxyz", action="store_true", help="Restrict editable rows to Gaussians with edit_state d_xyz above threshold")
    parser.add_argument("--affected-dxyz-threshold", type=float, default=0.001)
    parser.add_argument("--min-affected-count", type=int, default=128)
    parser.add_argument("--nearby-views", type=int, default=2, help="Small synthetic camera jitters near the source view; ignored when scene_meta.json has views")
    parser.add_argument("--fixed-camera-index", type=int, default=None, help="Use one fixed camera from the source+nearby set")
    parser.add_argument("--view-mask-dilate", type=int, default=31, help="Pixel dilation for projected editable-Gaussian masks on non-source views")
    parser.add_argument(
        "--optimization-scope",
        choices=("object", "all"),
        default="object",
        help="Rows that receive gradients. `object` uses edit_state/projection; `all` updates the full scene.",
    )
    parser.add_argument(
        "--target-mask-mode",
        choices=("edit", "projected", "all"),
        default="edit",
        help="2D mask used to merge StableSR targets with baseline targets.",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--stable-sr-root", default="")
    parser.add_argument("--stable-sr-env", default="stablesr")
    parser.add_argument("--stable-sr-command", default="", help="Command template with {input}, {output}, and optional {root}")
    parser.add_argument("--stable-sr-worker-command", default="", help="Persistent StableSR JSONL worker command")
    parser.add_argument("--reuse-stable-sr-output", action="store_true", help="Reuse an existing IDU *_sr.png instead of running StableSR again")
    parser.add_argument("--stable-sr-stub", action="store_true", help="Copy rendered images instead of running StableSR")
    parser.add_argument("--save-idu-targets", action="store_true", help="Save merged IDU target images for debugging")
    parser.add_argument("--min-stable-sr-mean", type=float, default=0.02, help="Reject StableSR outputs darker than this image mean")
    parser.add_argument("--min-stable-sr-std", type=float, default=0.01, help="Reject near-constant StableSR outputs")
    parser.add_argument("--lr-xyz", type=float, default=0.0)
    parser.add_argument("--lr-rotation", type=float, default=0.0)
    parser.add_argument("--lr-feature", type=float, default=0.0001)
    parser.add_argument("--lr-opacity", type=float, default=0.0)
    parser.add_argument("--lr-scaling", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=0, help="Save intermediate refined PLY checkpoints every N iterations")
    parser.add_argument("--white-background", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    cli_args = parse_args()
    refined = fine_tune(cli_args)
    print(f"REFINED_PLY={refined}")
