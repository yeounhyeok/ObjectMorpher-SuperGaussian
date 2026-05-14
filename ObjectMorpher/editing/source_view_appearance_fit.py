import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageFilter

from editing.diffusion_prior_finetune import (
    MetadataCamera,
    build_camera_set,
    ensure_runtime_imports,
    load_rgb_tensor,
    load_scene_metadata,
    save_tensor_image,
)
import editing.diffusion_prior_finetune as ft
from editing.utils.loss_utils import ssim


def load_dilated_mask(path: Path, size: tuple[int, int], dilation: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    if dilation > 1:
        kernel = int(dilation)
        if kernel % 2 == 0:
            kernel += 1
        image = image.filter(ImageFilter.MaxFilter(kernel))
    array = (np.asarray(image, dtype=np.float32) > 127).astype(np.float32)
    return torch.from_numpy(array)[None].cuda()


def load_editable_mask(path: Path, gaussian_count: int) -> torch.Tensor:
    edit_state = np.load(path)
    if "editable_mask" not in edit_state:
        raise KeyError(f"{path} does not contain editable_mask")
    editable_mask = torch.from_numpy(edit_state["editable_mask"].astype(np.bool_)).cuda()
    if editable_mask.shape[0] != gaussian_count:
        raise ValueError(
            f"editable_mask length {editable_mask.shape[0]} does not match Gaussian count {gaussian_count}"
        )
    if int(editable_mask.sum().item()) == 0:
        raise ValueError("editable_mask selects zero Gaussians")
    return editable_mask


def choose_source_camera(cameras: list[MetadataCamera]) -> MetadataCamera:
    if not cameras:
        raise ValueError("scene metadata produced zero cameras")
    for camera in cameras:
        if (camera.name or "").lower() == "source":
            return camera
    return cameras[0]


def render_train(camera: MetadataCamera, gaussians, pipe, background: torch.Tensor) -> torch.Tensor:
    out = ft.render(
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
    return out["render"].clamp(0, 1)


def configure_feature_only_optimizer(gaussians, editable_mask: torch.Tensor, lr_feature: float) -> torch.optim.Optimizer:
    gaussians._xyz.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._features_dc.requires_grad_(True)
    if gaussians._features_rest.numel() > 0:
        gaussians._features_rest.requires_grad_(False)

    feature_mask = editable_mask.bool()[:, None, None].float()
    gaussians._features_dc.register_hook(lambda grad: grad * feature_mask)
    return torch.optim.Adam([{"params": [gaussians._features_dc], "lr": lr_feature, "name": "f_dc"}], lr=0.0, eps=1e-15)


@torch.no_grad()
def tensor_max_abs_delta(lhs: torch.Tensor, rhs: torch.Tensor, rows: torch.Tensor | None = None) -> float:
    if rows is not None:
        lhs = lhs[rows]
        rhs = rhs[rows]
    if lhs.numel() == 0:
        return 0.0
    return float((lhs - rhs).abs().max().item())


def fit_source_view_appearance(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("source_view_appearance_fit requires CUDA because the 3DGS renderer is CUDA-only")
    ensure_runtime_imports()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(args.source_image) as source_image:
        source_width, source_height = source_image.size
    width = args.width or source_width
    height = args.height or source_height
    size = (width, height)

    metadata = load_scene_metadata(args.scene_meta)
    camera = choose_source_camera(build_camera_set(metadata, width, height, nearby_views=0))
    target = load_rgb_tensor(Path(args.source_image), size)
    mask = load_dilated_mask(Path(args.edit_mask), size, args.mask_dilate)

    gaussians = ft.GaussianModel(0, with_motion_mask=False)
    gaussians.load_ply(args.gs_path)
    editable_mask = load_editable_mask(Path(args.edit_state), gaussians.get_xyz.shape[0])

    initial_xyz = gaussians._xyz.detach().clone()
    initial_rotation = gaussians._rotation.detach().clone()
    initial_scaling = gaussians._scaling.detach().clone()
    initial_opacity = gaussians._opacity.detach().clone()
    initial_features_dc = gaussians._features_dc.detach().clone()
    initial_features_rest = gaussians._features_rest.detach().clone()

    optimizer = configure_feature_only_optimizer(gaussians, editable_mask, args.lr_feature)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    with torch.no_grad():
        before = render_train(camera, gaussians, pipe, background)
        save_tensor_image(before, out_dir / "before.png")
        save_tensor_image(target, out_dir / "target.png")
        save_tensor_image(mask.repeat(3, 1, 1), out_dir / "mask.png")

    mask_denominator = mask.sum().clamp_min(1.0) * 3.0
    for iteration in range(1, args.iterations + 1):
        image = render_train(camera, gaussians, pipe, background)
        masked_l1 = ((image - target).abs() * mask).sum() / mask_denominator
        masked_image = mask * image + (1.0 - mask) * target
        loss_ssim = 1.0 - ssim(masked_image[None], target[None])
        color_reg = torch.mean((gaussians._features_dc[editable_mask] - initial_features_dc[editable_mask]) ** 2)
        loss = args.lambda_l1 * masked_l1 + args.lambda_ssim * loss_ssim + args.lambda_color_reg * color_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iteration == 1 or iteration % args.log_interval == 0 or iteration == args.iterations:
            print(
                f"[SourceAppearanceFit] iter={iteration} loss={loss.item():.6f} "
                f"l1={masked_l1.item():.6f} ssim={loss_ssim.item():.6f} "
                f"color_reg={color_reg.item():.6f} editable={int(editable_mask.sum().item())}",
                flush=True,
            )

    with torch.no_grad():
        after = render_train(camera, gaussians, pipe, background)
        save_tensor_image(after, out_dir / "after.png")

    output_path = out_dir / args.out_name
    gaussians.save_ply(str(output_path))

    background_rows = ~editable_mask.bool()
    summary = {
        "input_ply": str(args.gs_path),
        "output_ply": str(output_path),
        "source_image": str(args.source_image),
        "edit_mask": str(args.edit_mask),
        "scene_meta": str(args.scene_meta),
        "edit_state": str(args.edit_state),
        "iterations": args.iterations,
        "lr_feature": args.lr_feature,
        "editable_gaussians": int(editable_mask.sum().item()),
        "total_gaussians": int(editable_mask.numel()),
        "max_abs_delta": {
            "features_dc_editable": tensor_max_abs_delta(gaussians._features_dc, initial_features_dc, editable_mask),
            "features_dc_background": tensor_max_abs_delta(gaussians._features_dc, initial_features_dc, background_rows),
            "features_rest_all": tensor_max_abs_delta(gaussians._features_rest, initial_features_rest),
            "xyz_all": tensor_max_abs_delta(gaussians._xyz, initial_xyz),
            "rotation_all": tensor_max_abs_delta(gaussians._rotation, initial_rotation),
            "scaling_all": tensor_max_abs_delta(gaussians._scaling, initial_scaling),
            "opacity_all": tensor_max_abs_delta(gaussians._opacity, initial_opacity),
        },
    }
    (out_dir / "appearance_fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bake source-view object appearance onto layered TRELLIS 3DGS geometry")
    parser.add_argument("--gs-path", required=True, help="Layered TRELLIS+background 3DGS PLY")
    parser.add_argument("--source-image", required=True, help="Original RGB image used as the object appearance target")
    parser.add_argument("--edit-mask", required=True, help="SAM object mask in source-view image space")
    parser.add_argument("--scene-meta", required=True, help="Scene camera metadata JSON")
    parser.add_argument("--edit-state", required=True, help="Layered edit_state.npz containing editable_mask for object rows")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-name", default="appearance_fit.ply")
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--lr-feature", type=float, default=0.001)
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--lambda-ssim", type=float, default=0.1)
    parser.add_argument("--lambda-color-reg", type=float, default=0.01)
    parser.add_argument("--mask-dilate", type=int, default=9)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--white-background", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    baked = fit_source_view_appearance(parse_args())
    print(f"APPEARANCE_FIT_PLY={baked}", flush=True)
