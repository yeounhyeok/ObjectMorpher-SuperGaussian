import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw

from editing.diffusion_prior_finetune import (
    build_camera_set,
    ensure_runtime_imports,
    load_scene_metadata,
    render_current,
    save_tensor_image,
)
import editing.diffusion_prior_finetune as ft


def _assert_real_multiview(metadata: dict) -> None:
    views = metadata.get("views") or []
    if len(views) < 2:
        raise ValueError(
            "Real multi-view rendering requires scene metadata with at least two `views`. "
            "This scene_meta.json only describes a single SHARP/source camera, so nearby synthetic fallback is disabled."
        )


def _save_contact_sheet(paths: list[Path], output: Path, thumb_size: int = 320) -> None:
    if not paths:
        return
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        images.append((path, image))

    label_height = 30
    sheet = Image.new("RGB", (thumb_size * len(images), thumb_size + label_height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for idx, (path, image) in enumerate(images):
        x = idx * thumb_size
        sheet.paste(image, (x, 0))
        draw.text((x + 8, thumb_size + 8), path.stem, fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def render_view(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("render_gs_view requires CUDA because the 3DGS renderer is CUDA-only")
    ensure_runtime_imports()
    with Image.open(args.source_image) as image:
        width, height = image.size
    metadata = load_scene_metadata(args.scene_meta)
    if args.require_real_views:
        _assert_real_multiview(metadata)
    cameras = build_camera_set(metadata, width, height, args.nearby_views)

    gaussians = ft.GaussianModel(0, with_motion_mask=False)
    gaussians.load_ply(args.gs_path)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    if args.render_all_views:
        out_dir = Path(args.out_dir) if args.out_dir else Path(args.output).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        rendered_paths = []
        for index, camera in enumerate(cameras):
            name = camera.name or f"view_{index:03d}"
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
            output = out_dir / f"view_{index:02d}_{safe_name}.png"
            image = render_current(camera, gaussians, pipe, background)
            save_tensor_image(image, output)
            rendered_paths.append(output)
            print(f"VIEW_{index}={output}", flush=True)
        if args.contact_sheet:
            contact_path = Path(args.contact_sheet)
            _save_contact_sheet(rendered_paths, contact_path)
            print(f"CONTACT_SHEET={contact_path}", flush=True)
        print(f"RENDERED_VIEW_COUNT={len(rendered_paths)}", flush=True)
        return

    if args.camera_index < 0 or args.camera_index >= len(cameras):
        raise ValueError(f"--camera-index {args.camera_index} is out of range for {len(cameras)} cameras")
    image = render_current(cameras[args.camera_index], gaussians, pipe, background)
    output = Path(args.output)
    save_tensor_image(image, output)
    print(f"RENDER_OUTPUT={output}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one or all metadata cameras from a 3DGS PLY")
    parser.add_argument("--gs-path", required=True)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--scene-meta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nearby-views", type=int, default=2)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--render-all-views", action="store_true", help="Render every camera in scene_meta.views")
    parser.add_argument("--out-dir", default="", help="Directory used with --render-all-views")
    parser.add_argument("--contact-sheet", default="", help="Optional contact sheet path used with --render-all-views")
    parser.add_argument(
        "--require-real-views",
        action="store_true",
        help="Fail unless scene_meta.json contains at least two explicit views; disables single-view nearby fallback.",
    )
    parser.add_argument("--white-background", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    render_view(parse_args())
