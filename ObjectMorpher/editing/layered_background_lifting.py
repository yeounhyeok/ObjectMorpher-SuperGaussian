import argparse
import json
from pathlib import Path

from editing.pixelhacker_inpaint import _cv2_fallback
from editing.utils.layered_scene_utils import merge_layered_gaussians
from editing.utils.scene_lifting_utils import extract_sharp_scene_metadata
from editing.workflow_gui import PixelHackerRunner, REPO_ROOT, SharpRunner


def _run_inpaint(args: argparse.Namespace, out_dir: Path) -> Path:
    if args.background_image:
        return Path(args.background_image)

    output = out_dir / "background_inpaint.png"
    weight = Path(args.pixelhacker_weight)
    if not weight.exists():
        if not args.fallback_cv2:
            raise FileNotFoundError(
                f"PixelHacker weight not found: {weight}. Provide --pixelhacker-weight, or add --fallback-cv2 for a smoke-test background."
            )
        _cv2_fallback(Path(args.source_image), Path(args.edit_mask), output, args.fallback_radius)
        return output

    runner = PixelHackerRunner(
        Path(args.pixelhacker_config),
        weight,
        device=args.pixelhacker_device,
        release_after_run=True,
    )
    runner.run(
        Path(args.source_image),
        Path(args.edit_mask),
        output,
        num_steps=args.pixelhacker_steps,
        strength=args.pixelhacker_strength,
        guidance_scale=args.pixelhacker_guidance_scale,
        noise_offset=args.pixelhacker_noise_offset,
        paste=True,
    )
    return output


def _run_background_lift(args: argparse.Namespace, background_image: Path, out_dir: Path) -> tuple[Path, Path]:
    if args.background_gs:
        background_gs = Path(args.background_gs)
        if args.background_scene_meta:
            return background_gs, Path(args.background_scene_meta)
        meta_path = out_dir / "background_scene_meta.json"
        extract_sharp_scene_metadata(background_gs, meta_path, background_image)
        return background_gs, meta_path

    sharp_runner = SharpRunner(args.sharp_bin, checkpoint_path=args.sharp_checkpoint or None, device=args.sharp_device)
    result = sharp_runner.run(background_image, out_dir / "sharp_background")
    return Path(result["ply"]), Path(result["scene_meta"])


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a layered no-hole scene: PixelHacker inpainted background -> SHARP 3DGS -> merge with deformed object splats."
    )
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--edit-mask", required=True)
    parser.add_argument("--foreground-gs", required=True, help="Deformed full-scene/object PLY")
    parser.add_argument("--edit-state", required=True, help="Foreground edit state with editable_mask")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-name", default="layered_merged")

    parser.add_argument("--background-image", default="", help="Use an existing inpainted background image instead of running PixelHacker")
    parser.add_argument("--background-gs", default="", help="Use an existing background 3DGS PLY instead of running SHARP")
    parser.add_argument("--background-scene-meta", default="", help="Camera metadata for --background-gs")

    parser.add_argument("--pixelhacker-config", default=str(REPO_ROOT / "inpainting/config/PixelHacker_sdvae_f8d4.yaml"))
    parser.add_argument("--pixelhacker-weight", default=str(REPO_ROOT / "inpainting/weight/ft_places2/diffusion_pytorch_model.bin"))
    parser.add_argument("--pixelhacker-device", default="cuda")
    parser.add_argument("--pixelhacker-steps", type=int, default=20)
    parser.add_argument("--pixelhacker-strength", type=float, default=0.999)
    parser.add_argument("--pixelhacker-guidance-scale", type=float, default=4.5)
    parser.add_argument("--pixelhacker-noise-offset", type=float, default=0.0357)
    parser.add_argument("--fallback-cv2", action="store_true", help="Smoke-test fallback when PixelHacker weights are not available")
    parser.add_argument("--fallback-radius", type=float, default=7.0)

    parser.add_argument("--sharp-bin", default="sharp")
    parser.add_argument("--sharp-checkpoint", default="")
    parser.add_argument("--sharp-device", default="default")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    background_image = _run_inpaint(args, out_dir)
    background_gs, background_meta = _run_background_lift(args, background_image, out_dir)

    out_ply = out_dir / f"{args.out_name}.ply"
    out_edit_state = out_dir / f"{args.out_name}_edit_state.npz"
    out_meta = out_dir / f"{args.out_name}_scene_meta.json"
    summary = merge_layered_gaussians(
        background_ply=background_gs,
        foreground_ply=Path(args.foreground_gs),
        edit_state=Path(args.edit_state),
        output_ply=out_ply,
        output_edit_state=out_edit_state,
        output_meta=out_meta,
        background_scene_meta=background_meta,
        source_image=Path(args.source_image),
    )
    summary["background_image"] = str(background_image)
    summary["background_scene_meta"] = str(background_meta)
    out_meta.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"BACKGROUND_IMAGE={background_image}", flush=True)
    print(f"BACKGROUND_GS={background_gs}", flush=True)
    print(f"LAYERED_PLY={out_ply}", flush=True)
    print(f"LAYERED_EDIT_STATE={out_edit_state}", flush=True)
    print(f"LAYERED_SCENE_META={out_meta}", flush=True)


if __name__ == "__main__":
    main()
