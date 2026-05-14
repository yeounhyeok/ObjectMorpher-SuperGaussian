import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from editing.workflow_gui import PixelHackerRunner, REPO_ROOT


def _cv2_fallback(source_image: Path, mask_path: Path, output_path: Path, radius: float) -> None:
    image = np.asarray(Image.open(source_image).convert("RGB"))
    mask = np.asarray(Image.open(mask_path).convert("L"))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, (mask > 127).astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)).save(output_path)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inpaint an object mask with PixelHacker for layered-background 3DGS lifting")
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "inpainting/config/PixelHacker_sdvae_f8d4.yaml"))
    parser.add_argument("--weight", default=str(REPO_ROOT / "inpainting/weight/ft_places2/diffusion_pytorch_model.bin"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.999)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--noise-offset", type=float, default=0.0357)
    parser.add_argument("--no-paste", action="store_true", help="Do not paste the inpainted area back over the original image")
    parser.add_argument(
        "--fallback-cv2",
        action="store_true",
        help="Use OpenCV Telea inpainting if PixelHacker weights are unavailable. This is for smoke tests only.",
    )
    parser.add_argument("--fallback-radius", type=float, default=7.0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    source_image = Path(args.source_image)
    mask = Path(args.mask)
    output = Path(args.output)
    config = Path(args.config)
    weight = Path(args.weight)

    if not weight.exists():
        if not args.fallback_cv2:
            raise FileNotFoundError(
                f"PixelHacker weight not found: {weight}. Provide --weight, or use --fallback-cv2 for a non-PixelHacker smoke test."
            )
        _cv2_fallback(source_image, mask, output, args.fallback_radius)
        print(f"INPAINT_OUTPUT={output}", flush=True)
        return

    runner = PixelHackerRunner(config, weight, device=args.device, release_after_run=True)
    runner.run(
        source_image,
        mask,
        output,
        num_steps=args.num_steps,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        noise_offset=args.noise_offset,
        paste=not args.no_paste,
    )
    print(f"INPAINT_OUTPUT={output}", flush=True)


if __name__ == "__main__":
    main()
