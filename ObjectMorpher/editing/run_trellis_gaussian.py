import argparse
import json
import os
import sys
import types
from pathlib import Path

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
TRELLIS_DIR = REPO_ROOT / "reconstruct_from_2d"
if str(TRELLIS_DIR) not in sys.path:
    sys.path.insert(0, str(TRELLIS_DIR))


def _stub_kaolin_for_gaussian_only() -> None:
    """Let TRELLIS import mesh modules while running gaussian-only decoding.

    The mesh path imports `kaolin.utils.testing.check_tensor`, but gaussian-only
    inference never calls FlexiCubes. This keeps TRELLIS usable in environments
    where a matching kaolin wheel is unavailable.
    """
    if "kaolin.utils.testing" in sys.modules:
        return
    kaolin = types.ModuleType("kaolin")
    utils = types.ModuleType("kaolin.utils")
    testing = types.ModuleType("kaolin.utils.testing")
    testing.check_tensor = lambda *args, **kwargs: True
    utils.testing = testing
    kaolin.utils = utils
    sys.modules["kaolin"] = kaolin
    sys.modules["kaolin.utils"] = utils
    sys.modules["kaolin.utils.testing"] = testing


def _load_gaussian_only_pipeline(model: str):
    from huggingface_hub import hf_hub_download
    from trellis import models
    from trellis.pipelines import samplers, TrellisImageTo3DPipeline

    local_config = Path(model) / "pipeline.json"
    if local_config.exists():
        config_file = local_config
    else:
        config_file = Path(hf_hub_download(model, "pipeline.json"))

    args = json.loads(config_file.read_text(encoding="utf-8"))["args"]
    required = [
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "slat_decoder_gs",
        "slat_flow_model",
    ]
    loaded_models = {}
    for key in required:
        loaded_models[key] = models.from_pretrained(f"{model}/{args['models'][key]}")

    pipeline = TrellisImageTo3DPipeline(
        models=loaded_models,
        sparse_structure_sampler=getattr(samplers, args["sparse_structure_sampler"]["name"])(
            **args["sparse_structure_sampler"]["args"]
        ),
        slat_sampler=getattr(samplers, args["slat_sampler"]["name"])(**args["slat_sampler"]["args"]),
        slat_normalization=args["slat_normalization"],
        image_cond_model=args["image_cond_model"],
    )
    pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"]["params"]
    pipeline.slat_sampler_params = args["slat_sampler"]["params"]
    return pipeline


def run(args: argparse.Namespace) -> Path:
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    _stub_kaolin_for_gaussian_only()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _load_gaussian_only_pipeline(args.model)
    if args.device.startswith("cuda"):
        pipeline.cuda()
    else:
        pipeline.to(torch.device(args.device))

    image = Image.open(args.image).convert("RGBA")
    sparse_params = {}
    slat_params = {}
    if args.sparse_steps > 0:
        sparse_params["steps"] = args.sparse_steps
    if args.slat_steps > 0:
        slat_params["steps"] = args.slat_steps

    outputs = pipeline.run(
        image,
        seed=args.seed,
        preprocess_image=not args.no_preprocess,
        formats=["gaussian"],
        sparse_structure_sampler_params=sparse_params,
        slat_sampler_params=slat_params,
    )
    gaussian = outputs["gaussian"][0]
    ply_path = output_dir / args.out_name
    gaussian.save_ply(str(ply_path))
    print(f"TRELLIS_GAUSSIAN_PLY={ply_path}", flush=True)
    return ply_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRELLIS image-to-3D and save only the Gaussian PLY output")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-name", default="model.ply")
    parser.add_argument("--model", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--sparse-steps", type=int, default=-1, help="Override sparse sampler steps; <=0 keeps model default")
    parser.add_argument("--slat-steps", type=int, default=-1, help="Override slat sampler steps; <=0 keeps model default")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
