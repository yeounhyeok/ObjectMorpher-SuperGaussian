import argparse
import json
from pathlib import Path

from editing.utils.layered_scene_utils import merge_layered_gaussians


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge an inpainted-background full-scene 3DGS with editable foreground Gaussians from a deformed scene."
    )
    parser.add_argument("--background-gs", required=True, help="PLY lifted from the object-removed/inpainted background image")
    parser.add_argument("--foreground-gs", required=True, help="Edited full-scene PLY whose editable rows contain the deformed object")
    parser.add_argument("--edit-state", required=True, help="NPZ containing editable_mask for the foreground PLY")
    parser.add_argument("--out-ply", required=True)
    parser.add_argument("--out-edit-state", default="", help="Defaults to <out-ply>.edit_state.npz")
    parser.add_argument("--out-meta", default="", help="Optional sidecar metadata JSON for the layered scene")
    parser.add_argument("--background-scene-meta", default="", help="Camera metadata from the background SHARP lifting")
    parser.add_argument("--source-image", default="", help="Original source image for traceability")
    parser.add_argument("--mask-key", default="editable_mask")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    summary = merge_layered_gaussians(
        background_ply=Path(args.background_gs),
        foreground_ply=Path(args.foreground_gs),
        edit_state=Path(args.edit_state),
        output_ply=Path(args.out_ply),
        output_edit_state=Path(args.out_edit_state) if args.out_edit_state else None,
        output_meta=Path(args.out_meta) if args.out_meta else None,
        background_scene_meta=Path(args.background_scene_meta) if args.background_scene_meta else None,
        source_image=Path(args.source_image) if args.source_image else None,
        mask_key=args.mask_key,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
