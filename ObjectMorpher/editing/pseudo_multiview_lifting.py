import argparse
from pathlib import Path

from editing.utils.pseudo_multiview_utils import prepare_pseudo_multiview_lifting


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import or run ExScene/CAT3D-style pseudo-multiview full-scene 3DGS lifting")
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pseudo-mv-command", default="", help="Generator command template with {input}, {output_dir}, {scene_meta}, and optional {ply}")
    parser.add_argument("--reconstruction-command", default="", help="3DGS reconstruction command template if the generator only creates views/cameras")
    parser.add_argument("--import-dir", default="", help="Existing pseudo multiview output directory")
    parser.add_argument("--ply", default="", help="Explicit PLY path")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    result = prepare_pseudo_multiview_lifting(
        image_path=Path(args.source_image),
        output_root=Path(args.out_dir),
        generator_command=args.pseudo_mv_command,
        reconstruction_command=args.reconstruction_command,
        import_dir=args.import_dir or None,
        explicit_ply=args.ply or None,
    )
    print(f"PSEUDO_MV_DIR={result['directory']}", flush=True)
    print(f"PSEUDO_MV_PLY={result['ply']}", flush=True)
    print(f"PSEUDO_MV_SCENE_META={result['scene_meta']}", flush=True)


if __name__ == "__main__":
    main()
