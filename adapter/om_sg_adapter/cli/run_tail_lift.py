"""CLI wrapper: programmatic ARAP tail lift on a TRELLIS-lifted coarse PLY.

Adopted from ObjectMorpher-modded's `editing/virtual_arap_tail_lift.py`.
This wrapper handles inputs that are convenient to feed from our adapter
(SAM crop + lifted PLY) and synthesises the scene_meta + binary mask
that the modded entrypoint needs.

Steps:
  1) Derive `edit_mask.png` from the SAM crop's alpha channel.
  2) Write `scene_meta.json` for the SAM crop's view (OpenCV camera).
  3) Invoke `virtual_arap_tail_lift.apply_tail_lift` with our defaults
     plus user-supplied overrides (tail position, drag, etc.).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from ..om_bridge.mask_utils import sam_crop_alpha_to_mask
from ..om_bridge.scene_meta import write_scene_meta


def _import_tail_lift(repo_root: Path):
    editing_dir = repo_root / "ObjectMorpher" / "editing"
    if str(editing_dir) not in sys.path:
        sys.path.insert(0, str(editing_dir))
    import virtual_arap_tail_lift as vatl  # type: ignore
    return vatl


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Programmatic ARAP tail-lift on a TRELLIS coarse PLY.")
    p.add_argument("--input-crop", required=True, type=Path, help="SAM RGBA crop (alpha = object)")
    p.add_argument("--input-ply", required=True, type=Path, help="TRELLIS coarse PLY")
    p.add_argument("--out-dir", required=True, type=Path, help="Where to write deformed.ply, mask, scene_meta, edit_state")
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))

    p.add_argument("--fovy-degrees", type=float, default=50.0, help="Source-view fovy in degrees")
    p.add_argument("--distance", type=float, default=2.0, help="Camera distance along +z (w2c translation)")

    p.add_argument("--tail-x", type=float, default=0.5, help="Tail target u (in mask bbox 0..1)")
    p.add_argument("--tail-y", type=float, default=0.85, help="Tail target v (in mask bbox 0..1)")
    p.add_argument("--tail-region-x", type=float, default=0.5, help="Tail region cutoff along x of bbox")
    p.add_argument("--tail-region-y", type=float, default=0.6, help="Tail region cutoff along y of bbox")
    p.add_argument("--anchor-region-x", type=float, default=0.5, help="Anchor region cutoff along x")
    p.add_argument("--anchor-region-y", type=float, default=0.45, help="Anchor region cutoff along y")
    p.add_argument("--drag-x", type=float, default=0.0, help="Pixel drag in x (positive = right)")
    p.add_argument("--drag-y", type=float, default=-85.0, help="Pixel drag in y (negative = up in image)")

    p.add_argument("--node-count", type=int, default=512)
    p.add_argument("--graph-k", type=int, default=12)
    p.add_argument("--skin-k", type=int, default=4)
    p.add_argument("--arap-iterations", type=int, default=6)
    p.add_argument("--tail-handles", type=int, default=10)
    p.add_argument("--anchor-count", type=int, default=110)
    p.add_argument("--min-opacity", type=float, default=0.05)
    p.add_argument("--mask-dilation", type=int, default=35)
    p.add_argument("--chunk-size", type=int, default=65536)
    p.add_argument("--seed", type=int, default=7)
    return p


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build edit mask from the SAM crop's alpha channel
    edit_mask_path = out_dir / "edit_mask.png"
    sam_crop_alpha_to_mask(args.input_crop, edit_mask_path)

    crop = Image.open(args.input_crop)
    image_size = (crop.size[0], crop.size[1])  # (W, H)

    # 2) Write scene_meta for that view
    scene_meta_path = out_dir / "scene_meta.json"
    write_scene_meta(scene_meta_path, image_size=image_size, fovy_degrees=args.fovy_degrees, distance=args.distance)

    # 3) Invoke virtual_arap_tail_lift.apply_tail_lift
    vatl = _import_tail_lift(repo_root)
    out_ply = out_dir / "deformed.ply"
    edit_state = out_dir / "edit_state.npz"
    out_union_mask = out_dir / "union_mask.png"

    inner = SimpleNamespace(
        gs_path=str(args.input_ply),
        edit_mask=str(edit_mask_path),
        scene_meta=str(scene_meta_path),
        out_ply=str(out_ply),
        edit_state=str(edit_state),
        out_mask=str(out_union_mask),
        node_count=args.node_count,
        graph_k=args.graph_k,
        skin_k=args.skin_k,
        arap_iterations=args.arap_iterations,
        tail_handles=args.tail_handles,
        anchor_count=args.anchor_count,
        tail_x=args.tail_x,
        tail_y=args.tail_y,
        tail_region_x=args.tail_region_x,
        tail_region_y=args.tail_region_y,
        anchor_region_x=args.anchor_region_x,
        anchor_region_y=args.anchor_region_y,
        drag_x=args.drag_x,
        drag_y=args.drag_y,
        min_opacity=args.min_opacity,
        mask_dilation=args.mask_dilation,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    vatl.apply_tail_lift(inner)

    print(f"DEFORMED_PLY={out_ply}")
    print(f"EDIT_STATE={edit_state}")
    print(f"SCENE_META={scene_meta_path}")
    print(f"EDIT_MASK={edit_mask_path}")
    print(f"UNION_MASK={out_union_mask}")


if __name__ == "__main__":
    main()
