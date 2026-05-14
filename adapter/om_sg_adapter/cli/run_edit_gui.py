"""Launch OM's editing/edit_gui.py with the small shims our environment needs.

We install:
  * pytorch3d.ops.knn_points  -> a `torch.cdist + topk` stand-in (the only
    pytorch3d symbol edit_gui actually calls), so we don't have to build
    pytorch3d on Blackwell.
  * SPCONV_ALGO=native / ATTN_BACKEND=sdpa env vars (defensive; edit_gui
    does not touch TRELLIS but its imports may transitively).

Then we hand over to editing/edit_gui.py via runpy. All other argv
is passed through to edit_gui's argparse.
"""

from __future__ import annotations

import os
import runpy
import sys
import types
from pathlib import Path

import torch


def _install_pytorch3d_shim() -> None:
    if "pytorch3d.ops" in sys.modules:
        return
    pytorch3d = types.ModuleType("pytorch3d")
    ops = types.ModuleType("pytorch3d.ops")

    def knn_points(p1, p2, lengths1=None, lengths2=None, K=1, return_nn=False, **_):
        # p1: [B, N1, D], p2: [B, N2, D].
        # OM uses the (sq-dist, idx) return; the third element is never read.
        dist_sq = torch.cdist(p1, p2) ** 2
        K_eff = min(K, p2.shape[1])
        values, indices = torch.topk(dist_sq, K_eff, dim=-1, largest=False)
        return values, indices, None

    ops.knn_points = knn_points
    pytorch3d.ops = ops
    sys.modules["pytorch3d"] = pytorch3d
    sys.modules["pytorch3d.ops"] = ops


def main() -> None:
    _install_pytorch3d_shim()
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")

    editing_dir = Path(__file__).resolve().parents[3] / "ObjectMorpher" / "editing"
    if str(editing_dir) not in sys.path:
        sys.path.insert(0, str(editing_dir))

    # Hand over to edit_gui.py with the surviving argv (after our own script name)
    sys.argv = ["edit_gui"] + sys.argv[1:]
    runpy.run_path(str(editing_dir / "edit_gui.py"), run_name="__main__")


if __name__ == "__main__":
    main()
