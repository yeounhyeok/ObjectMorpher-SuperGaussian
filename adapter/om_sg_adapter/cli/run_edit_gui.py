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

    # Mark each container as a package via __path__ so dotted submodules resolve.
    pytorch3d = types.ModuleType("pytorch3d");                       pytorch3d.__path__ = []
    ops = types.ModuleType("pytorch3d.ops")
    loss = types.ModuleType("pytorch3d.loss");                       loss.__path__ = []
    mls = types.ModuleType("pytorch3d.loss.mesh_laplacian_smoothing")
    io_mod = types.ModuleType("pytorch3d.io")

    def knn_points(p1, p2, lengths1=None, lengths2=None, K=1, return_nn=False, **_):
        # p1: [B, N1, D], p2: [B, N2, D]. Returns (sq_dist, idx, None).
        dist_sq = torch.cdist(p1, p2) ** 2
        K_eff = min(K, p2.shape[1])
        values, indices = torch.topk(dist_sq, K_eff, dim=-1, largest=False)
        return values, indices, None

    def ball_query(p1, p2, lengths1=None, lengths2=None, K=500, radius=0.2, return_nn=False, **_):
        # p1: [B, N1, D], p2: [B, N2, D]. Returns (sq_dist [B,N1,K], idx [B,N1,K], nn|None).
        # For each query in p1, find up to K nearest neighbors in p2 within `radius`.
        # Points beyond radius get idx=-1 and sq_dist=0 (pytorch3d convention).
        dist_sq = torch.cdist(p1, p2) ** 2
        K_eff = min(K, p2.shape[1])
        values, indices = torch.topk(dist_sq, K_eff, dim=-1, largest=False)
        within = values <= (radius * radius)
        indices = torch.where(within, indices, torch.full_like(indices, -1))
        values = torch.where(within, values, torch.zeros_like(values))
        nn = None
        if return_nn:
            # Gather neighbor coordinates; out-of-radius slots get zero
            safe_idx = indices.clamp(min=0)
            B, N1, D = p1.shape
            nn = torch.gather(p2.unsqueeze(1).expand(B, N1, p2.shape[1], D), 2,
                              safe_idx.unsqueeze(-1).expand(-1, -1, -1, D))
            nn = torch.where(within.unsqueeze(-1), nn, torch.zeros_like(nn))
        return values, indices, nn

    def cot_laplacian(verts: "torch.Tensor", faces: "torch.Tensor", eps: float = 1e-12):
        """Cotangent Laplacian + per-vertex inv-area (pytorch3d-compatible).

        verts: [V, 3], faces: [F, 3].
        Returns sparse_coo L [V,V] and dense inv_areas [V, 1].
        """
        V = verts.shape[0]
        face_verts = verts[faces]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
        A = (v1 - v2).norm(dim=1)
        B = (v0 - v2).norm(dim=1)
        C = (v0 - v1).norm(dim=1)
        s = 0.5 * (A + B + C)
        area = (s * (s - A) * (s - B) * (s - C)).clamp(min=eps).sqrt()
        A2, B2, C2 = A * A, B * B, C * C
        cota = (B2 + C2 - A2) / area
        cotb = (A2 + C2 - B2) / area
        cotc = (A2 + B2 - C2) / area
        cot = torch.stack([cota, cotb, cotc], dim=1) / 4.0
        ii = faces[:, [1, 2, 0]]
        jj = faces[:, [2, 0, 1]]
        idx = torch.stack([ii.reshape(-1), jj.reshape(-1)], dim=0)
        L = torch.sparse_coo_tensor(idx, cot.reshape(-1), (V, V))
        L = L + L.t()
        face_idx = faces.reshape(-1)
        area3 = area.unsqueeze(1).expand(-1, 3).reshape(-1)
        inv_areas = torch.zeros(V, device=verts.device, dtype=verts.dtype)
        inv_areas.scatter_add_(0, face_idx, area3)
        nz = inv_areas > 0
        inv_areas[nz] = 1.0 / inv_areas[nz]
        return L, inv_areas.reshape(-1, 1)

    def load_ply(filename):
        """Minimal pytorch3d.io.load_ply equivalent; returns (verts, faces) tensors."""
        from plyfile import PlyData
        plydata = PlyData.read(str(filename))
        v = plydata["vertex"]
        verts = torch.tensor([[v["x"][i], v["y"][i], v["z"][i]] for i in range(len(v))],
                             dtype=torch.float32)
        if "face" in plydata:
            f = plydata["face"]
            faces_list = [list(face[0]) for face in f]
            faces = torch.tensor(faces_list, dtype=torch.int64)
        else:
            faces = torch.zeros((0, 3), dtype=torch.int64)
        return verts, faces

    ops.knn_points = knn_points
    ops.ball_query = ball_query
    mls.cot_laplacian = cot_laplacian
    io_mod.load_ply = load_ply

    pytorch3d.ops = ops
    pytorch3d.loss = loss
    pytorch3d.io = io_mod
    loss.mesh_laplacian_smoothing = mls

    sys.modules["pytorch3d"] = pytorch3d
    sys.modules["pytorch3d.ops"] = ops
    sys.modules["pytorch3d.loss"] = loss
    sys.modules["pytorch3d.loss.mesh_laplacian_smoothing"] = mls
    sys.modules["pytorch3d.io"] = io_mod


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
