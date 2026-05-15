"""Compose a foreground (object) PLY with a background PLY into a single 3DGS.

Both inputs are INRIA-format gaussian PLYs. We optionally translate/scale
each cloud independently, then concatenate the per-row attributes and
write a single combined PLY. No densification or re-fitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def load_vertices(path: Path) -> np.ndarray:
    return PlyData.read(str(path))["vertex"].data


def transform_xyz(v: np.ndarray, scale: float, translate: tuple[float, float, float]) -> np.ndarray:
    """Apply uniform scale around origin + translation. Also scales gaussian sizes (log-scale: add log(scale))."""
    out = v.copy()
    out["x"] = v["x"] * scale + translate[0]
    out["y"] = v["y"] * scale + translate[1]
    out["z"] = v["z"] * scale + translate[2]
    log_s = float(np.log(max(scale, 1e-9)))
    for k in ("scale_0", "scale_1", "scale_2"):
        if k in out.dtype.names:
            out[k] = v[k] + log_s
    return out


def normalize_dtypes(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """If the two PLYs have different fields (different SH degrees), zero-pad the missing ones."""
    fa = set(a.dtype.names); fb = set(b.dtype.names)
    common = list(a.dtype.names)
    extra_b = [n for n in b.dtype.names if n not in fa]
    extra_a = [n for n in a.dtype.names if n not in fb]
    target_names = list(a.dtype.names)
    for n in b.dtype.names:
        if n not in fa:
            target_names.append(n)
    target_dtype = [(n, "f4") for n in target_names]

    def pad(src: np.ndarray) -> np.ndarray:
        out = np.empty(len(src), dtype=target_dtype)
        for n in target_names:
            if n in src.dtype.names:
                out[n] = src[n]
            else:
                out[n] = 0.0
        return out

    return pad(a), pad(b)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Concatenate foreground + background INRIA-format 3DGS PLYs.")
    p.add_argument("--object-ply", required=True, type=Path, help="Foreground PLY (e.g. edit_0.ply)")
    p.add_argument("--background-ply", required=True, type=Path, help="Background PLY (e.g. bg.ply)")
    p.add_argument("--out-ply", required=True, type=Path)
    p.add_argument("--bg-scale", type=float, default=2.5, help="Uniform scale applied to background gaussians.")
    p.add_argument("--bg-translate", default="0,0,1.0", help="Translation 'x,y,z' applied to background after scaling.")
    p.add_argument("--object-scale", type=float, default=1.0)
    p.add_argument("--object-translate", default="0,0,0")
    return p


def parse_xyz(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 'x,y,z', got {s}")
    return parts[0], parts[1], parts[2]


def main() -> None:
    args = build_parser().parse_args()
    obj = load_vertices(args.object_ply)
    bg = load_vertices(args.background_ply)
    print(f"[load] object  {args.object_ply.name}: N={len(obj)}, fields={len(obj.dtype.names)}")
    print(f"[load] bg      {args.background_ply.name}: N={len(bg)}, fields={len(bg.dtype.names)}")

    obj = transform_xyz(obj, args.object_scale, parse_xyz(args.object_translate))
    bg = transform_xyz(bg, args.bg_scale, parse_xyz(args.bg_translate))

    obj_n, bg_n = normalize_dtypes(obj, bg)
    merged = np.concatenate([obj_n, bg_n])
    print(f"[merge] total N = {len(merged)} (obj={len(obj_n)} + bg={len(bg_n)})")

    args.out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(merged, "vertex")]).write(str(args.out_ply))
    print(f"[save] {args.out_ply}")


if __name__ == "__main__":
    main()
