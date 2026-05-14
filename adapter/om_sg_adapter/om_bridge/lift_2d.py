from __future__ import annotations

import os
import sys
from pathlib import Path


def _reconstruct_root(repo_root: Path) -> Path:
    return repo_root / "ObjectMorpher" / "reconstruct_from_2d"


def add_trellis_to_syspath(repo_root: Path) -> Path:
    rroot = _reconstruct_root(repo_root)
    path_str = str(rroot)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return rroot


def _install_kaolin_stub() -> None:
    """Stub out kaolin.utils.testing.check_tensor so flexicubes can import.
    The only kaolin symbol used in trellis is `check_tensor`, an assert-style helper.
    """
    import types
    if "kaolin" in sys.modules:
        return
    kaolin = types.ModuleType("kaolin")
    kaolin_utils = types.ModuleType("kaolin.utils")
    kaolin_testing = types.ModuleType("kaolin.utils.testing")
    kaolin_testing.check_tensor = lambda *a, **kw: None
    kaolin.utils = kaolin_utils
    kaolin_utils.testing = kaolin_testing
    sys.modules["kaolin"] = kaolin
    sys.modules["kaolin.utils"] = kaolin_utils
    sys.modules["kaolin.utils.testing"] = kaolin_testing


def lift_image_to_ply(
    repo_root: Path,
    image_path: Path,
    output_ply: Path,
    seed: int = 1,
    output_glb: Path | None = None,
    model_id: str = "microsoft/TRELLIS-image-large",
) -> None:
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    add_trellis_to_syspath(repo_root)
    _install_kaolin_stub()

    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline  # type: ignore

    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_id)
    pipeline.cuda()

    image = Image.open(str(image_path))
    # Restrict outputs to the Gaussian path so we skip the mesh decoder
    # (which uses windowed sparse attention and nvdiffrast / kaolin).
    formats = ["gaussian"] if output_glb is None else ["gaussian", "mesh"]
    outputs = pipeline.run(image, seed=seed, formats=formats)

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    outputs["gaussian"][0].save_ply(str(output_ply))

    if output_glb is not None:
        from trellis.utils import postprocessing_utils  # type: ignore  # requires nvdiffrast
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=0.95,
            texture_size=1024,
        )
        output_glb.parent.mkdir(parents=True, exist_ok=True)
        glb.export(str(output_glb))
