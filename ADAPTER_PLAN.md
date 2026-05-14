# ObjectMorpher ↔ SuperGaussian Adapter / Orchestration Plan

## 0. Goals & Principles

- **Keep both upstream repos clean and pinnable.** `ObjectMorpher/` and `SuperGaussian/` are treated as third-party engines. They should remain submodule-style trees that can be re-synced from upstream with minimal merge pain.
- **Add a root-level adapter** (`adapter/`) that contains every line of integration code we author. Anything that needs to live *inside* an upstream tree is restricted to thin, isolated "hook" files that are easy to identify and re-apply after upstream pulls.
- **First use-case:** use SuperGaussian-style upsampled video-frame outputs as *pseudo-GT* targets to refine an ObjectMorpher edit (3DGS deformation + composition).
- **Longer-term:** strengthen the 2D→3D path in ObjectMorpher by piping SuperGaussian's video-prior-conditioned 3DGS fitting into the ObjectMorpher rasterization stack.

This document is a build-order plan, not yet code.

---

## 1. Repo Inspection Summary

### 1.1 Workspace layout

```
ObjectMorpher-SuperGaussian/
├── ObjectMorpher/                # CVPR 2026 ObjectMorpher (deformable 3DGS editing)
├── SuperGaussian/                # ECCV 2024 SuperGaussian (video-prior 3D super-res)
├── config/                       # claude notifier config (job orchestration only)
├── scripts/                      # claude job runner shells (not paper-related)
├── README.md                     # workspace-level README (notifier wiring)
└── .agent_runs/                  # background-job logs
```

`config/` and `scripts/` are unrelated to the research integration. They are pure ops glue for the background Claude jobs. Keep them, but do not let the adapter depend on them.

### 1.2 ObjectMorpher (Stage 1 = interactive 3D editing)

Key directories and entrypoints:

- `ObjectMorpher/preprocess/sam_processor.py` — SAM-based object segmentation from an input image. Writes masks + isolated object crops into `outputs/`.
- `ObjectMorpher/reconstruct_from_2d/app.py` — Gradio app driving TRELLIS image-to-3D, producing a Gaussian model PLY (`reconstruct_from_2d/gs_ply/gaussian_model.ply`). This is the **2D→3D lifting** path we want to strengthen later.
- `ObjectMorpher/editing/edit_gui.py` — Interactive ARAP deformation GUI over the 3DGS. Imports `gaussian_renderer.render` and `scene.GaussianModel`. Loads/edits a `.ply`.
- `ObjectMorpher/editing/gaussian_renderer/__init__.py` — Differentiable rasterization wrapper around `diff_gaussian_rasterization`. **This is the rendering seam** we will reuse to render comparable frames non-interactively.
- `ObjectMorpher/editing/scene/cameras.py` — `Camera` (R, T, FoVx, FoVy, image, …) and `MiniCam`. World-view + projection are computed via `utils/graphics_utils.getWorld2View2` and `getProjectionMatrix`. This is the **camera seam**.
- `ObjectMorpher/editing/scene/gaussian_model.py` — Standard 3DGS Gaussian container with `.load_ply / .save_ply / get_xyz / get_rotation / get_scaling / get_opacity / get_features`.
- `ObjectMorpher/editing/utils/arap_deform.py`, `lap_deform.py` — Geodesic / ARAP / Laplacian deformation logic.
- `ObjectMorpher/inpainting/` — PixelHacker-based background fill after object removal.
- `ObjectMorpher/outputs/` — Existing on-disk outputs split into `objects/`, `mask_*/`, `holes/`, `workflow/{sam,trellis,pixelhacker}/`. Use this as the convention for any new artifacts we add.

Environment: a single conda env (`spaceedit`) with PyTorch 2.5+cu121, `diff-gaussian-rasterization`, `pytorch3d`, `xformers`, `flash-attn`, `kaolin`, TRELLIS deps, `dearpygui`, etc. Heavy and CUDA-version-locked.

### 1.3 SuperGaussian

- `SuperGaussian/main_supergaussian.py` — multi-GPU orchestration script. Per-scene flow:
  1. Dump low-res 3DGS renderings at 64×64 into `step_0` files.
  2. Run video upsampling prior (RealBasicVSR / VideoGigaGAN / GigaGAN) → 256×256 frames in `step_1_upsampling/256x256/`.
  3. Run `fitting_with_3dgs` (the bundled `third_parties/gaussian-splatting/train.py`) to lift the upsampled frame sequence back into a high-res 3DGS.
  4. Outputs `step_2_fitting_with_3dgs/point_cloud/iteration_2000/{predicted/*.png, point_cloud.ply}`.
- `SuperGaussian/sg_utils/sg_helper.py` — three shell-invocation helpers: `run_bilinear_resampling`, `run_video_upsampling`, `fitting_with_3dgs`. Each `subprocess.run`'s a hard-coded conda env (`~/miniconda3/envs/realbasicvsr/bin/python`, `~/miniconda3/envs/super_gaussian_eccv24/bin/python`). **Three separate conda envs are mandatory**: `super_gaussian_eccv24`, `realbasicvsr`, `supergaussian_evaluation`.
- `SuperGaussian/dataset/mvimg_test_dataset.py` — strongly tied to MVImgNet's pickle layout (`cam_extrinsics.pkl`, `cam_intrinsics.pkl`, `xyz.pkl`, `rgb.pkl`) and to `data/mvimgnet_testset_500/...`. Produces per-scene batches with `images`, `extrinsics` (w2c, 4×4), `fxfycxcy`, `xyz`, `rgb`, `high_res_images`, plus filenames including novel-trajectory frames `traj_0_XXX.png`, `traj_1_XXX.png`.
- `SuperGaussian/third_parties/gaussian-splatting/` — INRIA 3DGS fork. Has its *own* `scene/cameras.py`, `gaussian_renderer/`, `train.py`. This is the lifting engine.

### 1.4 Shared concepts (the integration substrate)

Both repos already speak overlapping 3DGS dialects:

| Concept              | ObjectMorpher                                                  | SuperGaussian                                                                |
|----------------------|----------------------------------------------------------------|------------------------------------------------------------------------------|
| Gaussian container   | `editing/scene/gaussian_model.py` (PLY in/out)                 | `third_parties/gaussian-splatting/scene/gaussian_model.py`                   |
| Camera class         | `editing/scene/cameras.py` (R, T, FoVx/FoVy, w2c, full_proj)   | `third_parties/gaussian-splatting/scene/cameras.py`                          |
| Camera I/O           | COLMAP-style + JSON via `dataset_readers.py`                   | `transforms.json` (nerfstudio-ish: fl_x, fl_y, cx, cy, w, h, frames[c2w])    |
| Differentiable raster| `diff_gaussian_rasterization` (same submodule)                 | `diff_gaussian_rasterization` (same submodule)                               |
| World convention     | OpenCV-style w2c from `getWorld2View2(R, T)`                   | OpenCV w2c stored, then **flipped** to OpenGL c2w when emitting `transforms.json` (`c2w[:3, 1:3] *= -1`) |

The c2w/OpenGL flip in `main_supergaussian.py:118-119` is the most error-prone detail. The adapter must canonicalize this once.

---

## 2. Integration Points

### 2.1 What ObjectMorpher exposes that we need

- A function-callable **render-from-camera** path. Today rendering is reached through `edit_gui.py`'s GUI loop; for an experiment we want a non-interactive `render(gaussians, camera) → image, depth, alpha`. The underlying `editing/gaussian_renderer/render()` is already pure — the GUI just constructs `MiniCam`s and calls it. We can reuse it directly.
- A way to **load a deformed 3DGS** (a PLY produced by the editing GUI). `GaussianModel.load_ply()` does this.
- The **camera trajectory** used by the editing GUI (`OrbitCamera` in `editing/cam_utils.py`). We can sample it to produce a frame sequence + per-frame extrinsics/intrinsics.
- Optionally, the **canonical (pre-edit) GS** to enable both baseline + edited renders for ablation.

### 2.2 What SuperGaussian exposes that we need

- A way to **run only the upsampling step** on a directory of low-res frames (`run_video_upsampling(prior, gpu, in_dir, out_dir)` from `sg_helper.py`). That subprocess hops into the `realbasicvsr` conda env, so the adapter must shell out, not import.
- A way to **run only the fitting step** on (upsampled frames + transforms.json + initial PLY) → high-res PLY. `fitting_with_3dgs(...)` in `sg_helper.py`.
- The **`transforms.json` schema** at `main_supergaussian.py:100-126`. This is our contract.

### 2.3 The matching seam

We can pair the two engines without entering either GUI:

1. Render ObjectMorpher 3DGS along a chosen trajectory → frames + `transforms.json` in SuperGaussian's expected schema.
2. Optionally feed those frames through the SuperGaussian upsampling step to produce pseudo-GT high-res frames.
3. Use the upsampled frames as photometric loss targets to re-optimize ObjectMorpher Gaussians (via its own renderer, in its own env).

The "minimal optimization loop" only requires **ObjectMorpher's renderer + an L1/LPIPS image loss + ObjectMorpher's Gaussian parameters as nn.Parameters**. The SuperGaussian side is purely an offline frame producer for Phase 1.

---

## 3. Proposed Folder / File Structure

All new code lives under `adapter/` at the repo root. Both upstream trees stay untouched in Phase 1.

```
ObjectMorpher-SuperGaussian/
├── adapter/
│   ├── README.md                        # how to run, env matrix, where outputs land
│   ├── pyproject.toml                   # optional; install as `om_sg_adapter`
│   ├── om_sg_adapter/
│   │   ├── __init__.py
│   │   ├── config.py                    # dataclasses for AdapterConfig, PathsConfig, TrajectoryConfig, LossConfig
│   │   ├── paths.py                     # single source of truth for run dirs
│   │   ├── envs.py                      # conda env names + paths, sanity probes
│   │   │
│   │   ├── cameras/
│   │   │   ├── trajectory.py            # build c2w sequences (orbit, spiral, hand-spec'd)
│   │   │   ├── conventions.py           # OpenCV <-> OpenGL conversions in one place
│   │   │   └── transforms_io.py         # read/write SuperGaussian transforms.json
│   │   │
│   │   ├── om_bridge/                   # ObjectMorpher-side wrappers (no code inside ObjectMorpher/)
│   │   │   ├── load_gs.py               # load .ply via OM's GaussianModel
│   │   │   ├── render_frames.py         # call OM gaussian_renderer.render over a trajectory
│   │   │   ├── export_pseudo_dataset.py # write transforms.json + frames + initial point_cloud.ply
│   │   │   └── optimize_against.py      # minimal photometric optimization loop
│   │   │
│   │   ├── sg_bridge/                   # SuperGaussian-side wrappers (shell-out only)
│   │   │   ├── run_upsampling.py        # wraps sg_utils.sg_helper.run_video_upsampling
│   │   │   ├── run_fitting.py           # wraps sg_utils.sg_helper.fitting_with_3dgs
│   │   │   └── stage_inputs.py          # arrange the scene-dir layout SG expects
│   │   │
│   │   ├── pipelines/
│   │   │   ├── phase1_render_then_upsample.py    # OM -> frames -> SG upsample -> pseudo-GT set
│   │   │   ├── phase1_optimize_om_against_gt.py  # consume pseudo-GT, optimize OM 3DGS
│   │   │   └── phase2_joint_loop.py              # placeholder for iterative loop
│   │   │
│   │   ├── losses/
│   │   │   ├── photometric.py           # L1, MS-SSIM, LPIPS (optional)
│   │   │   └── geometric.py             # depth/alpha consistency (later)
│   │   │
│   │   └── cli/
│   │       ├── run_phase1.py            # one CLI entry per pipeline
│   │       └── run_optimize.py
│   │
│   ├── hooks/                           # the only code that may need to live in-tree
│   │   ├── objectmorpher/
│   │   │   └── renderer_entrypoint.py   # tiny file we copy/symlink into OM/editing/ if needed
│   │   └── supergaussian/
│   │       └── (empty for Phase 1)
│   │
│   └── scripts/
│       ├── run_phase1.sh                # activates objectmorpher env, then super_gaussian_eccv24, then back
│       └── env_check.sh
│
├── runs/                                # adapter-owned outputs (gitignored)
│   └── <run_name>/
│       ├── config.yaml
│       ├── om_baseline_frames/          # what OM renders before any optimization
│       │   ├── 0000.png ...
│       │   └── transforms.json
│       ├── sg_pseudo_gt/
│       │   ├── lr_64x64/                # input to SG upsampling
│       │   ├── upsampled_256x256/       # SG's video-prior output (the pseudo-GT)
│       │   └── transforms.json          # same poses, possibly rescaled
│       ├── optimization/
│       │   ├── checkpoints/iter_XXXX.ply
│       │   ├── renders/iter_XXXX/
│       │   └── loss_log.json
│       └── reports/
│           └── comparison_grid.png
└── ADAPTER_PLAN.md                      # this file
```

### File responsibilities (one-liners)

- `config.py` — Pydantic/dataclass schema for every knob; loaded from a YAML in `runs/<run_name>/config.yaml`.
- `cameras/conventions.py` — single canonical helper for the OpenCV↔OpenGL flip (`c2w[:3, 1:3] *= -1`). Both bridges import from here.
- `cameras/trajectory.py` — generate c2w stacks (orbit around object centroid, novel trajectory, jittered eval set). Returns numpy arrays + an FoV, never tensors.
- `cameras/transforms_io.py` — exact schema match with SuperGaussian's `transforms.json` (`fl_x, fl_y, cx, cy, w, h, k1..k4, p1, p2, frames=[{file_path, transform_matrix}]`).
- `om_bridge/render_frames.py` — adds `ObjectMorpher/editing` to `sys.path`, constructs `MiniCam`s, runs `render(...)` headlessly, writes PNGs.
- `om_bridge/optimize_against.py` — loads the same GS, exposes `_xyz / _rotation / _scaling / _features_dc / _opacity` as `nn.Parameter`, runs an Adam loop minimizing L1+LPIPS against the pseudo-GT frames at matching cameras.
- `sg_bridge/run_upsampling.py` — `subprocess.run` into the `realbasicvsr` env; mirrors `sg_helper.run_video_upsampling` but with adapter-friendly paths.
- `sg_bridge/run_fitting.py` — `subprocess.run` into `super_gaussian_eccv24`; uses `sg_helper.fitting_with_3dgs`. Only used in Phase 2 when we let SG re-lift.
- `pipelines/*.py` — wire the bridges together; no engine logic.
- `hooks/objectmorpher/renderer_entrypoint.py` — *only created if* `om_bridge/render_frames.py` cannot import `gaussian_renderer.render` cleanly from outside `editing/` (it uses relative `from scene.gaussian_model import ...`). In that case we copy a 10-line shim into `ObjectMorpher/editing/` exposing a clean function.

---

## 4. Execution Flow — First Experiment

End-to-end Phase 1 with concrete steps. Each step is a script invocation; the operator (or a shell wrapper) handles the env switches.

### Step A — Prepare an ObjectMorpher baseline edit

In the `objectmorpher` (a.k.a. `spaceedit`) conda env:

1. Run SAM segmentation + TRELLIS lift to get `reconstruct_from_2d/gs_ply/gaussian_model.ply` (or reuse the bundled `ObjectMorpher/sample.ply` / `mecha1.ply`).
2. Optionally run `editing/edit_gui.py` to perform an ARAP edit. Save the edited PLY to `runs/<run>/om_input/edited.ply`.

(Phase 1 does not actually need the edit step — even with the canonical PLY we can validate the loop. The edit is what makes the pseudo-GT *meaningful*.)

### Step B — Render ObjectMorpher baseline frames

```bash
conda activate objectmorpher
python -m adapter.om_sg_adapter.cli.run_phase1 \
       --config adapter/configs/phase1.yaml \
       --stage render_baseline
```

`om_bridge/render_frames.py`:

- Loads `runs/<run>/om_input/edited.ply` via `GaussianModel.load_ply`.
- Builds a c2w sequence from `cameras/trajectory.py` (default: 60-frame orbit around the GS centroid, FoV from config).
- Converts each c2w to a `MiniCam` (OpenCV `world_view_transform`, `full_proj_transform`).
- Calls `gaussian_renderer.render(...)` for each camera; saves `0000.png`, `0001.png`, … into `runs/<run>/om_baseline_frames/`.
- Writes `runs/<run>/om_baseline_frames/transforms.json` in SuperGaussian's schema. The c2w stored on disk uses **the same OpenGL flip** SG applies (`c2w[:3, 1:3] *= -1`) so SG's `train.py` consumes it unchanged.

### Step C — Stage SuperGaussian inputs

Still in `objectmorpher` env (no GPU needed for staging):

- Downsample the OM frames to 64×64 → `runs/<run>/sg_pseudo_gt/lr_64x64/`. (Or render OM at 64×64 directly; we'll do both and compare.)
- Initial point cloud: write a `surface_pcd_131072_seed_0.ply` from the loaded GS xyz/rgb using SG's `storePly` helper (copy the 12-line function into `sg_bridge/stage_inputs.py`).
- Symlink/copy `transforms.json`.

### Step D — Run SuperGaussian upsampling (pseudo-GT generation)

```bash
conda activate super_gaussian_eccv24    # only to launch; helper subprocess.run's into realbasicvsr
python -m adapter.om_sg_adapter.cli.run_phase1 \
       --config adapter/configs/phase1.yaml \
       --stage upsample
```

`sg_bridge/run_upsampling.py` shells out exactly like `sg_helper.run_video_upsampling('realbasicvsr', gpu_i, in_dir, out_dir)`. Output: 256×256 frames in `runs/<run>/sg_pseudo_gt/upsampled_256x256/`.

We do **not** run `fitting_with_3dgs` in Phase 1. The upsampled frames are the only thing we need as targets.

### Step E — Pair frames and camera metadata

`pipelines/phase1_optimize_om_against_gt.py`:

- Reads `transforms.json` once.
- For each frame `i`, makes a `(image_path, c2w_i, fx_i, fy_i, cx_i, cy_i, w, h)` tuple.
- Reconstructs an OM `MiniCam` from each tuple using `cameras/conventions.py` to invert the SG OpenGL flip back into the OM/OpenCV world-view convention.
- Frames and cameras are aligned by **index**, not filename, but filenames must remain unique and zero-padded. The adapter enforces this.

### Step F — Minimal optimization loop

```bash
conda activate objectmorpher
python -m adapter.om_sg_adapter.cli.run_optimize \
       --config adapter/configs/phase1.yaml
```

`om_bridge/optimize_against.py`:

```
load GS = GaussianModel.load_ply(edited.ply)
make optimizable subset of GS attributes (start: only _features_dc + _xyz; later: full set)
optimizer = Adam([...], lr_xyz=1e-4, lr_color=2.5e-3)
for it in range(N_iters):
    cam_i = random.choice(cameras)
    pred = render(cam_i, GS, pipe, bg_color, d_xyz=0, d_rotation=0, d_scaling=0)["render"]
    gt   = load_image(frame_paths[i_of(cam_i)])
    loss = lambda_l1 * L1(pred, gt) + lambda_lpips * LPIPS(pred, gt)
    loss.backward(); optimizer.step()
    every K iters: dump pred + GT side-by-side + ply checkpoint
```

Output: `runs/<run>/optimization/checkpoints/iter_2000.ply` plus a comparison grid.

### Step G — Evaluate / inspect

A tiny `pipelines/.../report.py` (or just `python -m adapter.om_sg_adapter.cli.report`) generates `runs/<run>/reports/comparison_grid.png`: baseline render | pseudo-GT | optimized render | residual.

---

## 5. Wrapper vs. Hook — what goes where

### Strictly wrapper (no upstream edits)

- All trajectory generation, transforms.json I/O, scene-dir staging, run management.
- All shelling out to SG's upsampling / fitting scripts.
- All Phase 1 optimization (uses OM's renderer as a library, no edits).
- All evaluation reports.
- Config, CLI, logging.

### Likely needs a light in-tree hook (kept under `adapter/hooks/`)

These are *small, named, copyable* shims; they get copied/symlinked into the upstream tree at install time so upstream pulls remain mostly clean:

1. **`ObjectMorpher/editing/__init__.py` or a `renderer_entrypoint.py`** — `gaussian_renderer/__init__.py` imports `from scene.gaussian_model import GaussianModel`, which only resolves when `ObjectMorpher/editing/` is on `sys.path`. Two acceptable options:
   - (Preferred) The adapter prepends `ObjectMorpher/editing` to `sys.path` at runtime. No in-tree change at all.
   - (Fallback) Drop a 10-line `renderer_entrypoint.py` into `ObjectMorpher/editing/` exposing `render_one(cam, gs, …)`. Track it in `adapter/hooks/objectmorpher/`.
2. **Headless `Camera` constructor** — `editing/scene/cameras.py:Camera.__init__` requires an `image` tensor. For pure rendering we don't need a GT image. `MiniCam` already supports this and is sufficient — **prefer MiniCam over Camera** in the bridge to avoid touching this file.
3. **SG `sg_helper.py` conda paths** — `run_video_upsampling` hard-codes `~/miniconda3/envs/realbasicvsr/bin/python` and `fitting_with_3dgs` hard-codes `~/miniconda3/envs/super_gaussian_eccv24/bin/python`. If those paths don't match the operator's install, this is a one-line in-tree edit. Better: copy `sg_helper.py` into `adapter/sg_bridge/sg_helper_patched.py` and import the patched version. Then *no* upstream edit is needed.

### Things to explicitly **not** modify upstream

- `SuperGaussian/main_supergaussian.py` — keep as-is; we never need to run the full MVImgNet pipeline. We only call the two helpers.
- `SuperGaussian/dataset/mvimg_test_dataset.py` — we are not using MVImgNet; we synthesize our own scene dir.
- `SuperGaussian/third_parties/gaussian-splatting/train.py` — the SG fitting step is invoked unmodified; transforms.json contract is enforced on the adapter side.
- `ObjectMorpher/editing/edit_gui.py` — GUI stays GUI; the adapter only depends on `gaussian_renderer`, `scene`, `utils` libraries inside `editing/`.

---

## 6. Risks

### 6.1 Environment separation

- Three SG conda envs plus one OM conda env. There is **no single env** that imports both repos. The adapter must orchestrate via subprocess at env boundaries. Implication: in-process Python integration is impossible for Phase 1; cross-env handoff is via files in `runs/<run>/`.
- `super_gaussian_eccv24` uses `torch==2.0.1+cu118`; OM uses `torch==2.5.0+cu121`. Forcing them to coexist will break `diff-gaussian-rasterization` builds in at least one of them.
- `RealBasicVSR` uses `torch==1.7.1+cu110` and `mmedit==0.15.0`. Extremely brittle. Do not attempt to upgrade it.
- Adapter implication: `envs.py` exposes `OM_ENV`, `SG_ENV`, `VSR_ENV` names and runs `conda run -n <env> python -m …` from a single launcher shell. CI/CLI activations must be explicit.

### 6.2 File I/O contracts

- `transforms.json` must match SG's schema exactly: `fl_x`, `fl_y` (not `fx`/`fy`), `cx`, `cy`, `w`, `h`, distortion zeros, and `frames[i].transform_matrix` is c2w in **OpenGL** convention (already flipped).
- The initial point cloud must be `surface_pcd_<N>_seed_0.ply` and exactly `N` points (default `131072`). Resampling is required if the OM GS has a different count. Use SG's `storePly` schema (xyz, normals=0, rgb 0-255). RGB **must be uint8**.
- Frame filenames must be zero-padded and sort-stable. SG's dataloader sorts by `cam_info.name`.
- Soft links: `sg_helper.fitting_with_3dgs` creates `os.symlink`s for `transforms.json`, `images`, `gt/high_res_images`, and the initial PLY. On Windows or on a filesystem without symlinks this will fail silently. Adapter should pre-create real files where symlinks are not supported.

### 6.3 Frame / camera alignment

- **OpenCV vs OpenGL.** ObjectMorpher renders with w2c in OpenCV convention (`getWorld2View2(R, T)`); SG stores c2w in OpenGL convention (`c2w[:3, 1:3] *= -1`). A single source of truth in `cameras/conventions.py` is non-negotiable. A 180° flip will silently produce upside-down or rotated pseudo-GT and the loss will look like it's converging while the geometry blows up.
- **Intrinsics scaling.** SG crops/resizes inputs to 256×256 and recomputes intrinsics through `autolab_core.CameraIntrinsics.resize/crop`. If we feed OM frames at a different resolution, `fl_x, fl_y, cx, cy` must be rescaled accordingly. The adapter renders OM frames at SG's target resolution to avoid this entirely.
- **FoV vs focal length.** OM cameras carry `FoVx, FoVy`; SG carries `fl_x, fl_y`. Conversion uses `fov2focal(fov, pixels) = pixels / (2 tan(fov/2))` (already in `utils/graphics_utils.py`). Centralize this in `cameras/conventions.py`.
- **Scene scale.** OM 3DGS coordinates come from TRELLIS (object-normalized space); SG expects MVImgNet-scale scenes with `nerf_normalization`-based bounding. The fitting step may diverge if the GS extent is unusual. For Phase 1 (no SG fitting), this risk is dormant; for Phase 2 it must be tested by a single-scene smoke run before any larger experiments.
- **No GT trajectory.** Unlike MVImgNet, our OM scenes have no recorded camera poses. We invent them. The "comparable" frames are only as comparable as our trajectory choice — pick a deterministic, dense orbit early and never change it across runs in a study.

### 6.4 Optimization-loop pitfalls

- Background color: OM renderer takes an explicit `bg_color`. Pseudo-GT frames from SG / RealBasicVSR have arbitrary backgrounds. Either composite both onto a fixed bg or use OM's alpha output and apply the loss only inside the alpha mask.
- LPIPS at 64×64 / 128×128 is unreliable. Run the photometric loss at ≥256×256.
- TRELLIS-lifted Gaussians often have very tight scales; optimizing `_scaling` without densification can quickly produce holes. Phase 1 should freeze topology (no clone/split/prune) and only optimize colors + small position deltas.

### 6.5 Repo hygiene

- `runs/`, the on-disk artifacts dir, must be added to `.gitignore` from day one. So must `adapter/.cache/`, `adapter/**/__pycache__/`. The repo already has a permissive `.gitignore` at the root (one line); extend it.
- `ObjectMorpher/outputs/` is already a sink — do not write adapter outputs there. Keep adapter outputs strictly under `runs/`.

---

## 7. Phased Plan

### Phase 1 — Headless render + pseudo-GT generation + minimal optimization (1–2 weeks)

Goal: a single command that takes an ObjectMorpher PLY and produces an "optimized PLY" using SG-upsampled pseudo-GT.

1. Scaffold `adapter/om_sg_adapter/` with `config.py`, `paths.py`, `envs.py`, `cameras/{conventions,trajectory,transforms_io}.py`.
2. Implement `om_bridge/load_gs.py` + `om_bridge/render_frames.py`. Acceptance: render `ObjectMorpher/sample.ply` along a 60-frame orbit at 256×256 and save with a valid `transforms.json`. Verify by re-loading via SG's `transforms.json` reader (smoke test, no fitting).
3. Implement `sg_bridge/run_upsampling.py` (shell-out into `realbasicvsr` env). Acceptance: 64×64 OM frames → 256×256 upsampled frames; sanity-check sharpness visually.
4. Implement `om_bridge/optimize_against.py` with L1 + alpha-mask + (optional) LPIPS. Freeze topology. Acceptance: loss decreases monotonically over 2k iters on a non-edited GS; the optimized renders are pixel-closer to the pseudo-GT than the baseline renders.
5. Wire all of the above behind `cli/run_phase1.py` and `scripts/run_phase1.sh`.
6. Write one comparison-grid reporter.

Deliverable: `runs/phase1_smoke/reports/comparison_grid.png` and a short writeup of L1/LPIPS deltas.

### Phase 2 — Apply to ARAP-edited GS + iterate, optionally close the loop with SG fitting (2–4 weeks)

Goal: prove pseudo-GT actually improves an *edit*, not just a canonical render.

1. Run an actual ARAP edit in `edit_gui.py`, save `edited.ply`. Repeat Phase 1 on `edited.ply`; measure quality vs. baseline edit.
2. Add `sg_bridge/run_fitting.py` so SG can re-lift the upsampled frames into a high-res 3DGS independently (`step_2_fitting_with_3dgs`). Compare three artifacts: (a) OM-edited GS, (b) OM-edited GS optimized against pseudo-GT (Phase 1), (c) SG-fit GS from the upsampled OM frames. The 3-way comparison tells us which path strengthens 2D→3D most.
3. Add geometric losses (`losses/geometric.py`): alpha consistency, optional monocular depth from the OM render. Re-run.
4. Sweep optimizable parameter sets: `{_features_dc}`, `{_features_dc, _xyz}`, full. Pick the sweet spot.
5. Move from a single orbit to multiple novel-trajectory sets (mirror SG's `traj_0`/`traj_1` idea) — `traj_0` for eval, `traj_1` for training only.

Deliverable: a metrics table (L1 / LPIPS / FID-against-original-image) over a small handful of ObjectMorpher scenes.

### Phase 3 — Strengthen 2D→3D in ObjectMorpher (longer)

Goal: replace or augment TRELLIS's image-to-3D step with an SG-style video-prior-conditioned lifter.

1. Take the *original* input image used by ObjectMorpher's `reconstruct_from_2d` step. Generate a synthesized multi-view video (any video prior we trust — RealBasicVSR is too weak, but the SG pipeline is structured to accept VideoGigaGAN / Upscale-a-Video / a diffusion video model). Feed it to SG's fitting step to produce a high-resolution initial 3DGS *for ObjectMorpher's editing stage*.
2. Adapter changes: `pipelines/phase3_2d_to_3d.py`. Upstream changes: probably a `--init_gs_ply` option in `editing/edit_gui.py` so the operator can load the SG-fit PLY directly. That's a tiny, isolated hook, kept under `adapter/hooks/objectmorpher/`.
3. Validate by running the existing user-study protocol (`ObjectMorpher/userstudy.py`) on edits made on SG-initialized GS vs. TRELLIS-initialized GS.
4. If results hold, write the actual research note.

Deliverable: an "SG-lifted initialization" option in the editing flow, plus a head-to-head comparison.

---

## 8. Open Questions to Resolve Before Coding

1. **Which conda envs are actually installed** on the target machine? `scripts/env_check.sh` should verify `objectmorpher`, `super_gaussian_eccv24`, `realbasicvsr` all exist with the right `python -c "import …"` probes.
2. **Which PLY do we start from?** `ObjectMorpher/sample.ply`, `mecha1.ply`, `tungtungtungsahur.ply`, or a TRELLIS output? Suggestion: start with `sample.ply` for Phase 1 smoke, switch to a TRELLIS output for Phase 2.
3. **Target output resolution.** SG's pipeline is built around 64→256 (4×). We can run OM rendering at 256 directly and skip the 64×64 stage; that loses the "video prior" benefit. Recommendation: keep the 4× ratio so the upsampler has work to do — render OM at 128 or 256, downsample to 32 or 64 for the LR input, upsample with SG, optimize against the 128/256 result.
4. **Where do we want frame outputs to live long-term?** Default `runs/<run>/` is fine; if disk is constrained, gate the per-iter render dump behind a config flag.

Until these are decided, the scaffolding in §3 still applies — only the YAML defaults in `adapter/configs/phase1.yaml` change.
