# CLAUDE.md

## Project context

This repository is a research workspace for integrating:
- `ObjectMorpher/` — 3D-aware image editing via deformable 3D Gaussian Splatting
- `SuperGaussian/` — video-prior-based 3D super-resolution / refinement

The current research direction is:
- use **SuperGaussian-style video-frame outputs as pseudo-GT targets**
- improve / optimize **ObjectMorpher** against those pseudo-GT frames
- longer-term, strengthen or replace parts of ObjectMorpher's **2D-to-3D lifting** path

This idea is already narrowed down to a practical Phase 1:
1. load an ObjectMorpher Gaussian (`.ply`)
2. render comparable multi-view frames headlessly
3. prepare SuperGaussian-compatible frame/camera inputs
4. use SuperGaussian-style outputs as pseudo-GT frame targets
5. only later add optimization and deeper 2D→3D integration

## Working principles

1. **Keep upstream repos mostly clean**
   - Treat `ObjectMorpher/` and `SuperGaussian/` as upstream-style third-party engines.
   - Avoid deep edits inside them unless absolutely necessary.

2. **Prefer a root-level adapter/orchestration layer**
   - Put new integration code under `adapter/`.
   - Wrapper code, path management, camera conversion, staging logic, and experiment pipelines should live there.

3. **File-based handoff is preferred**
   - Environment separation is expected.
   - Do not assume both projects should be tightly imported into one runtime immediately.
   - Frame folders, `transforms.json`, and `.ply` files are valid contracts.

4. **Do not jump too early into full optimization**
   - First make headless render/export work.
   - Then make SuperGaussian input staging work.
   - Then verify frame/camera pairing.
   - Only after that should optimization be added.

## Important technical context

### ObjectMorpher seams of interest
- `ObjectMorpher/reconstruct_from_2d/app.py` — TRELLIS-based lifting path
- `ObjectMorpher/editing/gaussian_renderer/` — differentiable rendering seam
- `ObjectMorpher/editing/scene/cameras.py` — camera representation seam
- `ObjectMorpher/editing/scene/gaussian_model.py` — Gaussian load/save seam
- `ObjectMorpher/editing/cam_utils.py` — orbit camera utility

### SuperGaussian seams of interest
- `SuperGaussian/main_supergaussian.py`
- `SuperGaussian/sg_utils/sg_helper.py`
- `SuperGaussian/third_parties/gaussian-splatting/`
- `transforms.json` camera schema used by SuperGaussian workflows

### Highest-risk bug class
The most likely silent failure is **camera convention mismatch**:
- OpenCV vs OpenGL convention
- c2w / w2c mismatch
- axis flip used in SuperGaussian export path

If results look visually wrong, check camera conventions first.

## Current repository intent

The repository now contains:
- `ADAPTER_PLAN.md` — detailed integration plan
- `adapter/` — Phase 1 scaffold already started

Current scaffold status:
- adapter structure exists
- path/config/camera/helper skeletons exist
- Phase 1 CLI exists
- actual headless ObjectMorpher rendering is **not fully wired yet**

## What Claude should do next

Good next tasks:
1. implement or unblock **headless ObjectMorpher render export**
2. implement **SuperGaussian input staging** using frame folders + `transforms.json`
3. verify a **single-sample smoke flow** with `ObjectMorpher/sample.ply`
4. only then move into minimal optimization

## What Claude should avoid

- Do **not** deeply refactor `ObjectMorpher/` or `SuperGaussian/` unless necessary.
- Do **not** start with end-to-end training or full optimization loops.
- Do **not** mix local operator tooling (`config/`, `scripts/`, `.agent_runs/`) into research adapter design.
- Do **not** assume environment/runtime unification is trivial.

## Validation expectations

When making changes:
- prefer the smallest meaningful verification
- e.g. import checks, compile checks, smoke CLI runs, file tree inspection
- if something is blocked, leave a clear TODO and explain the blocker rather than pretending it works
