# Adapter scaffold

This directory holds the integration layer between `ObjectMorpher/` and `SuperGaussian/`.

## Goals

- Keep both paper repos mostly untouched.
- Put orchestration, path management, camera conversion, and staging code here.
- Start with Phase 1: ObjectMorpher headless frame export + SuperGaussian input staging.

## Layout

- `om_sg_adapter/config.py` — config dataclasses
- `om_sg_adapter/paths.py` — run directory conventions
- `om_sg_adapter/cameras/` — trajectory + transforms.json helpers
- `om_sg_adapter/om_bridge/` — ObjectMorpher wrappers
- `om_sg_adapter/sg_bridge/` — SuperGaussian wrappers
- `om_sg_adapter/pipelines/` — high-level workflows
- `om_sg_adapter/cli/` — command entrypoints
- `hooks/` — tiny optional in-tree hook shims only if imports force it
- `configs/` — sample YAML configs
- `scripts/` — convenience launchers

## Phase 1 intended flow

1. Load an ObjectMorpher `.ply`.
2. Build a deterministic camera trajectory.
3. Render comparable frames headlessly.
4. Write `transforms.json` + staged frame folders in a SuperGaussian-friendly layout.
5. Verify file contracts before adding optimization.

## Current status

This is a scaffold. Rendering and staging entrypoints exist with conservative wrappers and TODOs. The optimization loop is intentionally not implemented yet.

## Quick check

```bash
PYTHONPATH=adapter python3 -m om_sg_adapter.cli.run_phase1 --run-name smoke
```

Expected current behavior: directories are created and the command reports that headless ObjectMorpher rendering is still blocked/pending deeper hook validation.
