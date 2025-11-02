# subsetix-amr2

Utilities for building two-level adaptive mesh refinement (AMR) workflows on
top of the GPU-based interval operations provided by `subsetix`.

The package exposes:

- `subsetix_amr2.geometry.TwoLevelGeometry` – construct AMR layouts from dense
  refinement masks and keep the corresponding `IntervalSet` representations.
- `subsetix_amr2.fields` – coarse/fine synchronisation helpers operating on
  dense CuPy arrays while respecting the geometry.
- `subsetix_amr2.regrid` – gradient-based mask generation and grading helpers.

The goal is to decouple AMR-specific logic from higher level demos so that
applications can create reusable pipelines (refinement, grading, transfers)
without reimplementing geometry plumbing.

All helpers assume a two-level hierarchy (coarse + single refined level) and
use the non-allocating graph-backed APIs from `subsetix`.

## Demo

Run the bundled advection demo (requires CUDA + CuPy):

```bash
python -m subsetix_amr2.demo_two_level_advection --coarse 96 --steps 200 --plot
```

Command-line flags allow you to tweak refinement thresholds, grading, and
regridding cadence. Add `--animate --interval 40` to preview an animation or
`--save-animation out.gif` to write a file.

To export rectilinear grids and a combined mesh for ParaView, add
`--save-vtk out_vtk` (optionally with `--vtk-every 20` and `--ghost-halo 1`).
