# Subsetix — CuPy Interval Set Operations

This repository is now a pure Python/CuPy project that focuses on set
operations (union / intersection / difference) between per-row interval sets.
It runs entirely on the GPU using custom CuPy `RawKernel`s and provides both a
high-level expression API plus targeted benchmarks and AMR demos.

## Requirements

- Python 3.9+
- [CuPy](https://cupy.dev/) built against a CUDA toolkit matching your GPU
  (`pip install cupy-cuda12x`, `cupy-cuda11x`, …)
- NumPy (tests, utilities)

## Installation

Create a virtual environment, install CuPy and the repository in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install cupy-cuda12x  # pick the wheel matching your CUDA version
pip install -r requirements.txt  # optional if you add one
```

The package is importable as `subsetix_cupy`.

## Project Layout

- `subsetix_cupy/expressions.py` — GPU-only expression system (union /
  intersection / difference / symmetric difference / complement) using CuPy
  kernels and a reusable `CuPyWorkspace`.
- `subsetix_cupy/kernels.py` — RawKernel definitions for the count/write passes.
- `subsetix_cupy/benchmark_multishape.py` — Headless benchmark that replays the
  operations repeatedly to stress the GPU.
- `subsetix_cupy/tests/test_expressions.py` — Unit tests covering the GPU
  expression API (requires a working CuPy + CUDA runtime).
- `subsetix_amr2/demo_two_level_advection.py` — Two-level AMR advection driver
  using the interval-set workspace + graph APIs.

## Usage

### Benchmark (no plots)

```bash
python -m subsetix_cupy.benchmark_multishape \
    --resolution 1024 \
    --rows 24 \
    --cols 24 \
    --rect-strips 2 \
    --circles-per-cell 2 \
    --iterations 1000 \
    --warmup 10
```

The script prints interval counts, total time, time per iteration, and average
nanoseconds per produced interval.

### Two-level AMR demo

```bash
python -m subsetix_amr2.demo_two_level_advection \
    --coarse 256 \
    --steps 200 \
    --refine-threshold 0.05 \
    --ghost-halo 1
```

This driver advects a square along the diagonal, rebuilds the interval-based
mesh every step, and can optionally export VTK snapshots with `--save-vtk`.

### Benchmark Suite (GPU micro-benchmarks)

Use the dedicated harness to profile the core functionalities (expressions,
morphology, multilevel helpers, AMR step, VTK export):

```bash
# List available cases
python -m subsetix_cupy.benchmarks --list

# Run everything (default repeat/warmup per case)
python -m subsetix_cupy.benchmarks

# Filter with a regex, override repeat/warmup and export JSON
python -m subsetix_cupy.benchmarks --pattern "amr_" --repeat 5 --warmup 2 --json results.json
```

The suite uses `cupyx.profiler.benchmark` under the hood, so timings are based
on CUDA events. Each case prints average GPU time (ms) plus the standard
deviation across repeats.

## Testing

```bash
python -m unittest subsetix_cupy.tests.test_expressions
```

The suite requires a functional CuPy installation with CUDA access. It skips
automatically otherwise.

## Development Notes

- `CuPyWorkspace` reuses device buffers between evaluations; copy the output if
  you need to keep previous results alive.
- The expression API accepts only CuPy arrays; there is no CPU fallback.
- Contributions are welcome — consider adding additional set operations,
  support for 3D volumes, or multi-stream benchmarking.
