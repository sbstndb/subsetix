# Subsetix — CuPy Interval Set Operations

This repository is now a pure Python/CuPy project that focuses on set
operations (union / intersection / difference) between per-row interval sets.
It runs entirely on the GPU using custom CuPy `RawKernel`s and provides both a
high-level expression API and ready-to-use demos/benchmarks.

## Requirements

- Python 3.9+
- [CuPy](https://cupy.dev/) built against a CUDA toolkit matching your GPU
  (`pip install cupy-cuda12x`, `cupy-cuda11x`, …)
- NumPy, Matplotlib (for the interactive demo)

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
- `subsetix_cupy/demo_multishape.py` — Interactive demo that rasterises a grid
  of rectangles and circles, computes set operations, and plots the results.
- `subsetix_cupy/benchmark_multishape.py` — Headless benchmark that replays the
  operations repeatedly to stress the GPU.
- `subsetix_cupy/tests/test_expressions.py` — Unit tests covering the GPU
  expression API (requires a working CuPy + CUDA runtime).

## Usage

### Demo (with plots)

```bash
python -m subsetix_cupy.demo_multishape \
    --resolution 1024 \
    --rows 24 \
    --cols 24 \
    --rect-strips 2 \
    --circles-per-cell 2
```

Arguments let you adjust the density of rectangles/circles and the raster
resolution. Omit the flags for the default 512² grid. Add `--no-plot` to skip
Matplotlib (useful on headless machines).

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
