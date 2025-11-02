# Subsetix — CUDA Interval Intersections (1D/2D/3D)

- GPU intersections of 1D intervals arranged per-row (CSR layout) with a linear, two-pass algorithm (count + write) per row.
- Supports 1D (single row), 2D (rows = y), and 3D (rows = (z,y) flattened via row maps).

**Data Layout (CSR)**
- Inputs per set S: `begin[]`, `end[]`, `row_offsets[]` (size = rows + 1).
- 3D adds `row_to_y[]`, `row_to_z[]` (size = rows) to recover (y,z) for each row.
- Outputs (optional): row indices (y or (z,y)), `r_begin[]`, `r_end[]`, `a_idx[]`, `b_idx[]`.

**Repository Layout**
- `src/interval_intersection.cuh`: Public API (classic, workspace, and CUDA Graphs). Start: src/interval_intersection.cuh:1
- `src/interval_intersection.cu`: Kernels `row_intersection_*` and orchestration (two-pass, Thrust/CUB, Graph capture). Start: src/interval_intersection.cu:1
- `src/surface_demo.cu`: 2D multi-shape demo (rectangles vs circles). Entry: src/surface_demo.cu:103
- `src/volume_demo.cu`: 3D multi-shape demo (boxes vs spheres). Entry: src/volume_demo.cu:120
- `bench/workspace_benchmark.cu`: Benchmarks for classic/workspace/graphs, multi-streams, empty cases, multi-shape sequences, and parallel multi-shape 2D. Start: bench/workspace_benchmark.cu:1
- `bench/count_multishape.cpp`: Host-only utility to count 2D multi-shape intervals. Start: bench/count_multishape.cpp:1
- Generators: `src/surface_generator.*` and `src/volume_generator.*` (rect/circle/box/sphere generation, unions, rasterization, VTK writers).

**Build**
- Requirements: CUDA Toolkit (nvcc, Thrust/CUB), CMake ≥ 3.18, GTest.
- CUDA architectures: configurable via `-DCMAKE_CUDA_ARCHITECTURES=...`; defaults to `75;86;89` if not provided.

Commands:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**Executables**
- `build/intersection_test` — GTest suite for 2D/3D (classic, workspace, graphs).
- `build/workspace_benchmark` — device-only timing for multiple modes (2D/3D, streams, graphs, empty, multi-shape).
- `build/surface_demo` — 2D demo (toggle `RUN_HOST_UNION` and `WRITE_VTK`). See src/surface_demo.cu:107
- `build/volume_demo` — 3D demo (`WRITE_VTK` off by default). See src/volume_demo.cu:124
- `build/exemple_main` — minimal example.

- **API Overview**
  - Workspace (no internal allocations):
    - Step 1 — offsets: `compute*IntersectionOffsets(...)` fills `counts/offsets` and optionally returns `total`. Start: src/interval_intersection.cuh:90
    - Step 2 — write: `write*IntersectionsWithOffsets(...)` consumes `offsets` into preallocated buffers. Start: src/interval_intersection.cuh:105
    - Union/Diff follow the same pattern via `compute*UnionOffsets(...)` / `write*UnionWithOffsets(...)` and `compute*DifferenceOffsets(...)` / `write*DifferenceWithOffsets(...)`. Start: src/interval_intersection.cuh:226
    - Lower-level `enqueue*` variants accept a CUB workspace (capture-safe). Start: src/interval_intersection.cuh:58, src/interval_intersection.cuh:128
  - CUDA Graphs (two-graph workflow):
    - Offsets graph: `create{Interval|Volume}IntersectionOffsetsGraph(...)` (needs `counts`, `offsets`, CUB workspace, optional `d_total`). Start: src/interval_intersection.cuh:147
    - Write graph: `create{Interval|Volume}IntersectionWriteGraph(...)` (after allocating outputs with capacity `total`). Start: src/interval_intersection.cuh:161
    - Launch with `launch*Graph(...)`; destroy with `destroy*Graph(...)`. Start: src/interval_intersection.cuh:165

Workspace 2D example (sketch):
```c++
// d_a_*/d_b_* on device; y_count identical
cudaStream_t s; cudaStreamCreate(&s);
thrust::device_vector<int> counts(y_count), offsets(y_count);
int total = 0;
computeIntervalIntersectionOffsets(d_a_begin, d_a_end, d_a_y_offsets, y_count,
                                   d_b_begin, d_b_end, d_b_y_offsets, y_count,
                                   counts.data().get(), offsets.data().get(), &total, s);
ResultBuffers out = allocResults(total); // app-side helper
writeIntervalIntersectionsWithOffsets(d_a_begin, d_a_end, d_a_y_offsets, y_count,
                                      d_b_begin, d_b_end, d_b_y_offsets, y_count,
                                      offsets.data().get(),
                                      out.y_idx, out.begin, out.end, out.a_idx, out.b_idx, s);
cudaStreamDestroy(s);
```

CUDA Graphs 2D (two graphs):
```c++
// 1) Offsets graph (with CUB workspace and d_total)
IntervalIntersectionOffsetsGraphConfig ocfg{ /* d_a_*, d_b_*, counts, offsets, temp, d_total, a_y_count */ };
IntervalIntersectionGraph og; createIntervalIntersectionOffsetsGraph(&og, ocfg);
launchIntervalIntersectionGraph(og);
int total=0; cudaMemcpy(&total, d_total, sizeof(int), cudaMemcpyDeviceToHost);
// 2) Write graph with preallocated outputs (capacity = total)
IntervalIntersectionWriteGraphConfig wcfg{ /* d_a_*, d_b_*, offsets, outputs*, total */ };
IntervalIntersectionGraph wg; createIntervalIntersectionWriteGraph(&wg, wcfg);
// 3) Reuse
launchIntervalIntersectionGraph(og);
launchIntervalIntersectionGraph(wg);
```

**Benchmarks**
- Run: `./build/workspace_benchmark`
- Modes: classic/workspace/graph for 2D/3D; multi-stream; “empty” intersections; multi-dataset sequences; 2D multi-shape parallel (8 pairs, one stream per pair).
- Device-only timing: allocations and H2D/D2H outside the measured window.
- Metrics:
  - ns/interval (inputs) = normalized by |A| + |B| (linear work per row).
  - ns/interval (outputs) = normalized by produced intersections (used by demos).

**Demos**
- 2D: `./build/surface_demo` — prints interval counts and “Intersection time (device)” in ns/interval (outputs).
  - Constants: `intervals_per_row`, `width`, `height`, `RUN_HOST_UNION`, `WRITE_VTK`. See src/surface_demo.cu:103
- 3D: `./build/volume_demo` — similar for volumes; `WRITE_VTK` off by default. See src/volume_demo.cu:124
- VTK: enabling VTK output generates large files; keep off for quick runs.

**Tests**
- Run: `ctest --test-dir build` or `./build/intersection_test`.
- The suite validates classic/workspace/graphs correctness for basic 2D cases. Start: src/test_intersection.cu:1
- Note: running requires a working CUDA driver; in restricted environments, device allocations may fail.

**Performance Notes**
- 2D under-occupancy: one thread per row can under-use the GPU if there are few rows. Batch multiple 2D pairs (multi-stream) to increase occupancy.
- Empty intersections: benchmarks measure offsets-only (count+scan); write is skipped if `total==0`.
- Dominant costs: launch/synchronization overheads and, for non-empty scenes, the write kernel; CUB scan is usually minor.
- Avoid repeated `cudaMalloc`: prefer the workspace/graphs API (preallocate/reuse).

**Roadmap**
- Active-row compaction (CUB DeviceSelect) for sparse scenes.
- Atomic fallback (count+write fused) for ultra-sparse cases.
- Negative tests for Graph configs (null pointers, zero capacities, mismatched rows).
- CLI flags for benchmarks (select scenarios, number of multi-shape pairs, etc.).
