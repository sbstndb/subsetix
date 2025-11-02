# Subsetix — CUDA Interval Intersections (1D/2D/3D)

- GPU intersections of 1D intervals arranged per-row (CSR layout) with a linear, two-pass algorithm (count + write) per row.
- Graph-first execution: all high-level helpers launch CUDA Graphs internally so the optimized replay path is the default.

**Data Layout (CSR)**
- Inputs per set S: `begin[]`, `end[]`, `row_offsets[]` (size = rows + 1).
- 3D adds `row_to_y[]`, `row_to_z[]` (size = rows) to recover `(y,z)` for each row.
- Outputs (optional): row indices (y or (z,y)), `r_begin[]`, `r_end[]`, `a_idx[]`, `b_idx[]`.

**Repository Layout**
- `src/interval_intersection.cuh`: Public API (graph launchers plus high-level wrappers). Start: src/interval_intersection.cuh:1
- `src/interval_intersection.cu`: Kernels `row_intersection_*`, graph capture helpers, and wrapper implementations. Start: src/interval_intersection.cu:1
- `src/surface_demo.cu`: 2D multi-shape demo (rectangles vs circles). Entry: src/surface_demo.cu:103
- `src/volume_demo.cu`: 3D multi-shape demo (boxes vs spheres). Entry: src/volume_demo.cu:120
- `bench/workspace_benchmark.cu`: Device-only benchmarks for the graph path (dense, sparse, multi-stream, multi-shape sequences). Start: bench/workspace_benchmark.cu:1
- `bench/count_multishape.cpp`: Host-only utility to count 2D multi-shape intervals. Start: bench/count_multishape.cpp:1
- Generators: `src/surface_generator.*` and `src/volume_generator.*` (shape generation, unions, rasterization, VTK writers).
- `subsetix_cupy/`: Python prototype exposing a simplified expression builder that mirrors the CUDA chain API (see subsetix_cupy/expressions.py:1). It requires a functional CuPy+CUDA installation (no CPU fallback) and evaluates operations with custom RawKernels (count/write per row). The kernels live in `subsetix_cupy/kernels.py`, and `CuPyWorkspace` allows reusing device buffers for repeated evaluations. Convenience helpers also cover symmetric difference and complement chains.

**Build**
- Requirements: CUDA Toolkit (nvcc, Thrust/CUB), CMake ≥ 3.18, GTest.
- CUDA architectures: configurable via `-DCMAKE_CUDA_ARCHITECTURES=...`; defaults to `75;86;89` if not provided.

Commands:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**Executables**
- `build/intersection_test` — GTest suite exercising the graph flow (2D/3D, positive and negative cases).
- `build/workspace_benchmark` — device-only timing for graph launches (multi-stream, empty scenes, multi-dataset sequences, multi-shape).
- `build/surface_demo` — 2D demo (toggle `RUN_HOST_UNION` and `WRITE_VTK`). See src/surface_demo.cu:103
- `build/volume_demo` — 3D demo (`WRITE_VTK` off by default). See src/volume_demo.cu:124
- `build/exemple_main` — minimal 3D example.
- `python -m subsetix_cupy.demo_multishape` — CuPy-only multi-shape demo (rectangles vs circles) that plots union, intersection, and difference via Matplotlib (requires GPU + CUDA driver). Defaults to a 512² grid with ~4.8k output intervals; customise density with `--resolution`, `--rows`, `--cols`, etc.
- `python -m subsetix_cupy.benchmark_multishape` — headless benchmark that replays the multi-shape union/intersection/difference loop (default 100 iterations) and reports timings/interval counts. Same CLI options as the demo minus plotting.

**API Overview**
- `SurfaceOperationChain` + `SurfaceChainExecutor` build and execute CUDA Graphs for arbitrary chains of set operations (Union, Intersection, Difference). The executor plans temp-storage, captures two graphs (offsets, write), and operates entirely on caller-provided device memory.
- Lower-level interval APIs remain available if you need to integrate existing code paths:
  - Offsets graph helpers: `create{Interval|Volume}IntersectionOffsetsGraph`, `launchIntervalIntersectionGraph`, `destroyIntervalIntersectionGraph`.
  - Write graph helpers mirroring the offsets API.

Executor workflow (2D chain example):
```c++
subsetix::SurfaceOperationChain chain;
auto a = chain.add_input({d_a_begin, d_a_end, d_a_offsets, rows});
auto b = chain.add_input({d_b_begin, d_b_end, d_b_offsets, rows});
auto u = chain.add_operation(subsetix::SurfaceOpType::Union, a, b);
auto diff = chain.add_operation(subsetix::SurfaceOpType::Difference, u, b);

subsetix::SurfaceChainExecutor exec;
CUDA_CHECK(exec.prepare(chain));
CUDA_CHECK(exec.plan_resources());

// Allocate workspace once (counts = rows, offsets = rows+1 + totals)
subsetix::SurfaceWorkspaceView ws{d_counts, d_offsets, exec.counts_stride(),
                                  exec.offsets_stride(), d_cub_temp, cub_bytes, d_totals};

// Pre-size arena slice for intermediate union (diff slice filled after totals are known)
std::vector<subsetix::SurfaceArenaSlice> slices(exec.node_count());
slices[0].d_begin = d_union_begin;
slices[0].d_end   = d_union_end;
slices[0].d_y_idx = d_union_y;
slices[0].capacity = union_capacity;
subsetix::SurfaceArenaView arena{ slices.data(), slices.size() };

subsetix::SurfaceChainGraph offsets_graph;
CUDA_CHECK(exec.capture_offsets_graph(ws, arena, &offsets_graph));
CUDA_CHECK(subsetix::launch_surface_chain_graph(offsets_graph));
CUDA_CHECK(cudaStreamSynchronize(offsets_graph.stream));

// Read totals -> allocate final capacity -> capture write graph
CUDA_CHECK(cudaMemcpyAsync(h_totals.data(), d_totals,
                           exec.node_count() * sizeof(int),
                           cudaMemcpyDeviceToHost));
int diff_total = h_totals.back();
slices.back().d_begin = d_diff_begin;
slices.back().d_end   = d_diff_end;
slices.back().d_y_idx = d_diff_y;
slices.back().capacity = diff_total;
subsetix::SurfaceChainGraph write_graph;
CUDA_CHECK(exec.capture_write_graph(ws, arena, &write_graph));
```

Once captured, `launch_surface_chain_graph` can replay offsets and write graphs every iteration without additional setup.

**Benchmarks**
- Run: `./build/workspace_benchmark`
- Reports classical, workspace, graph, and SurfaceChainExecutor timings across dense/sparse, empty, multi-stream, and multi-dataset scenarios (2D + 3D).
- Metrics: ms/iter (allocations happen outside the timed region) with optional normalization per interval where applicable.

- **Demos**
  - `./build/surface_demo` — constructs 5 rectangles vs 5 circles and runs the `SurfaceChainExecutor` pipeline end-to-end (offsets → totals → write) while reporting timings.
  - `./build/volume_demo` — 5 boxes vs 5 spheres; currently relies on the lower-level graph helpers until a volume-chain executor is available. Optional VTK export remains disabled by default.
- VTK export can generate large files; leave disabled for quick runs.

**Tests**
- `ctest --test-dir build` compiles/launches the graph path (basic overlaps, sparse scenes, error validation).
- Device execution requires a working CUDA driver; in restricted environments the tests compile but kernels may not launch.
- Python prototype: `python -m unittest subsetix_cupy.tests.test_expressions` validates the nested-expression flow (requires CuPy+CUDA, otherwise import fails).

**Performance Notes**
- The two-graph sequence keeps allocations outside the captured workload; reuse the instantiated graphs for steady-state iterations.
- Empty scenes skip the write graph; `workspace_benchmark` reports the offsets-only time separately.
- For very sparse inputs consider adding row-compaction or an atomic fallback (see roadmap).

**Roadmap**
- Active-row compaction (CUB DeviceSelect) for sparse scenes.
- Atomic fallback (count+write fused) for ultra-sparse cases.
- Additional negative tests for graph configs (null pointers, mismatched row counts).
- CLI flags for benchmarks (scenario selection, iteration control).
