from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None  # type: ignore


def _to_numpy(a):
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def _write_dataarray_ascii(f, name: str, arr: np.ndarray, vtk_type: str) -> None:
    flat = arr.ravel(order="C")
    f.write(f'        <DataArray type="{vtk_type}" Name="{name}" format="ascii">\n')
    # Write in chunks to avoid gigantic single lines
    n = flat.size
    stride = 4096
    for i in range(0, n, stride):
        chunk = flat[i : i + stride]
        f.write("          ")
        f.write(" ".join(str(x) for x in chunk))
        f.write("\n")
    f.write("        </DataArray>\n")


def write_rectilinear_grid_vtr(
    path: str,
    u_cell,
    dx: float,
    dy: float,
    *,
    origin=(0.0, 0.0, 0.0),
    cell_arrays: Dict[str, np.ndarray] | None = None,
) -> None:
    """Write a 2D cell-centered scalar field and optional cell arrays to VTK XML RectilinearGrid (.vtr).

    - u_cell: (H, W) cell-centered scalar array (NumPy or CuPy)
    - dx, dy: cell sizes; origin: (x0, y0, z0)
    - cell_arrays: dict of additional cell arrays (bool/int/float) of shape (H, W)
    """
    u_np = _to_numpy(u_cell)
    H, W = u_np.shape
    x0, y0, z0 = origin
    xs = x0 + dx * np.arange(W + 1, dtype=np.float32)
    ys = y0 + dy * np.arange(H + 1, dtype=np.float32)
    zs = np.array([z0], dtype=np.float32)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("<?xml version=\"1.0\"?>\n")
        f.write("<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
        f.write(f"  <RectilinearGrid WholeExtent=\"0 {W} 0 {H} 0 0\">\n")
        f.write(f"    <Piece Extent=\"0 {W} 0 {H} 0 0\">\n")
        f.write("      <CellData Scalars=\"u\">\n")
        _write_dataarray_ascii(f, "u", u_np.astype(np.float32, copy=False), "Float32")
        if cell_arrays:
            for name, arr in cell_arrays.items():
                arr_np = _to_numpy(arr)
                if arr_np.dtype == np.bool_:
                    arr_np = arr_np.astype(np.int32)
                    vtk_type = "Int32"
                elif np.issubdtype(arr_np.dtype, np.integer):
                    arr_np = arr_np.astype(np.int32)
                    vtk_type = "Int32"
                else:
                    arr_np = arr_np.astype(np.float32)
                    vtk_type = "Float32"
                _write_dataarray_ascii(f, name, arr_np, vtk_type)
        f.write("      </CellData>\n")
        f.write("      <Coordinates>\n")
        f.write('        <DataArray type="Float32" Name="X_COORDINATES" NumberOfComponents="1" format="ascii">\n')
        f.write("          ")
        f.write(" ".join(str(x) for x in xs))
        f.write("\n        </DataArray>\n")
        f.write('        <DataArray type="Float32" Name="Y_COORDINATES" NumberOfComponents="1" format="ascii">\n')
        f.write("          ")
        f.write(" ".join(str(y) for y in ys))
        f.write("\n        </DataArray>\n")
        f.write('        <DataArray type="Float32" Name="Z_COORDINATES" NumberOfComponents="1" format="ascii">\n')
        f.write(f"          {zs[0]}\n")
        f.write("        </DataArray>\n")
        f.write("      </Coordinates>\n")
        f.write("    </Piece>\n")
        f.write("  </RectilinearGrid>\n")
        f.write("</VTKFile>\n")


def write_pvd(path: str, datasets: List[Tuple[float, List[str]]]) -> None:
    """Write a ParaView .pvd collection file.

    datasets: list of (time, [fileL0, fileL1, fileL2]) relative to the pvd file location.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("<?xml version=\"1.0\"?>\n")
        f.write("<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
        f.write("  <Collection>\n")
        for t, files in datasets:
            for part, fname in enumerate(files):
                f.write(f"    <DataSet timestep=\"{t}\" group=\"\" part=\"{part}\" file=\"{fname}\"/>\n")
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")


def save_amr3_step_vtr(
    out_dir: str,
    base: str,
    step: int,
    time_value: float,
    *,
    u0,
    u1,
    u2,
    refine0,
    L1_mask,
    refine1_mid,
    L2_mask,
    dx0: float,
    dy0: float,
    dx1: float,
    dy1: float,
    dx2: float,
    dy2: float,
) -> Tuple[List[str], Tuple[float, List[str]]]:
    """Save three .vtr files (L0/L1/L2) for the current step and return filenames and the PVD entry."""
    os.makedirs(out_dir, exist_ok=True)
    # Filenames
    f0 = f"{base}_step{step:04d}_L0.vtr"
    f1 = f"{base}_step{step:04d}_L1.vtr"
    f2 = f"{base}_step{step:04d}_L2.vtr"
    p0 = os.path.join(out_dir, f0)
    p1 = os.path.join(out_dir, f1)
    p2 = os.path.join(out_dir, f2)

    # Coarse arrays
    coarse_only = _to_numpy(~_to_numpy(refine0).astype(bool)).astype(np.int32)
    write_rectilinear_grid_vtr(
        p0, u0, dx0, dy0, cell_arrays={"coarse_only": coarse_only}
    )

    # Mid arrays
    mid_active = _to_numpy(L1_mask).astype(bool)
    mid_only = _to_numpy(L1_mask & (~refine1_mid)).astype(bool)
    write_rectilinear_grid_vtr(
        p1,
        u1,
        dx1,
        dy1,
        cell_arrays={
            "mid_active": mid_active,
            "mid_only": mid_only,
        },
    )

    # Fine arrays
    fine_active = _to_numpy(L2_mask).astype(bool)
    write_rectilinear_grid_vtr(
        p2,
        u2,
        dx2,
        dy2,
        cell_arrays={"fine_active": fine_active},
    )

    rels = [f0, f1, f2]
    return rels, (time_value, rels)


def _append_cell(points: List[Tuple[float, float, float]],
                 connectivity: List[int], offsets: List[int], types: List[int],
                 x0: float, y0: float, dx: float, dy: float, i: int, j: int) -> None:
    # 4 points per quad, order: (x0,y0),(x1,y0),(x1,y1),(x0,y1)
    idx0 = len(points)
    x = x0 + j * dx; y = y0 + i * dy
    points.append((x, y, 0.0))
    points.append((x + dx, y, 0.0))
    points.append((x + dx, y + dy, 0.0))
    points.append((x, y + dy, 0.0))
    connectivity.extend([idx0, idx0 + 1, idx0 + 2, idx0 + 3])
    offsets.append(len(connectivity))
    types.append(9)  # VTK_QUAD


def write_unstructured_quads_vtu(
    path: str,
    cells: List[Tuple[np.ndarray, np.ndarray, int, float, float, float, float]],
    origin=(0.0, 0.0),
) -> None:
    """Write a single UnstructuredGrid (.vtu) where each selected cell across levels becomes a quad.

    cells: list of tuples (mask, u, level_id, dx, dy, x0, y0) per level.
    u is sampled cell-centered per cell; level_id is written to CellData 'level'.
    """
    points: List[Tuple[float, float, float]] = []
    connectivity: List[int] = []
    offsets: List[int] = []
    types: List[int] = []
    levels: List[int] = []
    values: List[float] = []
    ghosts: List[int] = []

    for entry in cells:
        if len(entry) == 7:
            mask, u, lvl, dx, dy, x0, y0 = entry
            ghost = None
        else:
            mask, u, lvl, dx, dy, x0, y0, ghost = entry
        m = _to_numpy(mask).astype(bool, copy=False)
        arr = _to_numpy(u).astype(np.float32, copy=False)
        g_arr = None if ghost is None else _to_numpy(ghost).astype(bool, copy=False)
        idxs = np.argwhere(m)
        for i, j in idxs:
            _append_cell(points, connectivity, offsets, types, x0, y0, dx, dy, int(i), int(j))
            levels.append(int(lvl))
            values.append(float(arr[i, j]))
            if g_arr is None:
                ghosts.append(0)
            else:
                ghosts.append(int(bool(g_arr[i, j])))

    P = np.asarray(points, dtype=np.float32)
    C = np.asarray(connectivity, dtype=np.int32)
    O = np.asarray(offsets, dtype=np.int32)
    T = np.asarray(types, dtype=np.uint8)
    L = np.asarray(levels, dtype=np.int32)
    U = np.asarray(values, dtype=np.float32)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("<?xml version=\"1.0\"?>\n")
        f.write("<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
        f.write("  <UnstructuredGrid>\n")
        f.write(f"    <Piece NumberOfPoints=\"{P.shape[0]}\" NumberOfCells=\"{len(L)}\">\n")
        f.write("      <CellData Scalars=\"u\">\n")
        _write_dataarray_ascii(f, "level", L, "Int32")
        ghost_arr = np.asarray(ghosts, dtype=np.int32)
        _write_dataarray_ascii(f, "ghost_mask", ghost_arr, "Int32")
        _write_dataarray_ascii(f, "u", U, "Float32")
        f.write("      </CellData>\n")
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        for i in range(0, P.shape[0], 4096):
            chunk = P[i : i + 4096]
            f.write("          ")
            f.write(" ".join(f"{x} {y} 0" for x, y, _ in chunk))
            f.write("\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("      <Cells>\n")
        _write_dataarray_ascii(f, "connectivity", C, "Int32")
        _write_dataarray_ascii(f, "offsets", O, "Int32")
        _write_dataarray_ascii(f, "types", T, "UInt8")
        f.write("      </Cells>\n")
        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


def save_amr3_mesh_vtu(
    out_dir: str,
    base: str,
    step: int,
    *,
    u0,
    u1,
    u2,
    coarse_only,
    mid_only,
    fine_active,
    dx0: float,
    dy0: float,
    dx1: float,
    dy1: float,
    dx2: float,
    dy2: float,
    origin=(0.0, 0.0),
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{base}_step{step:04d}_mesh.vtu"
    path = os.path.join(out_dir, fname)
    x0, y0 = origin
    write_unstructured_quads_vtu(
        path,
        cells=[
            (coarse_only, u0, 0, dx0, dy0, x0, y0),
            (mid_only, u1, 1, dx1, dy1, x0, y0),
            (fine_active, u2, 2, dx2, dy2, x0, y0),
        ],
        origin=(x0, y0),
    )
    return fname
