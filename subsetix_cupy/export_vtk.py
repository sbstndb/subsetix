from __future__ import annotations

import os
import struct
from typing import Any, Dict, Iterable, List, Tuple, NamedTuple

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


class _QuadChunk(NamedTuple):
    cell_count: int
    points: np.ndarray
    connectivity: np.ndarray
    offsets: np.ndarray
    types: np.ndarray
    levels: np.ndarray
    values: np.ndarray
    ghosts: np.ndarray


def _empty_chunk() -> _QuadChunk:
    empty_float = np.zeros((0, 3), dtype=np.float32)
    empty_int = np.zeros(0, dtype=np.int32)
    empty_u8 = np.zeros(0, dtype=np.uint8)
    return _QuadChunk(
        cell_count=0,
        points=empty_float,
        connectivity=empty_int,
        offsets=empty_int,
        types=empty_u8,
        levels=empty_int,
        values=np.zeros(0, dtype=np.float32),
        ghosts=empty_int,
    )


def _assemble_quads(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    *,
    level: int,
    dx: float,
    dy: float,
    x0: float,
    y0: float,
    ghost_values: np.ndarray,
) -> _QuadChunk:
    total = rows.size
    if total == 0:
        return _empty_chunk()

    rows = rows.astype(np.int32, copy=False)
    cols = cols.astype(np.int32, copy=False)
    vals = values.astype(np.float32, copy=False)
    ghost_vals = ghost_values.astype(np.int32, copy=False)

    dx32 = np.float32(dx)
    dy32 = np.float32(dy)
    x0_32 = np.float32(x0)
    y0_32 = np.float32(y0)

    x_left = x0_32 + cols.astype(np.float32, copy=False) * dx32
    x_right = x_left + dx32
    y_bottom = y0_32 + rows.astype(np.float32, copy=False) * dy32
    y_top = y_bottom + dy32

    points = np.empty((total * 4, 3), dtype=np.float32)
    # (x0, y0)
    points[0::4, 0] = x_left
    points[0::4, 1] = y_bottom
    points[0::4, 2] = 0.0
    # (x1, y0)
    points[1::4, 0] = x_right
    points[1::4, 1] = y_bottom
    points[1::4, 2] = 0.0
    # (x1, y1)
    points[2::4, 0] = x_right
    points[2::4, 1] = y_top
    points[2::4, 2] = 0.0
    # (x0, y1)
    points[3::4, 0] = x_left
    points[3::4, 1] = y_top
    points[3::4, 2] = 0.0

    connectivity = np.arange(total * 4, dtype=np.int32)
    offsets = np.arange(1, total + 1, dtype=np.int32) * 4
    types = np.full(total, 9, dtype=np.uint8)
    levels = np.full(total, int(level), dtype=np.int32)

    return _QuadChunk(
        cell_count=total,
        points=points,
        connectivity=connectivity,
        offsets=offsets,
        types=types,
        levels=levels,
        values=vals,
        ghosts=ghost_vals,
    )


def _interval_field_chunk(
    field,
    level: int,
    dx: float,
    dy: float,
    x0: float,
    y0: float,
    ghost_flag: int,
) -> _QuadChunk:
    if cp is None:
        raise RuntimeError("IntervalField export requires CuPy")

    offsets = field.interval_cell_offsets
    if offsets.size == 0:
        return _empty_chunk()

    total = int(offsets[-1].item())
    if total == 0:
        return _empty_chunk()

    offsets = offsets.astype(cp.int32, copy=False)
    begin = field.interval_set.begin.astype(cp.int32, copy=False)
    interval_rows = field.interval_set.interval_rows().astype(cp.int32, copy=False)

    cell_ids = cp.arange(total, dtype=cp.int32)
    interval_idx = cp.searchsorted(offsets[1:], cell_ids, side="right")
    local = cell_ids - offsets[interval_idx]
    cols = begin[interval_idx] + local
    rows = interval_rows[interval_idx]

    rows_np = cp.asnumpy(rows)
    cols_np = cp.asnumpy(cols)
    values_np = cp.asnumpy(field.values.astype(cp.float32, copy=False))
    ghost_vals = np.full(total, int(ghost_flag), dtype=np.int32)

    return _assemble_quads(
        rows_np,
        cols_np,
        values_np,
        level=level,
        dx=dx,
        dy=dy,
        x0=x0,
        y0=y0,
        ghost_values=ghost_vals,
    )


def _mask_chunk(
    mask,
    u,
    level: int,
    dx: float,
    dy: float,
    x0: float,
    y0: float,
    ghost,
) -> _QuadChunk:
    mask_np = _to_numpy(mask).astype(np.bool_, copy=False)
    idxs = np.argwhere(mask_np)
    if idxs.size == 0:
        return _empty_chunk()

    rows = idxs[:, 0].astype(np.int32, copy=False)
    cols = idxs[:, 1].astype(np.int32, copy=False)
    values = _to_numpy(u).astype(np.float32, copy=False)[rows, cols]

    if ghost is None:
        ghost_vals = np.zeros(rows.shape[0], dtype=np.int32)
    else:
        ghost_arr = _to_numpy(ghost)
        if ghost_arr.dtype == np.bool_:
            ghost_arr = ghost_arr.astype(np.int32, copy=False)
        ghost_vals = ghost_arr[rows, cols].astype(np.int32, copy=False)

    return _assemble_quads(
        rows,
        cols,
        values,
        level=level,
        dx=dx,
        dy=dy,
        x0=x0,
        y0=y0,
        ghost_values=ghost_vals,
    )

def write_unstructured_quads_vtu(
    path: str,
    cells: List[Tuple[Any, ...]],
    origin=(0.0, 0.0),
) -> None:
    """Write a single UnstructuredGrid (.vtu) where each selected cell across levels becomes a quad.

    cells: list of tuples (mask, u, level_id, dx, dy, x0, y0) per level.
    u is sampled cell-centered per cell; level_id is written to CellData 'level'.
    """
    chunks: List[_QuadChunk] = []

    for entry in cells:
        if not entry:
            continue
        first = entry[0]
        if hasattr(first, "interval_set"):
            if len(entry) < 7:
                raise ValueError("IntervalField entries must be (field, level, dx, dy, x0, y0, ghost_flag)")
            field, lvl, dx, dy, x0, y0, ghost_flag = entry[:7]
            chunk = _interval_field_chunk(field, int(lvl), float(dx), float(dy), float(x0), float(y0), int(ghost_flag))
            if chunk.cell_count:
                chunks.append(chunk)
            continue

        if len(entry) == 7:
            mask, u, lvl, dx, dy, x0, y0 = entry
            ghost = None
        else:
            mask, u, lvl, dx, dy, x0, y0, ghost = entry
        chunk = _mask_chunk(mask, u, int(lvl), float(dx), float(dy), float(x0), float(y0), ghost)
        if chunk.cell_count:
            chunks.append(chunk)

    if chunks:
        point_count = sum(chunk.points.shape[0] for chunk in chunks)
        cell_count = sum(chunk.cell_count for chunk in chunks)
        connectivity_size = sum(chunk.connectivity.size for chunk in chunks)

        P = np.empty((point_count, 3), dtype=np.float32)
        C = np.empty(connectivity_size, dtype=np.int32)
        O = np.empty(cell_count, dtype=np.int32)
        T = np.empty(cell_count, dtype=np.uint8)
        L = np.empty(cell_count, dtype=np.int32)
        U = np.empty(cell_count, dtype=np.float32)
        G = np.empty(cell_count, dtype=np.int32)

        point_offset = 0
        conn_offset = 0
        cell_offset = 0
        for chunk in chunks:
            pts = chunk.points
            cnt = chunk.connectivity
            offs = chunk.offsets
            cells = chunk.cell_count

            P[point_offset : point_offset + pts.shape[0]] = pts
            C[conn_offset : conn_offset + cnt.size] = cnt + point_offset
            O[cell_offset : cell_offset + cells] = offs + conn_offset
            T[cell_offset : cell_offset + cells] = chunk.types
            L[cell_offset : cell_offset + cells] = chunk.levels
            U[cell_offset : cell_offset + cells] = chunk.values
            G[cell_offset : cell_offset + cells] = chunk.ghosts

            point_offset += pts.shape[0]
            conn_offset += cnt.size
            cell_offset += cells
    else:
        P = np.zeros((0, 3), dtype=np.float32)
        C = np.zeros(0, dtype=np.int32)
        O = np.zeros(0, dtype=np.int32)
        T = np.zeros(0, dtype=np.uint8)
        L = np.zeros(0, dtype=np.int32)
        U = np.zeros(0, dtype=np.float32)
        G = np.zeros(0, dtype=np.int32)

    appended_arrays: List[Tuple[np.ndarray, int]] = []
    appended_offset = 0

    def _register_array(arr: np.ndarray) -> int:
        nonlocal appended_offset
        array = arr
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        offset = appended_offset
        appended_arrays.append((array, offset))
        appended_offset += 8 + array.nbytes
        return offset

    level_offset = _register_array(L)
    ghost_offset = _register_array(G)
    u_offset = _register_array(U)
    points_offset = _register_array(P)
    connectivity_offset = _register_array(C)
    offsets_offset = _register_array(O)
    types_offset = _register_array(T)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        def _write(line: str) -> None:
            f.write(line.encode("utf-8"))

        _write("<?xml version=\"1.0\"?>\n")
        _write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt64">\n')
        _write("  <UnstructuredGrid>\n")
        _write(f"    <Piece NumberOfPoints=\"{int(P.shape[0])}\" NumberOfCells=\"{int(L.size)}\">\n")
        _write("      <CellData Scalars=\"u\">\n")
        _write(f'        <DataArray type="Int32" Name="level" format="appended" offset="{level_offset}"/>\n')
        _write(f'        <DataArray type="Int32" Name="ghost_mask" format="appended" offset="{ghost_offset}"/>\n')
        _write(f'        <DataArray type="Float32" Name="u" format="appended" offset="{u_offset}"/>\n')
        _write("      </CellData>\n")
        _write("      <Points>\n")
        _write(f'        <DataArray type="Float32" NumberOfComponents="3" format="appended" offset="{points_offset}"/>\n')
        _write("      </Points>\n")
        _write("      <Cells>\n")
        _write(f'        <DataArray type="Int32" Name="connectivity" format="appended" offset="{connectivity_offset}"/>\n')
        _write(f'        <DataArray type="Int32" Name="offsets" format="appended" offset="{offsets_offset}"/>\n')
        _write(f'        <DataArray type="UInt8" Name="types" format="appended" offset="{types_offset}"/>\n')
        _write("      </Cells>\n")
        _write("    </Piece>\n")
        _write("  </UnstructuredGrid>\n")
        _write("  <AppendedData encoding=\"raw\">\n")
        f.write(b"_")
        for array, _ in appended_arrays:
            f.write(struct.pack("<Q", array.nbytes))
            f.write(array.tobytes(order="C"))
        f.write(b"\n")
        _write("  </AppendedData>\n")
        _write("</VTKFile>\n")


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
