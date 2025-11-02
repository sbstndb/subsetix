"""
Matplotlib helpers shared across the CuPy demos.

Most demos need to convert sparse interval sets into dense masks or
per-cell rectangles.  Centralising the helpers keeps the demos compact
and makes styling consistent.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons

from .expressions import IntervalSet, _require_cupy
from .multilevel import prolong_set


def intervals_to_mask(interval_set: IntervalSet, width: int) -> np.ndarray:
    """
    Render an IntervalSet onto a dense binary mask of shape (rows, width).
    """

    cp = _require_cupy()
    offsets = cp.asnumpy(interval_set.row_offsets)
    begin = cp.asnumpy(interval_set.begin)
    end = cp.asnumpy(interval_set.end)
    rows = offsets.size - 1
    mask = np.zeros((rows, width), dtype=np.uint8)
    if rows == 0 or begin.size == 0:
        return mask

    counts = np.diff(offsets)
    row_indices = np.repeat(np.arange(rows, dtype=np.int32), counts)
    for row, start, stop in zip(row_indices, begin, end, strict=False):
        mask[row, start:stop] = 1
    return mask


def build_cell_rectangles(
    interval_set: IntervalSet | None,
    base_dim: int,
    target_ratio: int,
    *,
    offset_x: float = 0.0,
) -> list[Rectangle]:
    """
    Produce matplotlib rectangles representing every active cell.
    """

    if interval_set is None:
        return []
    cp = _require_cupy()
    row_count = interval_set.row_count
    if row_count == 0:
        return []
    actual_ratio = max(1, row_count // base_dim)
    if target_ratio % actual_ratio != 0:
        raise ValueError("Ratios must divide the target ratio")
    scale = target_ratio // actual_ratio
    refined = prolong_set(interval_set, scale) if scale > 1 else interval_set
    begin = cp.asnumpy(refined.begin)
    end = cp.asnumpy(refined.end)
    offsets = cp.asnumpy(refined.row_offsets)
    cell_size = 1.0 / target_ratio
    patches: list[Rectangle] = []
    for row in range(refined.row_count):
        start = offsets[row]
        stop = offsets[row + 1]
        if start == stop:
            continue
        y = row / target_ratio
        for x0, x1 in zip(begin[start:stop], end[start:stop], strict=False):
            for x in range(x0, x1):
                patches.append(Rectangle((x / target_ratio + offset_x, y), cell_size, cell_size))
    return patches


def make_cell_collection(
    interval_set: IntervalSet | None,
    base_dim: int,
    target_ratio: int,
    *,
    facecolor: str,
    edgecolor: str = "k",
    linewidth: float = 0.3,
    offset_x: float = 0.0,
) -> PatchCollection:
    patches = build_cell_rectangles(interval_set, base_dim, target_ratio, offset_x=offset_x)
    return PatchCollection(patches, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)


def field_collection_from_dense(
    interval_set: IntervalSet | None,
    dense: np.ndarray,
    base_dim: int,
    target_ratio: int,
    cmap,
    *,
    edgecolor: str = "k",
    linewidth: float = 0.15,
    offset_x: float = 0.0,
) -> PatchCollection:
    if interval_set is None:
        return PatchCollection([])

    cp = _require_cupy()
    offsets = cp.asnumpy(interval_set.row_offsets)
    begin = cp.asnumpy(interval_set.begin)
    end = cp.asnumpy(interval_set.end)
    rows = offsets.size - 1
    size = 1.0 / max(1, target_ratio)
    patches: list[Rectangle] = []
    values: list[float] = []
    for row in range(rows):
        start = offsets[row]
        stop = offsets[row + 1]
        y = row / max(1, target_ratio)
        for x0, x1 in zip(begin[start:stop], end[start:stop], strict=False):
            for x in range(x0, x1):
                patches.append(Rectangle((x / target_ratio + offset_x, y), size, size))
                values.append(dense[row, x])
    collection = PatchCollection(patches, edgecolor=edgecolor, linewidth=linewidth, cmap=cmap)
    if values:
        collection.set_array(np.asarray(values))
    return collection


def setup_cell_axes(ax, width: int, height: int, *, title: str | None = None, offset: float = 0.0) -> None:
    ax.set_xlim(offset, offset + width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    ax.axis("off")


def add_toggle_row(
    fig,
    collections: Sequence[PatchCollection],
    labels: Sequence[str],
    *,
    start: tuple[float, float] = (0.05, 0.04),
    width: float = 0.08,
    height: float = 0.03,
    spacing: float = 0.02,
) -> Iterable[CheckButtons]:
    """
    Attach a row of CheckButtons below the figure to toggle collections.
    """

    checkboxes = []
    for idx, (label, coll) in enumerate(zip(labels, collections, strict=False)):
        x = start[0] + idx * (width + spacing)
        ax_box = fig.add_axes([x, start[1], width, height], facecolor="lightgoldenrodyellow")
        checkbox = CheckButtons(ax_box, [label], [coll.get_visible()])

        def _toggle(_label: str, target=coll):
            target.set_visible(not target.get_visible())
            fig.canvas.draw_idle()

        checkbox.on_clicked(_toggle)
        checkboxes.append(checkbox)
    return checkboxes


def plot_cell_layout_from_sets(
    level_sets: Sequence[IntervalSet | None],
    ratios: Sequence[int],
    base_width: int,
    *,
    labels: Sequence[str] | None = None,
) -> None:
    base_colors = ["#bdbdbd", "#ffb347", "#ff6961", "#8fd694", "#7fa2ff", "#c18aff"]
    if labels is None:
        labels = [f"Level {idx}" for idx in range(len(level_sets))]

    finest_ratio = max(max(1, r) for r in ratios) if ratios else 1
    fig, ax = plt.subplots(figsize=(7, 7))
    collections = []
    for level, (interval_set, ratio) in enumerate(zip(level_sets, ratios, strict=False)):
        collection = make_cell_collection(
            interval_set,
            base_width,
            finest_ratio,
            facecolor=base_colors[level % len(base_colors)],
        )
        collection.set_visible(True)
        ax.add_collection(collection)
        collections.append(collection)

    ax.set_xlim(0, base_width)
    ax.set_ylim(0, base_width)
    ax.set_aspect("equal")
    ax.set_title("Refinement levels (per-cell view)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks(np.arange(0, base_width + 1, max(1, base_width // 8)))
    ax.set_yticks(np.arange(0, base_width + 1, max(1, base_width // 8)))

    fig.subplots_adjust(bottom=0.12)
    add_toggle_row(fig, collections, labels)
    plt.show()
