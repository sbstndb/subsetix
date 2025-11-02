"""Plotting helpers for two-level AMR demos."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import cupy as cp
import numpy as np

from .geometry import interval_set_to_mask
from .simulation import AMRState

try:  # pragma: no cover - plotting optional
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib import animation as mpl_animation
except ImportError:  # pragma: no cover - plotting optional
    plt = None
    mcolors = None
    mpl_animation = None

Frame = Tuple[int, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]


def capture_frame(state: AMRState, step: int) -> Frame:
    refine_mask = state.refine_mask
    coarse_only = interval_set_to_mask(state.geometry.coarse_only, refine_mask.shape[1])
    return (
        step,
        cp.array(state.coarse, copy=True),
        cp.array(state.fine, copy=True),
        cp.array(refine_mask, copy=True),
        cp.array(coarse_only, copy=True),
    )


def render(state: AMRState, dt: float, frames: List[Frame], *, animate: bool, plot: bool, interval: int, loop: bool, save_animation: str | None) -> None:
    if not (plot or animate):
        return
    if plt is None or mcolors is None:
        raise RuntimeError("matplotlib is required for plotting/animation")

    refined_mask = state.refine_mask
    coarse_only_mask = interval_set_to_mask(state.geometry.coarse_only, refined_mask.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Two-level AMR (ratio=2, dt={dt:.5f})")

    fine_img = axes[0].imshow(
        cp.asnumpy(state.fine),
        origin="lower",
        cmap="turbo",
        animated=animate,
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title("Fine field")
    fig.colorbar(fine_img, ax=axes[0], fraction=0.046)

    overlay = np.zeros((*refined_mask.shape, 3), dtype=np.float32)
    overlay[..., 0] = cp.asnumpy(refined_mask)
    overlay[..., 1] = cp.asnumpy(coarse_only_mask)
    level_img = axes[1].imshow(overlay, origin="lower", animated=animate)
    axes[1].set_title("Level map (red = refine)")

    plt.tight_layout()

    if not animate:
        if plot:
            plt.show()
        return

    if mpl_animation is None:
        raise RuntimeError("matplotlib.animation required for animations")
    if not frames:
        raise RuntimeError("animation frames are empty")

    def _update(idx: int):
        (_, coarse_frame, fine_frame, refine_frame, coarse_only_frame) = frames[idx]
        fine_img.set_array(cp.asnumpy(fine_frame))
        overlay = np.zeros((*refine_frame.shape, 3), dtype=np.float32)
        overlay[..., 0] = cp.asnumpy(refine_frame)
        overlay[..., 1] = cp.asnumpy(coarse_only_frame)
        level_img.set_array(overlay)
        axes[1].set_xlabel(f"frame {idx+1}/{len(frames)}")
        return fine_img, level_img

    anim = mpl_animation.FuncAnimation(
        fig,
        _update,
        frames=len(frames),
        interval=max(20, interval),
        blit=True,
        repeat=loop,
    )

    if save_animation is not None:
        anim.save(save_animation, fps=1000.0 / max(1, interval))
        print(f"Saved animation to {save_animation}")
    elif plot:
        plt.show()

