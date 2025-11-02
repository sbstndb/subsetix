# CuPy RawKernel helpers for interval operations.

from __future__ import annotations

from textwrap import dedent
from typing import Any, Dict, Tuple

_CACHE: Dict[int, Tuple[Any, Any]] = {}


_COUNT_SRC = dedent(
    r"""
    extern "C" __global__
    void count_segments(const int* lhs_begin,
                        const int* lhs_end,
                        const int* lhs_offsets,
                        const int* rhs_begin,
                        const int* rhs_end,
                        const int* rhs_offsets,
                        int row_count,
                        int op,
                        int* counts)
    {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        if (row >= row_count) return;

        int lhs_start = lhs_offsets[row];
        int lhs_stop = lhs_offsets[row + 1];
        int rhs_start = rhs_offsets[row];
        int rhs_stop = rhs_offsets[row + 1];

        int i = lhs_start;
        int j = rhs_start;
        int count = 0;

        if (op == 0) {
            bool has_active = false;
            int active_start = 0;
            int active_end = 0;
            while (i < lhs_stop || j < rhs_stop) {
                bool take_lhs = (j >= rhs_stop) ||
                                (i < lhs_stop && lhs_begin[i] <= rhs_begin[j]);
                int seg_start = take_lhs ? lhs_begin[i] : rhs_begin[j];
                int seg_end   = take_lhs ? lhs_end[i]   : rhs_end[j];
                if (!has_active) {
                    has_active = true;
                    active_start = seg_start;
                    active_end = seg_end;
                } else if (seg_start <= active_end) {
                    if (seg_end > active_end) {
                        active_end = seg_end;
                    }
                } else {
                    ++count;
                    active_start = seg_start;
                    active_end = seg_end;
                }
                if (take_lhs) {
                    ++i;
                } else {
                    ++j;
                }
            }
            if (has_active) {
                ++count;
            }
        } else if (op == 1) {
            while (i < lhs_stop && j < rhs_stop) {
                int a_start = lhs_begin[i];
                int a_end = lhs_end[i];
                int b_start = rhs_begin[j];
                int b_end = rhs_end[j];
                int seg_start = a_start > b_start ? a_start : b_start;
                int seg_end = a_end < b_end ? a_end : b_end;
                if (seg_start < seg_end) {
                    ++count;
                }
                if (a_end <= b_end) {
                    ++i;
                } else {
                    ++j;
                }
            }
        } else {
            while (i < lhs_stop) {
                int a_start = lhs_begin[i];
                int a_end = lhs_end[i];
                int cursor = a_start;
                while (cursor < a_end) {
                    while (j < rhs_stop && rhs_end[j] <= cursor) {
                        ++j;
                    }
                    if (j >= rhs_stop || rhs_begin[j] >= a_end) {
                        if (cursor < a_end) {
                            ++count;
                        }
                        break;
                    }
                    if (rhs_begin[j] > cursor) {
                        int seg_end = rhs_begin[j] < a_end ? rhs_begin[j] : a_end;
                        if (seg_end > cursor) {
                            ++count;
                        }
                    }
                    if (rhs_end[j] <= cursor) {
                        ++j;
                    } else {
                        cursor = rhs_end[j];
                    }
                }
                ++i;
            }
        }
        counts[row] = count;
    }
    """
)


_WRITE_SRC = dedent(
    r"""
    extern "C" __global__
    void write_segments(const int* lhs_begin,
                        const int* lhs_end,
                        const int* lhs_offsets,
                        const int* rhs_begin,
                        const int* rhs_end,
                        const int* rhs_offsets,
                        const int* offsets,
                        int row_count,
                        int op,
                        int* out_begin,
                        int* out_end)
    {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        if (row >= row_count) return;

        int lhs_start = lhs_offsets[row];
        int lhs_stop = lhs_offsets[row + 1];
        int rhs_start = rhs_offsets[row];
        int rhs_stop = rhs_offsets[row + 1];

        int i = lhs_start;
        int j = rhs_start;
        int write_pos = offsets[row];

        if (op == 0) {
            bool has_active = false;
            int active_start = 0;
            int active_end = 0;
            while (i < lhs_stop || j < rhs_stop) {
                bool take_lhs = (j >= rhs_stop) ||
                                (i < lhs_stop && lhs_begin[i] <= rhs_begin[j]);
                int seg_start = take_lhs ? lhs_begin[i] : rhs_begin[j];
                int seg_end   = take_lhs ? lhs_end[i]   : rhs_end[j];
                if (!has_active) {
                    has_active = true;
                    active_start = seg_start;
                    active_end = seg_end;
                } else if (seg_start <= active_end) {
                    if (seg_end > active_end) {
                        active_end = seg_end;
                    }
                } else {
                    out_begin[write_pos] = active_start;
                    out_end[write_pos] = active_end;
                    ++write_pos;
                    active_start = seg_start;
                    active_end = seg_end;
                }
                if (take_lhs) {
                    ++i;
                } else {
                    ++j;
                }
            }
            if (has_active) {
                out_begin[write_pos] = active_start;
                out_end[write_pos] = active_end;
            }
        } else if (op == 1) {
            while (i < lhs_stop && j < rhs_stop) {
                int a_start = lhs_begin[i];
                int a_end = lhs_end[i];
                int b_start = rhs_begin[j];
                int b_end = rhs_end[j];
                int seg_start = a_start > b_start ? a_start : b_start;
                int seg_end = a_end < b_end ? a_end : b_end;
                if (seg_start < seg_end) {
                    out_begin[write_pos] = seg_start;
                    out_end[write_pos] = seg_end;
                    ++write_pos;
                }
                if (a_end <= b_end) {
                    ++i;
                } else {
                    ++j;
                }
            }
        } else {
            while (i < lhs_stop) {
                int a_start = lhs_begin[i];
                int a_end = lhs_end[i];
                int cursor = a_start;
                while (cursor < a_end) {
                    while (j < rhs_stop && rhs_end[j] <= cursor) {
                        ++j;
                    }
                    if (j >= rhs_stop || rhs_begin[j] >= a_end) {
                        if (cursor < a_end) {
                            out_begin[write_pos] = cursor;
                            out_end[write_pos] = a_end;
                            ++write_pos;
                        }
                        break;
                    }
                    if (rhs_begin[j] > cursor) {
                        int seg_end = rhs_begin[j] < a_end ? rhs_begin[j] : a_end;
                        if (seg_end > cursor) {
                            out_begin[write_pos] = cursor;
                            out_end[write_pos] = seg_end;
                            ++write_pos;
                        }
                    }
                    if (rhs_end[j] <= cursor) {
                        ++j;
                    } else {
                        cursor = rhs_end[j];
                    }
                }
                ++i;
            }
        }
    }
    """
)


def get_kernels(cp_module):
    """
    Compile and cache the RawKernels for the provided CuPy module instance.

    Parameters
    ----------
    cp_module : module
        CuPy module (imported) used to instantiate RawKernels.
    """

    key = id(cp_module)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    count_kernel = cp_module.RawKernel(
        _COUNT_SRC, "count_segments", options=("--std=c++11",)
    )
    write_kernel = cp_module.RawKernel(
        _WRITE_SRC, "write_segments", options=("--std=c++11",)
    )
    _CACHE[key] = (count_kernel, write_kernel)
    return _CACHE[key]
