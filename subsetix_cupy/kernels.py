# CuPy RawKernel helpers for interval operations.

from __future__ import annotations

from textwrap import dedent
from typing import Any, Dict, Tuple

_CACHE: Dict[int, Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]] = {}


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


_VERTICAL_DILATE_COUNT_CLAMP_SRC = dedent(
    r"""
    extern "C" __global__
    void vertical_dilate_count_clamp(const int* begin,
                                     const int* end,
                                     const int* row_offsets,
                                     int row_count,
                                     int halo,
                                     int* counts)
    {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        if (row >= row_count) {
            return;
        }

        const int MAX_NEIGHBORS = 64;

        int start_row = row - halo;
        if (start_row < 0) {
            start_row = 0;
        }
        int stop_row = row + halo + 1;
        if (stop_row > row_count) {
            stop_row = row_count;
        }
        int neighbor_count = stop_row - start_row;
        if (neighbor_count <= 0) {
            counts[row] = 0;
            return;
        }
        if (neighbor_count > MAX_NEIGHBORS) {
            counts[row] = 0;
            return;
        }

        int ptrs[MAX_NEIGHBORS];
        int limits[MAX_NEIGHBORS];
        for (int n = 0; n < neighbor_count; ++n) {
            int row_idx = start_row + n;
            ptrs[n] = row_offsets[row_idx];
            limits[n] = row_offsets[row_idx + 1];
        }

        bool has_active = false;
        int active_begin = 0;
        int active_end = 0;
        int total = 0;

        while (true) {
            int best = -1;
            int best_begin = 0;
            for (int n = 0; n < neighbor_count; ++n) {
                int p = ptrs[n];
                if (p < limits[n]) {
                    int candidate = begin[p];
                    if (best == -1 || candidate < best_begin) {
                        best = n;
                        best_begin = candidate;
                    }
                }
            }
            if (best == -1) {
                break;
            }

            int seg_begin = begin[ptrs[best]];
            int seg_end = end[ptrs[best]];
            ptrs[best] += 1;

            if (!has_active) {
                has_active = true;
                active_begin = seg_begin;
                active_end = seg_end;
                continue;
            }

            if (seg_begin <= active_end) {
                if (seg_end > active_end) {
                    active_end = seg_end;
                }
            } else {
                ++total;
                active_begin = seg_begin;
                active_end = seg_end;
            }
        }

        if (has_active) {
            ++total;
        }

        counts[row] = total;
    }
    """
)


_VERTICAL_DILATE_WRITE_CLAMP_SRC = dedent(
    r"""
    extern "C" __global__
    void vertical_dilate_write_clamp(const int* begin,
                                     const int* end,
                                     const int* row_offsets,
                                     const int* out_offsets,
                                     int row_count,
                                     int halo,
                                     int* out_begin,
                                     int* out_end)
    {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        if (row >= row_count) {
            return;
        }

        const int MAX_NEIGHBORS = 64;

        int start_row = row - halo;
        if (start_row < 0) {
            start_row = 0;
        }
        int stop_row = row + halo + 1;
        if (stop_row > row_count) {
            stop_row = row_count;
        }
        int neighbor_count = stop_row - start_row;
        if (neighbor_count <= 0 || neighbor_count > MAX_NEIGHBORS) {
            return;
        }

        int ptrs[MAX_NEIGHBORS];
        int limits[MAX_NEIGHBORS];
        for (int n = 0; n < neighbor_count; ++n) {
            int row_idx = start_row + n;
            ptrs[n] = row_offsets[row_idx];
            limits[n] = row_offsets[row_idx + 1];
        }

        int write_pos = out_offsets[row];
        bool has_active = false;
        int active_begin = 0;
        int active_end = 0;

        while (true) {
            int best = -1;
            int best_begin = 0;
            for (int n = 0; n < neighbor_count; ++n) {
                int p = ptrs[n];
                if (p < limits[n]) {
                    int candidate = begin[p];
                    if (best == -1 || candidate < best_begin) {
                        best = n;
                        best_begin = candidate;
                    }
                }
            }
            if (best == -1) {
                break;
            }

            int seg_begin = begin[ptrs[best]];
            int seg_end = end[ptrs[best]];
            ptrs[best] += 1;

            if (!has_active) {
                has_active = true;
                active_begin = seg_begin;
                active_end = seg_end;
                continue;
            }

            if (seg_begin <= active_end) {
                if (seg_end > active_end) {
                    active_end = seg_end;
                }
            } else {
                out_begin[write_pos] = active_begin;
                out_end[write_pos] = active_end;
                ++write_pos;
                active_begin = seg_begin;
                active_end = seg_end;
            }
        }

        if (has_active) {
            out_begin[write_pos] = active_begin;
            out_end[write_pos] = active_end;
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
    merge_count_kernel = cp_module.RawKernel(
        _MERGE_COUNT_SRC, "count_merged_segments", options=("--std=c++11",)
    )
    merge_write_kernel = cp_module.RawKernel(
        _MERGE_WRITE_SRC, "write_merged_segments", options=("--std=c++11",)
    )
    prolong_f32_kernel = cp_module.RawKernel(
        _PROLONG_FIELD_F32_SRC, "prolong_field_f32", options=("--std=c++11",)
    )
    prolong_f64_kernel = cp_module.RawKernel(
        _PROLONG_FIELD_F64_SRC, "prolong_field_f64", options=("--std=c++11",)
    )
    restrict_f32_kernel = cp_module.RawKernel(
        _RESTRICT_FIELD_F32_SRC, "restrict_field_f32", options=("--std=c++11",)
    )
    restrict_f64_kernel = cp_module.RawKernel(
        _RESTRICT_FIELD_F64_SRC, "restrict_field_f64", options=("--std=c++11",)
    )
    vertical_count_kernel = cp_module.RawKernel(
        _VERTICAL_DILATE_COUNT_CLAMP_SRC,
        "vertical_dilate_count_clamp",
        options=("--std=c++11",),
    )
    vertical_write_kernel = cp_module.RawKernel(
        _VERTICAL_DILATE_WRITE_CLAMP_SRC,
        "vertical_dilate_write_clamp",
        options=("--std=c++11",),
    )
    _CACHE[key] = (
        count_kernel,
        write_kernel,
        None,
        merge_count_kernel,
        merge_write_kernel,
        prolong_f32_kernel,
        prolong_f64_kernel,
        restrict_f32_kernel,
        restrict_f64_kernel,
        vertical_count_kernel,
        vertical_write_kernel,
    )
    return _CACHE[key]


_MERGE_COUNT_SRC = dedent(
    r"""
    extern "C" __global__
    void count_merged_segments(const int* begin,
                               const int* end,
                               const int* row_offsets,
                               int row_count,
                               int* counts)
    {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        if (row >= row_count) return;

        int start = row_offsets[row];
        int stop  = row_offsets[row + 1];

        if (start >= stop) {
            counts[row] = 0;
            return;
        }

        int active_begin = begin[start];
        int active_end   = end[start];
        int local_count = 1;

        for (int idx = start + 1; idx < stop; ++idx) {
            int seg_begin = begin[idx];
            int seg_end   = end[idx];
            if (seg_begin <= active_end) {
                if (seg_end > active_end) {
                    active_end = seg_end;
                }
            } else {
                ++local_count;
                active_begin = seg_begin;
                active_end   = seg_end;
            }
        }

        counts[row] = local_count;
    }
    """
)


_MERGE_WRITE_SRC = dedent(
    r"""
    extern "C" __global__
    void write_merged_segments(const int* begin,
                               const int* end,
                               const int* row_offsets,
                               const int* out_offsets,
                               int row_count,
                               int* out_begin,
                               int* out_end)
    {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        if (row >= row_count) return;

        int start = row_offsets[row];
        int stop  = row_offsets[row + 1];
        int write_pos = out_offsets[row];

        if (start >= stop) {
            return;
        }

        int active_begin = begin[start];
        int active_end   = end[start];

        for (int idx = start + 1; idx < stop; ++idx) {
            int seg_begin = begin[idx];
            int seg_end   = end[idx];
            if (seg_begin <= active_end) {
                if (seg_end > active_end) {
                    active_end = seg_end;
                }
            } else {
                out_begin[write_pos] = active_begin;
                out_end[write_pos]   = active_end;
                ++write_pos;
                active_begin = seg_begin;
                active_end   = seg_end;
            }
        }

        out_begin[write_pos] = active_begin;
        out_end[write_pos]   = active_end;
    }
    """
)


_PROLONG_FIELD_F32_SRC = dedent(
    r"""
    extern "C" __global__
    void prolong_field_f32(const int* coarse_offsets,
                           const int* fine_offsets,
                           const int* fine_interval_indices,
                           int ratio,
                           int interval_count,
                           const float* coarse_values,
                           float* fine_values)
    {
        int coarse_idx = blockIdx.x;
        if (coarse_idx >= interval_count) return;

        int coarse_begin = coarse_offsets[coarse_idx];
        int coarse_end = coarse_offsets[coarse_idx + 1];
        int width = coarse_end - coarse_begin;
        if (width <= 0) {
            return;
        }

        int base = coarse_idx * ratio;
        for (int delta = 0; delta < ratio; ++delta) {
            int fine_interval = fine_interval_indices[base + delta];
            int fine_begin = fine_offsets[fine_interval];
            int fine_end = fine_offsets[fine_interval + 1];
            int fine_width = fine_end - fine_begin;
            for (int k = threadIdx.x; k < fine_width; k += blockDim.x) {
                int src = coarse_begin + k / ratio;
                fine_values[fine_begin + k] = coarse_values[src];
            }
        }
    }
    """
)


_PROLONG_FIELD_F64_SRC = dedent(
    r"""
    extern "C" __global__
    void prolong_field_f64(const int* coarse_offsets,
                           const int* fine_offsets,
                           const int* fine_interval_indices,
                           int ratio,
                           int interval_count,
                           const double* coarse_values,
                           double* fine_values)
    {
        int coarse_idx = blockIdx.x;
        if (coarse_idx >= interval_count) return;

        int coarse_begin = coarse_offsets[coarse_idx];
        int coarse_end = coarse_offsets[coarse_idx + 1];
        int width = coarse_end - coarse_begin;
        if (width <= 0) {
            return;
        }

        int base = coarse_idx * ratio;
        for (int delta = 0; delta < ratio; ++delta) {
            int fine_interval = fine_interval_indices[base + delta];
            int fine_begin = fine_offsets[fine_interval];
            int fine_end = fine_offsets[fine_interval + 1];
            int fine_width = fine_end - fine_begin;
            for (int k = threadIdx.x; k < fine_width; k += blockDim.x) {
                int src = coarse_begin + k / ratio;
                fine_values[fine_begin + k] = coarse_values[src];
            }
        }
    }
    """
)


_RESTRICT_FIELD_F32_SRC = dedent(
    r"""
    extern "C" __global__
    void restrict_field_f32(const int* coarse_offsets,
                            const int* fine_offsets,
                            const int* fine_interval_indices,
                            int ratio,
                            int interval_count,
                            int reducer,
                            float norm,
                            const float* fine_values,
                            float* coarse_values)
    {
        int coarse_idx = blockIdx.x;
        if (coarse_idx >= interval_count) return;

        int coarse_begin = coarse_offsets[coarse_idx];
        int coarse_end = coarse_offsets[coarse_idx + 1];
        int width = coarse_end - coarse_begin;
        if (width <= 0) {
            return;
        }

        int base = coarse_idx * ratio;

        for (int k = threadIdx.x; k < width; k += blockDim.x) {
            float acc = 0.0f;
            float min_val = 0.0f;
            float max_val = 0.0f;
            bool have_val = false;

            for (int dy = 0; dy < ratio; ++dy) {
                int fine_interval = fine_interval_indices[base + dy];
                int fine_begin = fine_offsets[fine_interval];
                int fine_end = fine_offsets[fine_interval + 1];
                int start = fine_begin + k * ratio;
                for (int dx = 0; dx < ratio; ++dx) {
                    int idx = start + dx;
                    if (idx >= fine_end) break;
                    float v = fine_values[idx];
                    if (reducer == 2) {
                        if (!have_val || v < min_val) {
                            min_val = v;
                        }
                    } else if (reducer == 3) {
                        if (!have_val || v > max_val) {
                            max_val = v;
                        }
                    } else {
                        acc += v;
                    }
                    have_val = true;
                }
            }

            float result;
            if (reducer == 0) {
                result = acc * norm;
            } else if (reducer == 1) {
                result = acc;
            } else if (reducer == 2) {
                result = min_val;
            } else {
                result = max_val;
            }
            coarse_values[coarse_begin + k] = result;
        }
    }
    """
)


_RESTRICT_FIELD_F64_SRC = dedent(
    r"""
    extern "C" __global__
    void restrict_field_f64(const int* coarse_offsets,
                            const int* fine_offsets,
                            const int* fine_interval_indices,
                            int ratio,
                            int interval_count,
                            int reducer,
                            double norm,
                            const double* fine_values,
                            double* coarse_values)
    {
        int coarse_idx = blockIdx.x;
        if (coarse_idx >= interval_count) return;

        int coarse_begin = coarse_offsets[coarse_idx];
        int coarse_end = coarse_offsets[coarse_idx + 1];
        int width = coarse_end - coarse_begin;
        if (width <= 0) {
            return;
        }

        int base = coarse_idx * ratio;

        for (int k = threadIdx.x; k < width; k += blockDim.x) {
            double acc = 0.0;
            double min_val = 0.0;
            double max_val = 0.0;
            bool have_val = false;

            for (int dy = 0; dy < ratio; ++dy) {
                int fine_interval = fine_interval_indices[base + dy];
                int fine_begin = fine_offsets[fine_interval];
                int fine_end = fine_offsets[fine_interval + 1];
                int start = fine_begin + k * ratio;
                for (int dx = 0; dx < ratio; ++dx) {
                    int idx = start + dx;
                    if (idx >= fine_end) break;
                    double v = fine_values[idx];
                    if (reducer == 2) {
                        if (!have_val || v < min_val) {
                            min_val = v;
                        }
                    } else if (reducer == 3) {
                        if (!have_val || v > max_val) {
                            max_val = v;
                        }
                    } else {
                        acc += v;
                    }
                    have_val = true;
                }
            }

            double result;
            if (reducer == 0) {
                result = acc * norm;
            } else if (reducer == 1) {
                result = acc;
            } else if (reducer == 2) {
                result = min_val;
            } else {
                result = max_val;
            }
            coarse_values[coarse_begin + k] = result;
        }
    }
    """
)


# Prolongation/restriction kernels are used by multilevel field transfers to avoid host round-trips.
