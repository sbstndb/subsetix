#ifndef INTERVAL_MERGE_HPP
#define INTERVAL_MERGE_HPP

#include <algorithm>
#include <utility>
#include <vector>

inline std::vector<std::pair<int, int>> mergeIntervals(const int* lhs_begin,
                                                       const int* lhs_end,
                                                       int lhs_count,
                                                       const int* rhs_begin,
                                                       const int* rhs_end,
                                                       int rhs_count) {
    std::vector<std::pair<int, int>> merged;
    merged.reserve(lhs_count + rhs_count);

    int i = 0;
    int j = 0;

    auto push_interval = [&](int start, int end) {
        if (start >= end) {
            return;
        }
        if (!merged.empty() && start <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, end);
        } else {
            merged.emplace_back(start, end);
        }
    };

    while (i < lhs_count || j < rhs_count) {
        bool take_lhs = (j >= rhs_count) || (i < lhs_count && lhs_begin[i] <= rhs_begin[j]);
        int start = 0;
        int end = 0;
        if (take_lhs) {
            start = lhs_begin[i];
            end = lhs_end[i];
            ++i;
        } else {
            start = rhs_begin[j];
            end = rhs_end[j];
            ++j;
        }
        push_interval(start, end);
    }

    return merged;
}

#endif // INTERVAL_MERGE_HPP
