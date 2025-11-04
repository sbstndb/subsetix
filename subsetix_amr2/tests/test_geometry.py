import unittest

from subsetix_amr2.fields import ActionField, Action
from subsetix_amr2.geometry import TwoLevelGeometry
from subsetix_cupy.expressions import IntervalSet, _REAL_CUPY


def _make_interval_set(cp_mod, height, spans, *, rows=None):
    begins = []
    ends = []
    row_offsets = [0]
    if rows is None:
        row_ids = list(range(height))
    else:
        row_ids = sorted(set(rows))
    extra = sorted(set(spans.keys()) - set(row_ids))
    row_ids.extend(extra)
    row_ids.sort()
    for row in row_ids:
        intervals = spans.get(row, ())
        for start, stop in intervals:
            begins.append(int(start))
            ends.append(int(stop))
        row_offsets.append(len(begins))
    begin_arr = cp_mod.asarray(begins, dtype=cp_mod.int32)
    end_arr = cp_mod.asarray(ends, dtype=cp_mod.int32)
    offsets_arr = cp_mod.asarray(row_offsets, dtype=cp_mod.int32)
    rows_arr = cp_mod.asarray(row_ids, dtype=cp_mod.int32)
    return IntervalSet(begin=begin_arr, end=end_arr, row_offsets=offsets_arr, rows=rows_arr)


def _empty_interval_set(cp_mod, height):
    zeros = cp_mod.zeros(0, dtype=cp_mod.int32)
    offsets = cp_mod.zeros(height + 1, dtype=cp_mod.int32)
    rows_arr = cp_mod.arange(height, dtype=cp_mod.int32)
    return IntervalSet(begin=zeros, end=zeros, row_offsets=offsets, rows=rows_arr)


def _assert_interval_equal(testcase, cp_mod, actual: IntervalSet, expected: IntervalSet):
    testcase.assertEqual(actual.row_offsets.size, expected.row_offsets.size)
    cp_mod.testing.assert_array_equal(actual.row_offsets, expected.row_offsets)
    cp_mod.testing.assert_array_equal(actual.begin, expected.begin)
    cp_mod.testing.assert_array_equal(actual.end, expected.end)
    cp_mod.testing.assert_array_equal(actual.rows_index(), expected.rows_index())


@unittest.skipUnless(_REAL_CUPY is not None, "CuPy backend with CUDA required")
class GeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cp = _REAL_CUPY

    def test_two_level_geometry_basic(self) -> None:
        refine_set = _make_interval_set(
            self.cp,
            4,
            {
                1: [(1, 3)],
                2: [(1, 3)],
            },
        )
        actions = ActionField.full_grid(4, 4, ratio=2, default=Action.KEEP)
        actions.set_from_interval_set(refine_set)

        geom = TwoLevelGeometry.from_action_field(actions)
        self.assertEqual(geom.ratio, 2)
        self.assertEqual(geom.width, 4)
        self.assertEqual(geom.height, 4)

        _assert_interval_equal(self, self.cp, geom.refine, refine_set)

        expected_coarse_only = _make_interval_set(
            self.cp,
            4,
            {
                0: [(0, 4)],
                1: [(0, 1), (3, 4)],
                2: [(0, 1), (3, 4)],
                3: [(0, 4)],
            },
        )
        _assert_interval_equal(self, self.cp, geom.coarse_only, expected_coarse_only)

        expected_fine = _make_interval_set(
            self.cp,
            8,
            {
                2: [(2, 6)],
                3: [(2, 6)],
                4: [(2, 6)],
                5: [(2, 6)],
            },
            rows=[2, 3, 4, 5],
        )
        _assert_interval_equal(self, self.cp, geom.fine, expected_fine)

    def test_with_action_field_reuses_workspace(self) -> None:
        initial = _make_interval_set(
            self.cp,
            4,
            {
                1: [(1, 3)],
                2: [(1, 3)],
            },
        )
        actions = ActionField.full_grid(4, 4, ratio=2)
        actions.set_from_interval_set(initial)
        geom = TwoLevelGeometry.from_action_field(actions)
        workspace_before = geom.workspace

        next_set = _make_interval_set(
            self.cp,
            4,
            {
                0: [(0, 2)],
                1: [(0, 2)],
            },
        )
        next_actions = ActionField.full_grid(4, 4, ratio=2)
        next_actions.set_from_interval_set(next_set)
        geom_next = geom.with_action_field(next_actions)
        self.assertIs(geom_next.workspace, workspace_before)
        _assert_interval_equal(self, self.cp, geom_next.refine, next_set)

    def test_dilation_modes(self) -> None:
        refine_set = _make_interval_set(self.cp, 5, {2: [(2, 3)]})
        actions = ActionField.full_grid(5, 5, ratio=2)
        actions.set_from_interval_set(refine_set)
        geom = TwoLevelGeometry.from_action_field(actions)

        vn = geom.dilate_refine(halo=1, mode="von_neumann")
        expected_vn = _make_interval_set(
            self.cp,
            5,
            {
                1: [(2, 3)],
                2: [(1, 4)],
                3: [(2, 3)],
            },
        )
        _assert_interval_equal(self, self.cp, vn.refine, expected_vn)

        moore = geom.dilate_refine(halo=1, mode="moore")
        expected_moore = _make_interval_set(
            self.cp,
            5,
            {
                1: [(1, 4)],
                2: [(1, 4)],
                3: [(1, 4)],
            },
        )
        _assert_interval_equal(self, self.cp, moore.refine, expected_moore)

    def test_empty_geometry(self) -> None:
        actions = ActionField.full_grid(3, 3, ratio=2)
        actions.set_from_interval_set(_empty_interval_set(self.cp, 3))
        geom = TwoLevelGeometry.from_action_field(actions)
        empty = _empty_interval_set(self.cp, 3)
        _assert_interval_equal(self, self.cp, geom.refine, empty)
        _assert_interval_equal(self, self.cp, geom.coarse_only, _make_interval_set(self.cp, 3, {0: [(0, 3)], 1: [(0, 3)], 2: [(0, 3)]}))
