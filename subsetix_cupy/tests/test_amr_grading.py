import unittest

import cupy as cp


def _init_condition(W: int, H: int, kind: str = "sharp", amp: float = 1.0) -> cp.ndarray:
    xx = cp.linspace(0.0, 1.0, W, dtype=cp.float32)
    yy = cp.linspace(0.0, 1.0, H, dtype=cp.float32)
    X, Y = cp.meshgrid(xx, yy)
    if kind == "gauss":
        g1 = cp.exp(-((X - 0.30) ** 2 + (Y - 0.65) ** 2) / (2 * 0.05**2))
        g2 = cp.exp(-((X - 0.70) ** 2 + (Y - 0.30) ** 2) / (2 * 0.06**2))
        ridge = cp.maximum(0.0, 1.0 - 25.0 * (Y - 0.5) ** 2)
        u = 0.9 * g1 + 0.7 * g2 + 0.3 * ridge
    elif kind == "disk":
        d1 = ((X - 0.35) ** 2 + (Y - 0.6) ** 2) <= (0.12 ** 2)
        d2 = ((X - 0.68) ** 2 + (Y - 0.32) ** 2) <= (0.10 ** 2)
        u = (d1 | d2).astype(cp.float32)
    elif kind == "square":
        s1 = (cp.abs(X - 0.30) <= 0.10) & (cp.abs(Y - 0.65) <= 0.10)
        s2 = (cp.abs(X - 0.70) <= 0.10) & (cp.abs(Y - 0.30) <= 0.12)
        u = (s1 | s2).astype(cp.float32)
    elif kind == "edge":
        u = (X + Y < 0.9).astype(cp.float32)
    else:
        g1 = cp.exp(-((X - 0.30) ** 2 + (Y - 0.65) ** 2) / (2 * 0.03**2))
        g2 = cp.exp(-((X - 0.70) ** 2 + (Y - 0.30) ** 2) / (2 * 0.035**2))
        ridge = cp.maximum(0.0, 1.0 - 60.0 * (Y - 0.5) ** 2)
        u = 1.2 * g1 + 1.0 * g2 + 0.25 * ridge
    if amp != 1.0:
        u = (u * amp).astype(cp.float32, copy=False)
    return u.astype(cp.float32, copy=False)


def _grad_mag(u: cp.ndarray) -> cp.ndarray:
    gy, gx = cp.gradient(u)
    return cp.sqrt(gx * gx + gy * gy)


def _prolong_repeat(arr: cp.ndarray, ratio: int) -> cp.ndarray:
    result = cp.repeat(cp.repeat(arr, ratio, axis=0), ratio, axis=1)
    if arr.dtype == cp.bool_:
        return result.astype(cp.bool_, copy=False)
    return result.astype(arr.dtype, copy=False)


def _dilate_mo(mask: cp.ndarray, wrap: bool) -> cp.ndarray:
    if wrap:
        shifts = [
            (0, 0),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        acc = cp.zeros_like(mask, dtype=cp.bool_)
        for dy, dx in shifts:
            acc |= cp.roll(cp.roll(mask, dy, axis=0), dx, axis=1)
        return acc
    H, W = mask.shape
    acc = cp.zeros_like(mask, dtype=cp.bool_)
    acc |= mask
    if H > 1:
        acc[1:, :] |= mask[:-1, :]
        acc[:-1, :] |= mask[1:, :]
    if W > 1:
        acc[:, 1:] |= mask[:, :-1]
        acc[:, :-1] |= mask[:, 1:]
    if H > 1 and W > 1:
        acc[1:, 1:] |= mask[:-1, :-1]
        acc[1:, :-1] |= mask[:-1, 1:]
        acc[:-1, 1:] |= mask[1:, :-1]
        acc[:-1, :-1] |= mask[1:, 1:]
    return acc


def _erode_mo(mask: cp.ndarray, wrap: bool) -> cp.ndarray:
    if wrap:
        shifts = [
            (0, 0),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        acc = cp.ones_like(mask, dtype=cp.bool_)
        for dy, dx in shifts:
            acc &= cp.roll(cp.roll(mask, dy, axis=0), dx, axis=1)
        return acc
    H, W = mask.shape
    acc = mask.astype(cp.bool_, copy=True)
    if H > 1:
        up = cp.zeros_like(mask); up[1:, :] = mask[:-1, :]
        down = cp.zeros_like(mask); down[:-1, :] = mask[1:, :]
        acc &= up
        acc &= down
    if W > 1:
        left = cp.zeros_like(mask); left[:, 1:] = mask[:, :-1]
        right = cp.zeros_like(mask); right[:, :-1] = mask[:, 1:]
        acc &= left
        acc &= right
    if H > 1 and W > 1:
        up_left = cp.zeros_like(mask); up_left[1:, 1:] = mask[:-1, :-1]
        up_right = cp.zeros_like(mask); up_right[1:, :-1] = mask[:-1, 1:]
        down_left = cp.zeros_like(mask); down_left[:-1, 1:] = mask[1:, :-1]
        down_right = cp.zeros_like(mask); down_right[:-1, :-1] = mask[1:, 1:]
        acc &= up_left
        acc &= up_right
        acc &= down_left
        acc &= down_right
    return acc


def _hysteresis_mask(
    gradient: cp.ndarray,
    frac_high: float,
    frac_low: float,
    previous: cp.ndarray | None,
) -> cp.ndarray:
    frac_high = max(0.0, min(1.0, float(frac_high)))
    frac_low = max(0.0, min(frac_high, float(frac_low)))
    if frac_high <= 0.0:
        return cp.zeros_like(gradient, dtype=cp.bool_)
    flat = gradient.ravel()
    t_high = cp.percentile(flat, (1.0 - frac_high) * 100.0)
    high = gradient >= t_high
    if previous is None or frac_low <= 0.0:
        return high
    t_low = cp.percentile(flat, (1.0 - frac_low) * 100.0)
    low = gradient >= t_low
    return high | (previous.astype(cp.bool_) & low)


def _no_contact_moore(level2_fine: cp.ndarray, level1_fine: cp.ndarray) -> bool:
    """Return True if L2 has no 8-neigh contact with cells outside L1 on the fine grid."""
    l2 = level2_fine.astype(cp.bool_)
    l1 = level1_fine.astype(cp.bool_)
    outside_l1 = ~l1
    halo_l2 = _dilate_mo(l2, wrap=False)
    return bool((halo_l2 & outside_l1).sum().item() == 0)


class AMRGrading3LevelTest(unittest.TestCase):
    def test_proper_nesting_initial(self):
        W = 48
        R = 2
        refine_frac = 0.12
        hyst = 0.5

        u0 = _init_condition(W, W, kind="sharp", amp=1.0)
        u1 = _prolong_repeat(u0, R)

        g0 = _grad_mag(u0)
        refine0 = _hysteresis_mask(g0, refine_frac, refine_frac * hyst, None)
        L1_base = _prolong_repeat(refine0.astype(cp.uint8), R).astype(cp.bool_)

        # Expand L1 with a one-cell ring (mid grid), and force coarse parents
        L1_expanded = _dilate_mo(L1_base, wrap=False)
        coarse_force_from_L1ring = L1_expanded.reshape(W, R, W, R).any(axis=(1, 3))
        refine0 = refine0 | coarse_force_from_L1ring
        L1_mask = _prolong_repeat(refine0.astype(cp.uint8), R).astype(cp.bool_)

        # Gate L2 by eroded L1
        g1 = _grad_mag(u1)
        refine1_mid = _hysteresis_mask(g1, refine_frac, refine_frac * hyst, None)
        refine1_mid = refine1_mid & _erode_mo(L1_mask, wrap=False)

        # Child forces parent (mid->coarse)
        coarse_force = refine1_mid.reshape(W, R, W, R).any(axis=(1, 3))
        refine0 = refine0 | coarse_force
        L1_mask = _prolong_repeat(refine0.astype(cp.uint8), R).astype(cp.bool_)

        L2_mask = _prolong_repeat(refine1_mid.astype(cp.uint8), R).astype(cp.bool_)
        L1_fine = _prolong_repeat(L1_mask.astype(cp.uint8), R).astype(cp.bool_)

        # Assertions
        # 1) L2 subset of L1 on fine grid
        self.assertTrue(bool(cp.all(L2_mask <= L1_fine)))
        # 2) No 8-neigh contact 2↔0 (i.e., L2 touching outside-L1)
        self.assertTrue(_no_contact_moore(L2_mask, L1_fine))
        # 3) All coarse parents of any L2 child remain refined at L1
        self.assertTrue(bool(cp.all(coarse_force <= refine0)))

    def test_proper_nesting_after_regrid(self):
        # Simulate a regrid pass with updated u0/u1 by small perturbation
        W = 32
        R = 2
        refine_frac = 0.10
        hyst = 0.5

        u0 = _init_condition(W, W, kind="edge", amp=1.0)
        u1 = _prolong_repeat(u0, R)

        g0 = _grad_mag(u0)
        refine0 = _hysteresis_mask(g0, refine_frac, refine_frac * hyst, None)
        L1_base = _prolong_repeat(refine0.astype(cp.uint8), R).astype(cp.bool_)
        L1_expanded = _dilate_mo(L1_base, wrap=False)
        refine0 = refine0 | L1_expanded.reshape(W, R, W, R).any(axis=(1, 3))
        L1_mask = _prolong_repeat(refine0.astype(cp.uint8), R).astype(cp.bool_)

        # First L2
        g1 = _grad_mag(u1)
        refine1_mid = _hysteresis_mask(g1, refine_frac, refine_frac * hyst, None)
        refine1_mid = refine1_mid & _erode_mo(L1_mask, wrap=False)
        refine0 = refine0 | refine1_mid.reshape(W, R, W, R).any(axis=(1, 3))
        L1_mask = _prolong_repeat(refine0.astype(cp.uint8), R).astype(cp.bool_)

        # Mimic evolution: shift field slightly and regrid again
        u0b = cp.roll(u0, 1, axis=1)
        u1b = _prolong_repeat(u0b, R)
        g0b = _grad_mag(u0b)
        refine0b = _hysteresis_mask(g0b, refine_frac, refine_frac * hyst, refine0)
        L1_base_b = _prolong_repeat(refine0b.astype(cp.uint8), R).astype(cp.bool_)
        L1_ring_b = _dilate_mo(L1_base_b, wrap=False)
        refine0b = refine0b | L1_ring_b.reshape(W, R, W, R).any(axis=(1, 3))
        L1_mask_b = _prolong_repeat(refine0b.astype(cp.uint8), R).astype(cp.bool_)

        g1b = _grad_mag(u1b)
        refine1_mid_b = _hysteresis_mask(g1b, refine_frac, refine_frac * hyst, refine1_mid)
        refine1_mid_b = refine1_mid_b & _erode_mo(L1_mask_b, wrap=False)
        refine0b = refine0b | refine1_mid_b.reshape(W, R, W, R).any(axis=(1, 3))
        L1_mask_b = _prolong_repeat(refine0b.astype(cp.uint8), R).astype(cp.bool_)

        L2_mask_b = _prolong_repeat(refine1_mid_b.astype(cp.uint8), R).astype(cp.bool_)
        L1_fine_b = _prolong_repeat(L1_mask_b.astype(cp.uint8), R).astype(cp.bool_)

        self.assertTrue(bool(cp.all(L2_mask_b <= L1_fine_b)))
        self.assertTrue(_no_contact_moore(L2_mask_b, L1_fine_b))

    def test_block_uniformity_L1_and_L2(self):
        W = 40
        R = 2
        refine_frac = 0.15
        hyst = 0.5

        u0 = _init_condition(W, W, kind="sharp", amp=1.0)
        u1 = _prolong_repeat(u0, R)

        # L0 -> L1 with ring expansion and parent forcing
        g0 = _grad_mag(u0)
        refine0 = _hysteresis_mask(g0, refine_frac, refine_frac * hyst, None)
        L1_base = _prolong_repeat(refine0.astype(cp.uint8), R).astype(cp.bool_)
        L1_ring = _dilate_mo(L1_base, wrap=False)
        refine0 = refine0 | L1_ring.reshape(W, R, W, R).any(axis=(1, 3))
        L1_mask = _prolong_repeat(refine0.astype(cp.uint8), R).astype(cp.bool_)

        # L1 -> L2 gated by eroded L1 and child->parent forcing
        g1 = _grad_mag(u1)
        refine1_mid = _hysteresis_mask(g1, refine_frac, refine_frac * hyst, None)
        refine1_mid = refine1_mid & _erode_mo(L1_mask, wrap=False)
        refine0 = refine0 | refine1_mid.reshape(W, R, W, R).any(axis=(1, 3))
        L1_mask = _prolong_repeat(refine0.astype(cp.uint8), R).astype(cp.bool_)

        L2_mask = _prolong_repeat(refine1_mid.astype(cp.uint8), R).astype(cp.bool_)

        # Block uniformity: each coarse parent block (R×R mid) must be uniform wrt L1
        L1_blocks = L1_mask.reshape(W, R, W, R)
        any_child = L1_blocks.any(axis=(1, 3))
        all_child = L1_blocks.all(axis=(1, 3))
        # Uniform if either all or none; mixed if any != all
        self.assertTrue(bool(cp.all(any_child == all_child)))
        # And parent flag matches block state
        self.assertTrue(bool(cp.all(any_child == refine0)))

        # Block uniformity: each mid parent block (R×R fine) must be uniform wrt L2
        Hm, Wm = W * R, W * R
        L2_blocks = L2_mask.reshape(Hm, R, Wm, R)
        any_fchild = L2_blocks.any(axis=(1, 3))
        all_fchild = L2_blocks.all(axis=(1, 3))
        self.assertTrue(bool(cp.all(any_fchild == all_fchild)))
        # And equals refine1_mid (definition of prolongation)
        self.assertTrue(bool(cp.all(any_fchild == refine1_mid)))


if __name__ == "__main__":
    unittest.main()
