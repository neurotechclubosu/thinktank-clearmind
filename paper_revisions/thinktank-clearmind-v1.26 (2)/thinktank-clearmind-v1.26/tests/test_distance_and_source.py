"""Tests for the corrected inverse-distance interpolation and the sphere
forward model used by the consistency analysis."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.eeg.EEG_DistanceFunc import DistanceFunc
from src.analysis.source_space import sphere_potential, electrode_positions


def test_nearer_point_weighted_more_than_farther_point():
    # Two neighbours at distance 1 and 10 with very different values: with inverse
    # -distance weighting the result must sit closer to the *near* neighbour's value.
    rng = np.random.default_rng(0)
    val = DistanceFunc([0, 0, 0], [[1, 0, 0], [10, 0, 0]], [0.0, 100.0], 2, rng=rng)
    assert val < 50, val  # closer to 0 (near point) than 100 (far point)


def test_exact_match_at_electrode_is_centred_on_that_value():
    # DistanceFunc always adds Gaussian jitter (a rendering device, see the
    # module docstring); the *centre* of the distribution is the exact value
    # when the selected point coincides with a neighbour.
    vals = [DistanceFunc([1, 0, 0], [[1, 0, 0], [5, 0, 0]], [42.0, 0.0], 2,
                         rng=np.random.default_rng(s)) for s in range(500)]
    assert abs(np.mean(vals) - 42.0) < 3.0, np.mean(vals)


def test_manhattan_vs_euclidean_differ_off_axis():
    pts = [[1, 1, 0], [2, 0, 0]]
    rng = np.random.default_rng(0)
    e = DistanceFunc([0, 0, 0], pts, [10.0, 20.0], 2, metric="euclidean", rng=rng)
    rng = np.random.default_rng(0)
    m = DistanceFunc([0, 0, 0], pts, [10.0, 20.0], 2, metric="manhattan", rng=rng)
    assert e != m


def test_seeded_rng_is_reproducible():
    a = DistanceFunc([0, 0, 0], [[1, 0, 0], [2, 0, 0]], [1.0, 2.0], 2,
                     rng=np.random.default_rng(5))
    b = DistanceFunc([0, 0, 0], [[1, 0, 0], [2, 0, 0]], [1.0, 2.0], 2,
                     rng=np.random.default_rng(5))
    assert a == b


def test_sphere_potential_matches_centre_dipole_closed_form():
    for theta in (0.3, 1.0, 2.0):
        e = np.array([[np.sin(theta), 0, np.cos(theta)]])
        v = sphere_potential([0, 0, 1e-9], 1.0, 0.0, e)
        assert abs(v[0] - 3 * np.cos(theta)) < 1e-6


def test_sphere_potential_converges_as_dipole_approaches_surface():
    # As a radial dipole approaches the boundary directly under an electrode,
    # the series must converge (not diverge or oscillate) and grow like 1/depth^2,
    # matching the leading behaviour of the infinite-medium potential p/d^2 --
    # this is the numerical-stability property tests/... relies on, since the
    # closed-form doubling factor for a *finite* conductivity ratio does not
    # apply to the (insulating-air) boundary used here.
    e = np.array([[0, 0, 1.0]])
    ratios = []
    for depth in (0.05, 0.02, 0.01, 0.005):
        r0 = np.array([0, 0, 1 - depth])
        v = sphere_potential(r0, 1.0, 0.0, e, n_terms=20000)[0]
        ratios.append(v * depth ** 2)
    # v*depth^2 should approach a stable constant (leading 1/depth^2 divergence
    # captured), i.e. the last two estimates agree to within 1%.
    assert abs(ratios[-1] - ratios[-2]) / abs(ratios[-1]) < 0.01, ratios


def test_electrode_positions_on_unit_sphere():
    xyz = electrode_positions()
    assert np.allclose(np.linalg.norm(xyz, axis=1), 1.0)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
