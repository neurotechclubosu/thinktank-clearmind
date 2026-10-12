"""Regression + sanity tests for the corrected neurovascular model."""
import sys, os, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.eeg.EEG_NeurovascularVariables import (calculate_neurovascular_variables,
                                                 neurovascular_from_cmro2, VARIABLES)

ORIG = os.path.join(os.path.dirname(__file__), "..", "..", "orig_snapshot",
                    "src/eeg/EEG_NeurovascularVariables.py")


def test_fractional_legacy_mode_matches_v100_exactly():
    if not os.path.exists(ORIG):
        print("  (skipped: no v1.0.0 snapshot available)")
        return
    spec = importlib.util.spec_from_file_location("old", ORIG)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    x = {"Fp1 trial 1": list(np.linspace(2.1, 3.1, 32)),
        "Fp2 trial 1": list(np.linspace(2.5, 2.9, 32))}
    a = old.calculate_neurovascular_variables(x)
    b = calculate_neurovascular_variables(x, input_mode="fractional_legacy")
    for v in a:
        for k in a[v]:
            assert np.allclose(a[v][k], b[v][k], atol=1e-10)


def test_absolute_mode_baseline_maps_to_baseline_oef():
    r = neurovascular_from_cmro2([2.1], input_mode="absolute")
    assert abs(r["OEF"][0] - 0.42) < 1e-9
    assert abs(r["CMRO2"][0] - 2.1) < 1e-9


def test_absolute_mode_stays_in_physiological_band():
    x = np.linspace(2.1, 3.1, 50)
    r = neurovascular_from_cmro2(x, input_mode="absolute")
    assert (r["CBF"] > 0).all()
    assert (0 < r["OEF"]).all() and (r["OEF"] <= 0.42).all()
    assert (r["ph_V"] > 7.0).all() and (r["ph_V"] < 7.45).all()


def test_affine_variables_are_monotone_in_delta():
    x = np.linspace(2.1, 3.1, 10)
    r = neurovascular_from_cmro2(x, input_mode="absolute")
    for v in ("CBF", "CMRO2"):
        assert np.all(np.diff(r[v]) > 0), v


def test_dict_api_shapes_match_vectorised_call():
    x = {"Fp1 trial 1": [2.1, 2.5, 3.1]}
    d = calculate_neurovascular_variables(x)
    v = neurovascular_from_cmro2([2.1, 2.5, 3.1])
    for k in VARIABLES:
        assert np.allclose(d[k]["Fp1 trial 1"], v[k])


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
