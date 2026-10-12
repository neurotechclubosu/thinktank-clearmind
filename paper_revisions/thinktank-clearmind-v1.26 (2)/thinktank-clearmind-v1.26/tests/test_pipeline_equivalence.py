"""Numerical-equivalence tests: array pipeline vs. legacy file-based chain,
and WindowBank (fast path) vs. full materialisation."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.eeg.synthesis import EpochBank, plan_composite, materialize
from src.eeg import pipeline as P
from src.eeg.EEG_Implement_Welch import EEG_Implement_Welch
from src.eeg.EEG_normalizedGamma_CMRO2 import plot_normalized_gamma_across_channels
from src.eeg.EEG_NeurovascularVariables import calculate_neurovascular_variables, VARIABLES

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "eeg_recordings")
FX = os.path.join(os.path.dirname(__file__), "..", "data/fixtures/trial_phonemes.json")


def _have_data():
    return os.path.isdir(DATA)


def test_array_pipeline_matches_legacy_file_chain():
    if not _have_data():
        print("  (skipped: no EEG data)")
        return
    fx = json.load(open(FX))
    lex = fx["lexicon"]
    look = lambda w: lex.get(w.lower(), [])
    bank = EpochBank(DATA)
    plan = plan_composite(fx["trials"]["trial3"]["response"], look, microgaps=True, seed=0)
    x = materialize(plan, bank)
    tmp = "/tmp/_test_pipeline.txt"
    from src.eeg.synthesis import write_tsv_lines
    write_tsv_lines(plan, bank, tmp)
    sp, T = EEG_Implement_Welch(tmp)
    cm = plot_normalized_gamma_across_channels(sp, P.ELECTRODES, T)
    legacy = calculate_neurovascular_variables(cm)
    res = P.analyse_composite(x)
    max_rel = 0.0
    for v in VARIABLES:
        for ci, e in enumerate(P.ELECTRODES):
            old = np.concatenate([legacy[v][f"{e} trial {k}"] for k in range(1, T + 1)])
            max_rel = max(max_rel, np.abs(old - res[v][:, ci]).max() / max(1e-12, np.abs(old).max()))
    assert max_rel < 1e-10, max_rel
    os.remove(tmp)


def test_windowbank_matches_full_materialisation():
    if not _have_data():
        print("  (skipped: no EEG data)")
        return
    fx = json.load(open(FX))
    lex = fx["lexicon"]
    look = lambda w: lex.get(w.lower(), [])
    bank = EpochBank(DATA)
    wb = P.WindowBank(bank)
    plan = plan_composite(fx["trials"]["trial1"]["response"], look, microgaps=True, seed=2)
    x = materialize(plan, bank)
    a = P.analyse_composite(x)
    b = P.analyse_gamma(wb.composite_gamma(plan))
    assert np.array_equal(a["gamma"], b["gamma"])
    assert np.allclose(a["CBF"], b["CBF"])


def test_welch_raises_on_short_recording():
    if not _have_data():
        print("  (skipped: no EEG data)")
        return
    short = "/tmp/_test_short.txt"
    bank = EpochBank(DATA)
    with open(short, "w") as fh:
        fh.writelines(bank.lines(0)[:100])
    try:
        EEG_Implement_Welch(short)
        raised = False
    except ValueError:
        raised = True
    assert raised
    os.remove(short)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
