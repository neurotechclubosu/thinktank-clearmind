"""Assert-based tests for src.eeg.synthesis (no pytest in this environment;
run directly: `python tests/test_synthesis.py`, or via tests/run_all.py)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.eeg.synthesis import (tokenize, plan_composite, stable_seed, EpochBank,
                               materialize, write_tsv_lines, normalize_punctuation)

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "eeg_recordings")


def test_tokenize_microgaps_splits_punct_and_space():
    toks = tokenize("Hi, Bob!", microgaps=True)
    kinds = [(t.kind, t.text) for t in toks]
    assert kinds == [("word", "Hi"), ("gap", ","), ("gap", " "),
                     ("word", "Bob"), ("gap", "!")], kinds


def test_tokenize_no_microgaps_strips_punct_no_gaps():
    toks = tokenize("Hi, Bob!", microgaps=False)
    assert [t.text for t in toks] == ["Hi", "Bob"]
    assert all(t.kind == "word" for t in toks)


def test_typographic_quotes_normalised():
    assert normalize_punctuation("it\u2019s") == "it's"
    toks = tokenize("it\u2019s ok", microgaps=False)
    assert [t.text for t in toks] == ["it's", "ok"]


def test_plan_deterministic_given_same_seed():
    look = lambda w: {"hi": ["HH", "AY"], "bob": ["B", "AA", "B"]}.get(w.lower(), [])
    p1 = plan_composite("Hi, Bob!", look, microgaps=True, seed=0)
    p2 = plan_composite("Hi, Bob!", look, microgaps=True, seed=0)
    assert p1 == p2


def test_plan_differs_across_seeds():
    look = lambda w: {"hi": ["HH", "AY"], "bob": ["B", "AA", "B"]}.get(w.lower(), [])
    p1 = plan_composite("Hi, Bob!", look, microgaps=True, seed=0)
    p2 = plan_composite("Hi, Bob!", look, microgaps=True, seed=1)
    assert p1 != p2


def test_plan_differs_across_text():
    assert stable_seed("a") != stable_seed("b")


def test_word_final_uses_last_phoneme():
    look = lambda w: ["B", "AA", "B"]
    p = plan_composite("Bob", look, microgaps=False, unit="word_final")
    assert len(p) == 1 and p[0].phoneme == "B" and p[0].file_index == 6  # last B


def test_phoneme_unit_one_segment_per_phoneme():
    look = lambda w: ["B", "AA", "B"]
    p = plan_composite("Bob", look, microgaps=False, unit="phoneme")
    assert len(p) == 3


def test_oov_word_contributes_no_segment():
    look = lambda w: []
    p = plan_composite("xyzzy plugh", look, microgaps=False)
    assert p == []


def test_microgap_sources_and_lengths_within_range():
    look = lambda w: []
    p = plan_composite("a, b", look, microgaps=True, seed=3)
    gaps = [s for s in p if s.kind == "microgap"]
    assert len(gaps) >= 1
    assert all(s.file_index in (39, 40, 41, 42, 43) for s in gaps)
    assert all(s.n_rows in (256, 512) for s in gaps)


def test_epochbank_start_row_and_materialize_matches_tsv():
    if not os.path.isdir(DATA):
        print("  (skipped: no EEG data directory)")
        return
    bank = EpochBank(DATA)
    look = lambda w: {"hi": ["HH", "AY"]}.get(w.lower(), [])
    plan = plan_composite("Hi hi", look, microgaps=True, seed=1)
    x = materialize(plan, bank)
    tmp = "/tmp/_test_composite.txt"
    write_tsv_lines(plan, bank, tmp)
    raw = np.loadtxt(tmp)
    assert np.array_equal(raw[:, 1:17], x)
    os.remove(tmp)


if __name__ == "__main__":
    import inspect
    ns = dict(globals())
    fails = 0
    for name, fn in sorted(ns.items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
