"""
Synthetic-EEG assembly: the single source of truth for how a text string is
turned into a composite EEG recording.

Design goals (manuscript revision, v1.1.0)
------------------------------------------
* **Deterministic given (text, seed).**  Microgap choices are drawn from a
  seeded ``random.Random`` -- "pseudorandom", not random.  Two runs with the
  same text, seed and settings produce byte-identical composites.
* **Explicit provenance.**  ``plan_composite`` returns a manifest (a list of
  ``Segment`` records) that says exactly which recording, which rows and which
  token produced every second of the composite.
* **Planning is separate from data access**, so the assembly logic can be unit
  tested without the 2 MB-per-file EEG data, and the GUI and the batch
  reproduction script share one implementation.

Definitions
-----------
Phoneme segment
    256 consecutive rows (1 s at 256 Hz) taken from the *first* epoch of
    ``DLR_<n>_1.txt`` (the first row whose time column is 0).  Each phoneme file
    holds ~20 repetitions (epochs); the assembly always uses epoch 0.
Microgap
    A segment inserted at every whitespace / punctuation token when microgaps
    are enabled.  Its source is one of five auxiliary recordings
    (``DLR_39..43``, "rand1..rand5"), chosen uniformly, and its length is 256
    or 512 rows (1 s or 2 s), chosen with equal probability.
Segment unit
    ``"word_final"`` reproduces the behaviour of the released GUI
    (v1.0.0 ... v1.0.x): **one** segment per word, taken from the recording of
    the word's *final* phoneme.  ``"phoneme"`` writes one segment per phoneme,
    which is how the Methods text of the original submission described the
    procedure.  See docs/CORRECTIONS.md, item C-4.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import string
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

FS = 256                      # sampling rate, Hz
SEGMENT_ROWS = 256            # one phoneme segment = 1 s
EPOCH_ROWS = 256
MICROGAP_LENGTHS = (256, 512)
ELECTRODES = ['Fp1', 'Fp2', 'F3', 'F4', 'T5', 'T6', 'O1', 'O2',
              'F7', 'F8', 'C3', 'C4', 'T3', 'T4', 'P3', 'P4']

PHONEME_TO_INDEX: Dict[str, int] = {
    "AA": 0, "AE": 1, "AH": 2, "AO": 3, "AW": 4, "AY": 5,
    "B": 6, "CH": 7, "D": 8, "DH": 9, "EH": 10, "ER": 11,
    "EY": 12, "F": 13, "G": 14, "HH": 15, "IH": 16, "IY": 17,
    "JH": 18, "K": 19, "L": 20, "M": 21, "N": 22, "NG": 23,
    "OW": 24, "OY": 25, "P": 26, "R": 27, "S": 28, "SH": 29,
    "T": 30, "TH": 31, "UH": 32, "UW": 33, "V": 34, "W": 35,
    "Y": 36, "Z": 37, "ZH": 38,
    "rand1": 39, "rand2": 40, "rand3": 41, "rand4": 42, "rand5": 43,
}
MICROGAP_FILES = (39, 40, 41, 42, 43)

# Typographic characters that LLM output often contains.  The released GUI
# treats them as *words* (they are not in string.punctuation), which made the
# assembly depend on g2p_en's handling of empty strings and silently re-used the
# previous word's recording.  They are mapped to ASCII here.
_TYPOGRAPHIC = {
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2014": " - ", "\u2013": " - ", "\u2026": "...", "\u00a0": " ",
}


@dataclass(frozen=True)
class Token:
    kind: str      # "word" | "gap"
    text: str


@dataclass(frozen=True)
class Segment:
    """One block of the composite.  ``file_index`` -> ``DLR_<n>_1.txt``."""
    kind: str          # "phoneme" | "word_final" | "microgap"
    token: str         # the word / gap character that caused it
    phoneme: str       # ARPAbet symbol or "randN"
    file_index: int
    n_rows: int        # 256 or 512

    def asdict(self):
        return asdict(self)


def normalize_punctuation(text: str) -> str:
    for k, v in _TYPOGRAPHIC.items():
        text = text.replace(k, v)
    return text


def tokenize(text: str, microgaps: bool) -> List[Token]:
    """Split text into word / gap tokens exactly as the GUI does.

    microgaps=True  : every run of whitespace and every punctuation character is
                      a "gap" token (regex ``\\w+|[^\\w\\s]|\\s+`` as in the GUI).
    microgaps=False : whitespace-separated words with leading/trailing
                      punctuation stripped; no gap tokens.
    """
    text = normalize_punctuation(text)
    if microgaps:
        tokens = []
        for m in re.findall(r"\w+|[^\w\s]|\s+", text):
            if m.strip() == "" or m in string.punctuation:
                tokens.append(Token("gap", m))
            else:
                tokens.append(Token("word", m))
        return tokens
    words = [w.strip(string.punctuation) for w in text.split()]
    return [Token("word", w) for w in words if w]


def stable_seed(text: str, seed: int = 0, microgaps: bool = True) -> int:
    """Derive a 64-bit RNG seed from (text, user seed, condition).

    Uses SHA-256 rather than ``hash()`` (which is salted per process), so the
    stream is identical across machines and Python versions.
    """
    h = hashlib.sha256(f"{seed}|{int(microgaps)}|{normalize_punctuation(text)}".encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big")


PhonemeLookup = Callable[[str], Sequence[str]]


def plan_composite(text: str,
                   phoneme_lookup: PhonemeLookup,
                   microgaps: bool = True,
                   seed: int = 0,
                   unit: str = "word_final") -> List[Segment]:
    """Return the manifest of segments for ``text``.

    ``phoneme_lookup(word) -> list of ARPAbet symbols (stress stripped)``.
    Words whose phoneme list is empty (or contains only unrecognised symbols)
    contribute **no** segment; the released GUI instead re-used the previous
    word's file (a stale-variable defect that only triggers for empty lookups).
    """
    if unit not in ("word_final", "phoneme"):
        raise ValueError("unit must be 'word_final' or 'phoneme'")
    rng = random.Random(stable_seed(text, seed, microgaps))
    plan: List[Segment] = []
    for tok in tokenize(text, microgaps):
        if tok.kind == "gap":
            # Draw order is part of the definition: file first, then length.
            k = rng.choice(MICROGAP_FILES)
            n = rng.choice(MICROGAP_LENGTHS)
            plan.append(Segment("microgap", tok.text, f"rand{k - 38}", k, n))
            continue
        phs = [p for p in phoneme_lookup(tok.text) if p in PHONEME_TO_INDEX]
        if not phs:
            continue
        if unit == "word_final":
            plan.append(Segment("word_final", tok.text, phs[-1],
                                PHONEME_TO_INDEX[phs[-1]], SEGMENT_ROWS))
        else:
            for p in phs:
                plan.append(Segment("phoneme", tok.text, p,
                                    PHONEME_TO_INDEX[p], SEGMENT_ROWS))
    return plan


def plan_summary(plan: Sequence[Segment]) -> dict:
    n_gap = sum(s.kind == "microgap" for s in plan)
    rows = sum(s.n_rows for s in plan)
    return {"n_segments": len(plan), "n_microgaps": n_gap,
            "n_rows": rows, "duration_s": rows / FS,
            "n_trials_1024": rows // 1024}


class EpochBank:
    """Lazy loader for the 44 ``DLR_<n>_1.txt`` recordings.

    ``start_row(n)`` is the first row whose time column is exactly 0, the same
    anchor used by the GUI.  ``epochs(n)`` yields the recording from that row
    as an ``(n_rows, 16)`` array; epoch k occupies rows ``[256k, 256(k+1))``.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._cache: Dict[int, np.ndarray] = {}
        self._start: Dict[int, int] = {}

    def _path(self, n: int) -> str:
        return os.path.join(self.data_dir, f"DLR_{n}_1.txt")

    def _load(self, n: int):
        if n not in self._cache:
            raw = np.loadtxt(self._path(n))
            zeros = np.flatnonzero(raw[:, 0] == 0.0)
            if zeros.size == 0:
                raise ValueError(f"no zero-time row in {self._path(n)}")
            s = int(zeros[0])
            self._start[n] = s
            self._cache[n] = raw[s:, 1:17].copy()
        return self._cache[n]

    def start_row(self, n: int) -> int:
        self._load(n)
        return self._start[n]

    def epochs(self, n: int) -> np.ndarray:
        return self._load(n)

    def n_epochs(self, n: int) -> int:
        return self._load(n).shape[0] // EPOCH_ROWS

    def lines(self, n: int) -> List[str]:
        """Raw text lines from the anchor row (for byte-identical .tsv output)."""
        with open(self._path(n), "r", encoding="utf-8") as fh:
            all_lines = fh.readlines()
        return all_lines[self.start_row(n):]


def materialize(plan: Sequence[Segment], bank: EpochBank,
                epoch_chooser: Optional[Callable[[Segment], int]] = None) -> np.ndarray:
    """Concatenate the planned segments into an ``(n_rows, 16)`` array.

    ``epoch_chooser(segment) -> k`` selects which repetition (epoch) of the
    source recording to use; the default (None) is epoch 0, which is what the
    application does.  Choosing k >= 1 gives *held-out real* alternatives used
    by the consistency analysis.
    """
    blocks = []
    for seg in plan:
        src = bank.epochs(seg.file_index)
        k = 0 if epoch_chooser is None else int(epoch_chooser(seg))
        a = k * EPOCH_ROWS
        b = a + seg.n_rows
        if b > src.shape[0]:
            raise ValueError(f"segment {seg} exceeds recording length at epoch {k}")
        blocks.append(src[a:b])
    return np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 16))


def write_tsv_lines(plan: Sequence[Segment], bank: EpochBank, out_path: str) -> None:
    """Write the composite in the original 32-column layout (GUI compatible)."""
    with open(out_path, "w", encoding="utf-8") as fh:
        for seg in plan:
            fh.writelines(bank.lines(seg.file_index)[:seg.n_rows])
