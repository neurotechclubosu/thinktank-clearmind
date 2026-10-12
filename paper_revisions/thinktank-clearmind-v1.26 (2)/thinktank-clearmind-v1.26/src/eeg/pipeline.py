"""
Array-level implementation of the analysis chain

    composite EEG -> 32-sample Welch windows -> gamma power
                  -> normalised CMRO2 -> eight neurovascular variables

It is numerically equivalent to the file-based chain used by the GUI
(``EEG_Implement_Welch`` -> ``plot_normalized_gamma_across_channels`` ->
``calculate_neurovascular_variables``); tests/test_pipeline_equivalence.py
proves this on real composites.  It exists so that thousands of microgap
realisations can be analysed without writing 2-MB text files.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .EEG_NeurovascularVariables import neurovascular_from_cmro2, VARIABLES
from .synthesis import ELECTRODES

FS = 256
WIN = 32                    # samples per Welch window (125 ms)
TRIAL_ROWS = 1024           # Timestep(256) * TestDuration(4)
WINDOWS_PER_TRIAL = TRIAL_ROWS // WIN   # 32
GAMMA = (30.0, 100.0)       # [30, 100) Hz, upper bound exclusive (PR #2)
CMRO2_BASELINE = 2.1
CMRO2_RANGE = 1.0           # normalised CMRO2 spans [2.1, 3.1]

BANDS = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
         'Beta': (13, 30), 'Gamma': (30, 100)}


def n_complete_trials(n_rows: int) -> int:
    return n_rows // TRIAL_ROWS


def window_psd(x: np.ndarray):
    """Welch PSD for every non-overlapping 32-sample window.

    ``x``: (n_rows, n_ch).  Only complete 1024-row trials are kept (the same
    truncation as EEG_Implement_Welch).  Returns freqs (17,) and
    psd (n_windows, 17, n_ch).
    """
    n_tr = n_complete_trials(x.shape[0])
    if n_tr < 1:
        raise ValueError(f"need >= {TRIAL_ROWS} rows for one trial, got {x.shape[0]}")
    n_win = n_tr * WINDOWS_PER_TRIAL
    w = x[:n_win * WIN].reshape(n_win, WIN, x.shape[1])
    f, p = welch(w, fs=FS, nperseg=WIN, axis=1, scaling='density')
    return f, p


def band_power(freqs: np.ndarray, psd: np.ndarray, band) -> np.ndarray:
    lo, hi = band
    idx = (freqs >= lo) & (freqs < hi)
    return psd[:, idx, :].sum(axis=1)


def gamma_windows(x: np.ndarray) -> np.ndarray:
    """(n_windows, n_ch) gamma-band power (sum of PSD bins, as in v1.0.0)."""
    f, p = window_psd(x)
    return band_power(f, p, GAMMA)


def normalise_gamma(gamma: np.ndarray, bounds=None):
    """Global min-max map to the 2.1-3.1 umol/g/min CMRO2 scale.

    bounds=None  -> the recording's own min/max (published behaviour).
    bounds=(a,b) -> fixed reference bounds, values clipped to [a,b].
    """
    if bounds is None:
        lo, hi = float(gamma.min()), float(gamma.max())
        g = gamma
    else:
        lo, hi = bounds
        g = np.clip(gamma, lo, hi)
    if hi - lo == 0:
        return np.full(gamma.shape, CMRO2_BASELINE)
    return CMRO2_BASELINE + CMRO2_RANGE * (g - lo) / (hi - lo)


def analyse_composite(x: np.ndarray, bounds=None, input_mode="absolute") -> dict:
    """Full chain for one composite.  Returns arrays shaped (n_windows, n_ch)."""
    gamma = gamma_windows(x)
    cmro2_in = normalise_gamma(gamma, bounds)
    nv = neurovascular_from_cmro2(cmro2_in, input_mode)
    out = {'gamma': gamma, 'cmro2_in': cmro2_in}
    out.update(nv)
    return out


def channel_means(res: dict, keys=VARIABLES) -> dict:
    """Per-channel mean over all windows (== legacy mean-of-trial-means)."""
    return {k: res[k].mean(axis=0) for k in keys}


def stack_channel_means(res: dict, keys=VARIABLES) -> np.ndarray:
    """(n_vars, n_ch) matrix in ``keys`` order."""
    return np.vstack([res[k].mean(axis=0) for k in keys])


class WindowBank:
    """Pre-computed gamma windows for the first 512 rows of every recording.

    Because every segment of a composite is 256 or 512 rows long and starts on a
    multiple of 256 rows, no 32-sample Welch window ever straddles a segment
    boundary.  A composite's window series is therefore the concatenation of the
    per-recording window series, which makes Monte-Carlo over thousands of
    microgap realisations cheap.  Equality with ``analyse_composite`` on the
    materialised composite is asserted in tests/test_pipeline_equivalence.py.
    """

    def __init__(self, bank, epoch: int = 0):
        self.bank = bank
        self.epoch = epoch
        self._g = {}

    def gamma_of(self, file_index: int) -> np.ndarray:
        if file_index not in self._g:
            src = self.bank.epochs(file_index)
            a = self.epoch * 256
            self._g[file_index] = gamma_windows_partial(src[a:a + 512])
        return self._g[file_index]

    def composite_gamma(self, plan) -> np.ndarray:
        blocks = []
        for seg in plan:
            blocks.append(self.gamma_of(seg.file_index)[: seg.n_rows // WIN])
        g = np.concatenate(blocks, axis=0)
        n_keep = (g.shape[0] * WIN // TRIAL_ROWS) * WINDOWS_PER_TRIAL
        return g[:n_keep]


def gamma_windows_partial(x: np.ndarray) -> np.ndarray:
    """Gamma power for every complete 32-sample window (no trial truncation)."""
    n_win = x.shape[0] // WIN
    w = x[:n_win * WIN].reshape(n_win, WIN, x.shape[1])
    f, p = welch(w, fs=FS, nperseg=WIN, axis=1, scaling='density')
    return band_power(f, p, GAMMA)


def analyse_gamma(gamma: np.ndarray, bounds=None, input_mode="absolute") -> dict:
    cmro2_in = normalise_gamma(gamma, bounds)
    out = {'gamma': gamma, 'cmro2_in': cmro2_in}
    out.update(neurovascular_from_cmro2(cmro2_in, input_mode))
    return out
