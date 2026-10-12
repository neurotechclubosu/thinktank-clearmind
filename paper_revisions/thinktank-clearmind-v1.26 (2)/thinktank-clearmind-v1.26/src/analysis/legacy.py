"""
Reference re-implementation of the v1.0.0 *defects*, used only to quantify what
the corrections changed (docs/CORRECTIONS.md, "impact" tables).  Never used by
the application.

v1.0.0 behaviours reproduced
----------------------------
C-1  Repeated samples: every Welch "trial" analysed the *first* 1024 samples of
     the composite, so all trials were identical and only the first 4 s of a
     composite ever entered the analysis.
C-2  Trial off-by-one: trials 1..N-1 were analysed (trial 0 does not exist, trial
     N was skipped).  Because of C-1 this does not change the numbers, but it
     makes the analysis fail for N = 1.
C-5  Units: normalised absolute CMRO2 (2.1-3.1) was fed in as fractional change.
"""
import numpy as np
from ..eeg import pipeline as P


def analyse_composite_v100(x: np.ndarray) -> dict:
    n_tr = P.n_complete_trials(x.shape[0])
    if n_tr < 2:
        raise ValueError("v1.0.0 required >= 2 complete trials (off-by-one)")
    first = x[:P.TRIAL_ROWS]                          # C-1: same samples every trial
    g = P.gamma_windows(first)                        # 32 windows
    g_all = np.tile(g, (n_tr - 1, 1))                 # C-2: trials 1..N-1, all identical
    return P.analyse_gamma(g_all, bounds=None, input_mode="fractional_legacy")
