"""
Synthetic-vs-real EEG consistency analysis (Reviewer #3, point 2).

What "real" means here
----------------------
No continuous imagined-speech EEG of the test *sentences* exists, so there is no
ground truth for a sentence-level composite.  The only real EEG available is the
source dataset itself (LaRocco et al., 2023): each ``DLR_<n>_1.txt`` holds ~20
repetitions ("epochs", 1 s) of one imagined phoneme, and the application uses
only epoch 0 of each.  That gives two legitimate real references:

R1  "Re-drawn real composites".  The identical segment manifest (same phoneme /
    microgap recordings, same lengths) filled with *held-out* epochs 1..19.  This
    asks: is the composite the application actually produces typical of what
    other real repetitions of the same sequence would give?
R2  "Real streams".  The 44 recordings themselves, taken as contiguous 19-epoch
    (4,864-sample) streams with no splicing.  This asks: does a synthetic composite
    look like real, unspliced EEG from this participant?

Both are compared with the synthetic composite in the three domains requested:
time-frequency, source space, and metabolic metrics.  Everything is descriptive:
one participant, one dataset, no independent ground truth.  Because the eight
metabolic outputs are functions of one scalar, CMRO2 is the primary metabolic
metric; the rest are listed for completeness.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view

from ..eeg import pipeline as P
from ..eeg.synthesis import EPOCH_ROWS, materialize
from .source_space import SourceImager, CHANNELS, electrode_positions

FS = 256
SEG, HOP = 128, 64
BANDS = P.BANDS
BAND_NAMES = list(BANDS)


# ---------------------------------------------------------------- real streams
def real_streams(bank, n_files=44, n_epochs=19):
    return [bank.epochs(n)[: n_epochs * EPOCH_ROWS] for n in range(n_files)]


def pooled_reference_bounds(streams, lo_pct=1, hi_pct=99):
    """Common gamma->CMRO2 scale: 1st-99th percentile of all real 32-sample windows."""
    g = np.concatenate([P.gamma_windows_partial(s) for s in streams])
    return float(np.percentile(g, lo_pct)), float(np.percentile(g, hi_pct))


# --------------------------------------------------------------- time-frequency
def _segment_psd(x):
    """(n_seg, 65, n_ch) PSD from 128-sample Hann segments, hop 64, constant detrend."""
    w = sliding_window_view(x, SEG, axis=0)[::HOP]          # (n_seg, ch, 128)
    w = w - w.mean(axis=-1, keepdims=True)
    win = np.hanning(SEG)
    F = np.fft.rfft(w * win, axis=-1)
    psd = (np.abs(F) ** 2) / (FS * (win ** 2).sum())
    psd[..., 1:-1] *= 2
    return np.transpose(psd, (0, 2, 1))


FREQS = np.fft.rfftfreq(SEG, 1 / FS)
_F_IDX = (FREQS >= 1) & (FREQS <= 100)


def tf_features(x, block=EPOCH_ROWS):
    psd = _segment_psd(x)                                   # (n_seg, 65, ch)
    mean_psd = psd.mean(axis=0)                             # (65, ch)
    logpsd = np.log10(mean_psd[_F_IDX]).mean(axis=1)        # channel-averaged, (n_f,)
    tot = psd[:, (FREQS >= 0.5) & (FREQS < 100), :].sum(axis=1)
    rel = []
    band_logpow = {}
    for b, (lo, hi) in BANDS.items():
        idx = (FREQS >= lo) & (FREQS < hi)
        bp = psd[:, idx, :].sum(axis=1)                     # (n_seg, ch)
        rel.append(float((bp / tot).mean()))
        band_logpow[b] = np.log10(bp).ravel()
    dx = np.abs(np.diff(x, axis=0))
    med = np.median(dx, axis=0) + 1e-12
    joins = np.arange(block, x.shape[0], block) - 1         # step between row j and j+1
    sdi = float(np.median(dx[joins] / med)) if len(joins) else np.nan
    return dict(logpsd=logpsd, relband=np.array(rel), band_logpow=band_logpow, sdi=sdi)


def tf_compare(feat, ref_logpsd, ref_relband):
    r = float(np.corrcoef(feat['logpsd'], ref_logpsd)[0, 1])
    rmse_db = float(np.sqrt(np.mean((10 * (feat['logpsd'] - ref_logpsd)) ** 2)))
    d_rel = np.abs(feat['relband'] - ref_relband)
    return dict(logpsd_r=r, logpsd_rmse_db=rmse_db, relband_mad=float(d_rel.mean()),
                gamma_rel=float(feat['relband'][-1]))


# ------------------------------------------------------------------ source space
def source_features(x, imager, dc_removed):
    pm = imager.power_map(x, remove_dc_per_block=EPOCH_ROWS if dc_removed else None)
    lp = np.log10(pm)
    y = imager.src[:, 1]
    x_ = imager.src[:, 0]
    w = pm / pm.sum()
    return dict(logmap=lp, left_frac=float(w[y > 0].sum()), ant_frac=float(w[x_ > 0].sum()))


# -------------------------------------------------------------------- metabolic
def metabolic_features(x, bounds):
    g = P.gamma_windows_partial(x[: (x.shape[0] // P.TRIAL_ROWS) * P.TRIAL_ROWS
                                  if x.shape[0] >= P.TRIAL_ROWS else x.shape[0]])
    res = P.analyse_gamma(g, bounds)
    return dict(cmro2_montage=float(res['CMRO2'].mean()),
                cmro2_profile=res['CMRO2'].mean(axis=0),
                all_montage={k: float(res[k].mean()) for k in
                             ('CBF', 'OEF', 'ph_V', 'p_CO2_V', 'pO2_cap', 'CMRO2',
                              'DeltaHCO2', 'DeltaLAC')})


# ------------------------------------------------------------------- orchestrate
class ConsistencyAnalysis:
    def __init__(self, bank, seed=0):
        self.bank = bank
        self.streams = real_streams(bank)
        self.bounds = pooled_reference_bounds(self.streams)
        self.imager = SourceImager()
        self.rng = np.random.default_rng(seed)
        self._prep_reference()

    def features(self, x):
        out = dict(tf=tf_features(x), metab=metabolic_features(x, self.bounds))
        for dc in (False, True):
            out['src_dc' if dc else 'src_raw'] = source_features(x, self.imager, dc)
        return out

    def _prep_reference(self):
        self.real = [self.features(s) for s in self.streams]
        self.ref = {}
        self._loo_stats = None

    def _ref_from(self, idx):
        f = [self.real[i] for i in idx]
        return dict(
            logpsd=np.mean([a['tf']['logpsd'] for a in f], axis=0),
            relband=np.mean([a['tf']['relband'] for a in f], axis=0),
            map_raw=np.mean([a['src_raw']['logmap'] for a in f], axis=0),
            map_dc=np.mean([a['src_dc']['logmap'] for a in f], axis=0),
            prof=np.mean([a['metab']['cmro2_profile'] for a in f], axis=0))

    def compare(self, feat, ref):
        d = tf_compare(feat['tf'], ref['logpsd'], ref['relband'])
        d['sdi'] = feat['tf']['sdi']
        d['src_raw_rho'] = float(stats.spearmanr(feat['src_raw']['logmap'], ref['map_raw']).statistic)
        d['src_dc_rho'] = float(stats.spearmanr(feat['src_dc']['logmap'], ref['map_dc']).statistic)
        d['src_raw_left'] = feat['src_raw']['left_frac']
        d['src_dc_left'] = feat['src_dc']['left_frac']
        d['cmro2_montage'] = feat['metab']['cmro2_montage']
        d['cmro2_profile_rho'] = float(stats.spearmanr(feat['metab']['cmro2_profile'], ref['prof']).statistic)
        return d

    def real_reference_distribution(self):
        """Leave-one-out comparison of each real stream to the other 43."""
        if self._loo_stats is None:
            rows = []
            for i in range(len(self.real)):
                ref = self._ref_from([j for j in range(len(self.real)) if j != i])
                rows.append(self.compare(self.real[i], ref))
            self._loo_stats = rows
        return self._loo_stats

    def synthetic_vs_real(self, x):
        ref = self._ref_from(range(len(self.real)))
        return self.compare(self.features(x), ref)

    def redraws(self, plan, B=200):
        """R1: same manifest filled with held-out epochs 1..; returns list of feature dicts."""
        ref = self._ref_from(range(len(self.real)))
        out = []
        for _ in range(B):
            def chooser(seg):
                n_blocks = seg.n_rows // EPOCH_ROWS
                n_ep = self.bank.n_epochs(seg.file_index)
                return int(self.rng.integers(1, n_ep - n_blocks + 1))
            x = materialize(plan, self.bank, chooser)
            out.append(self.compare(self.features(x), ref))
        return out


def percentile_rank(value, sample):
    sample = np.asarray(sample, float)
    return float(100 * (np.mean(sample < value) + 0.5 * np.mean(sample == value)))


# ---------------------------------------------------- phoneme-specificity check
def epoch_features(bank, files, epochs):
    """Log band power per channel for single 1-s epochs -> (n, 80) feature rows."""
    rows, labels, ep_id = [], [], []
    for f in files:
        src = bank.epochs(f)
        for k in epochs:
            seg = src[k * EPOCH_ROWS:(k + 1) * EPOCH_ROWS]
            psd = _segment_psd(seg).mean(axis=0)            # (65, ch)
            v = []
            for b, (lo, hi) in BANDS.items():
                idx = (FREQS >= lo) & (FREQS < hi)
                v.append(np.log10(psd[idx].sum(axis=0)))
            rows.append(np.concatenate(v)); labels.append(f); ep_id.append(k)
    return np.array(rows), np.array(labels), np.array(ep_id)


def identification_test(bank, n_perm=2000, seed=0):
    """Is epoch 0 of a phoneme identifiable from held-out epochs of the same
    phoneme, in a 39-way nearest-centroid test on band-power features?  Also runs
    the same test within the real held-out data (leave-one-epoch-out) as the
    ceiling for 'real' phoneme information at this feature level."""
    files = list(range(39))
    X0, y0, _ = epoch_features(bank, files, [0])
    Xh, yh, eh = epoch_features(bank, files, range(1, 19))
    mu, sd = Xh.mean(0), Xh.std(0) + 1e-12
    Z0, Zh = (X0 - mu) / sd, (Xh - mu) / sd
    cent = np.array([Zh[yh == f].mean(0) for f in files])

    def acc(Z, y, C):
        pred = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        return float((pred == y).mean())

    a_syn = acc(Z0, y0, cent)
    rng = np.random.default_rng(seed)
    null = np.array([acc(Z0, rng.permutation(y0), cent) for _ in range(n_perm)])
    p_syn = float((1 + (null >= a_syn).sum()) / (n_perm + 1))
    # ceiling: leave-one-epoch-out within held-out real data
    hits = 0
    for i in range(len(yh)):
        m = np.ones(len(yh), bool); m[i] = False
        C = np.array([Zh[m & (yh == f)].mean(0) for f in files])
        hits += int(np.argmin(((Zh[i] - C) ** 2).sum(1)) == yh[i])
    a_real = hits / len(yh)
    return dict(n_classes=39, chance=1 / 39, acc_epoch0_vs_heldout=a_syn, p_perm=p_syn,
                acc_real_loo=a_real, n_perm=n_perm)
