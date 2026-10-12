"""
Statistical analysis for the microgap comparison (manuscript revision, v1.1.0).

Why the analysis was redefined
------------------------------
v1.0.0 averaged each recording per electrode and ran a paired t-test across the
16 electrodes (n = 16).  Those 16 numbers come from one participant, one
synthesised recording per condition, are spatially correlated, and -- for the
eight neurovascular outputs -- are all deterministic functions of one scalar.
They are not 16 independent replicates, and the p-values that resulted describe
the consistency of an effect *across the montage*, not the probability that
microgaps have an effect.

What is random in this design?
------------------------------
Only one thing: the pseudorandom microgap draws (which auxiliary recording, and
whether 1 s or 2 s).  The no-microgap composite is fully deterministic.  The
honest statistical unit for "does inserting microgaps change the simulated
output, and by how much does the answer depend on the particular draw?" is
therefore the **microgap realisation** (one seed), sampled R times per sentence.
Channels are treated as repeated measures and summarised *within* each
realisation (montage mean); sentences (n = 3) are reported separately and never
pooled into an inferential test.

Everything here is stimulus-conditional: inference is about the simulation
process for the sentences tested, not about people.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..eeg import pipeline as P
from ..eeg.synthesis import plan_composite
from ..eeg.EEG_NeurovascularVariables import VARIABLES, AFFINE_VARIABLES

R_DEFAULT = 2000
CMRO2_IDX = VARIABLES.index('CMRO2')


# --------------------------------------------------------------------------
# Monte-Carlo over microgap realisations
# --------------------------------------------------------------------------
def condition_means(gamma, bounds=None, input_mode="absolute"):
    """(n_vars, 16) per-channel means for one composite."""
    res = P.analyse_gamma(gamma, bounds, input_mode)
    return P.stack_channel_means(res), res


def run_mc(text, look, wbank, R=R_DEFAULT, unit="word_final", bounds=None,
           input_mode="absolute", legacy=False, bank=None, base_seed=0):
    """Analyse the no-microgap composite once and R seeded microgap composites.

    Returns dict with
      m2   : (n_vars, 16)        no-microgap channel means (deterministic)
      m1   : (R, n_vars, 16)     microgap channel means, one row per seed
      n_win1: (R,) windows analysed per realisation
      win2 : (n_win2, 16)        gamma windows of the no-microgap composite
      win1_seed0 : gamma windows of realisation seed=base_seed
    ``legacy=True`` applies the v1.0.0 defects (first 4 s only, legacy units)
    for the impact analysis; it needs ``bank`` to materialise the composite.
    """
    from .legacy import analyse_composite_v100
    from ..eeg.synthesis import materialize

    def analyse(plan):
        if legacy:
            x = materialize(plan, bank)
            res = analyse_composite_v100(x)
            return P.stack_channel_means(res), res['gamma']
        g = wbank.composite_gamma(plan)
        m, res = condition_means(g, bounds, input_mode)
        return m, g

    plan2 = plan_composite(text, look, microgaps=False, unit=unit)
    m2, g2 = analyse(plan2)
    m1 = np.empty((R, len(VARIABLES), 16))
    n_win1 = np.empty(R, dtype=int)
    g1_seed0 = None
    for r in range(R):
        plan1 = plan_composite(text, look, microgaps=True, seed=base_seed + r, unit=unit)
        m, g = analyse(plan1)
        m1[r] = m
        n_win1[r] = g.shape[0]
        if r == 0:
            g1_seed0 = g
    return dict(m2=m2, m1=m1, n_win1=n_win1, win2=g2, win1_seed0=g1_seed0,
                n_win2=g2.shape[0], n_segments2=len(plan2))


def summarise_mc(mc, alpha=0.05):
    """Per-variable summary of the montage-mean effect D_r = mean_c(m1_rc - m2_c)."""
    D = (mc['m1'] - mc['m2'][None]).mean(axis=2)        # (R, n_vars)
    R = D.shape[0]
    rows = []
    for k, v in enumerate(VARIABLES):
        d = D[:, k]
        mean, sd = d.mean(), d.std(ddof=1)
        lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        n_le, n_ge = int((d <= 0).sum()), int((d >= 0).sum())
        # exact Monte-Carlo p (Phipson & Smyth 2010): never reported as 0
        p_mc = min(1.0, 2 * (1 + min(n_le, n_ge)) / (R + 1))
        rows.append(dict(variable=v, no_microgap=float(mc['m2'][k].mean()),
                         microgap_mean=float(mc['m1'][:, k].mean()),
                         effect_mean=float(mean), effect_sd_across_seeds=float(sd),
                         ci_lo=float(lo), ci_hi=float(hi),
                         d_mc=float(mean / sd) if sd > 0 else np.nan,
                         prob_positive=float((d > 0).mean()),
                         p_mc=float(p_mc), p_mc_floor=float(2 / (R + 1)), R=R))
    return rows


# --------------------------------------------------------------------------
# Channels as repeated measures (descriptive; one realisation)
# --------------------------------------------------------------------------
def effective_n(win_matrix: np.ndarray) -> float:
    """Effective number of independent channels from the inter-channel
    correlation of the window-level series.

    n_eff = (sum lambda)^2 / sum lambda^2  (participation ratio of the eigenvalues
    of the correlation matrix); equals 16 for independent channels and 1 for
    perfectly collinear ones.  (Same construction as the effective-number-of-
    variables approaches of Cheverud 2001 / Nyholt 2004 for correlated tests.)
    """
    C = np.corrcoef(win_matrix, rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def channel_paired(a, b, n_eff=None, n_boot=5000, seed=0):
    """Paired comparison of two length-16 channel vectors, treated as repeated
    measures.  Returns exact p-values (never rounded to 0) and effect sizes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    n = diff.size
    md, sd = diff.mean(), diff.std(ddof=1)
    if sd == 0:
        t = p_t = p_adj = np.nan
        dz = np.nan
    else:
        t = md / (sd / np.sqrt(n))
        p_t = 2 * stats.t.sf(abs(t), n - 1)
        dz = md / sd
        p_adj = np.nan
        if n_eff is not None:
            df_eff = max(1.0, n_eff - 1)
            t_eff = md / (sd / np.sqrt(n_eff))          # variance inflation by n / n_eff
            p_adj = 2 * stats.t.sf(abs(t_eff), df_eff)
    try:
        p_w = stats.wilcoxon(a, b, method='auto').pvalue
    except ValueError:
        p_w = np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    ci_boot = np.percentile(boot, [2.5, 97.5])
    tcrit = stats.t.ppf(0.975, n - 1)
    half = tcrit * sd / np.sqrt(n)
    return dict(mean_diff=float(md), sd_diff=float(sd), t=float(t), df=n - 1,
                p_t=float(p_t), p_t_neff=float(p_adj) if n_eff is not None else np.nan,
                n_eff=float(n_eff) if n_eff is not None else np.nan,
                p_wilcoxon=float(p_w), d_z=float(dz),
                ci_t_lo=float(md - half), ci_t_hi=float(md + half),
                ci_boot_lo=float(ci_boot[0]), ci_boot_hi=float(ci_boot[1]))


# --------------------------------------------------------------------------
# Dependence among the eight derived variables
# --------------------------------------------------------------------------
def dependence_report(res: dict):
    """Empirical dependence among the eight derived variables.

    ``res``: output of ``analyse_gamma``; arrays (n_windows, 16).  Returns the
    Spearman and Pearson correlation matrices over all (window, channel) points,
    the eigenvalues of the Pearson matrix, and an effective number of independent
    variables (participation ratio) -- which equals 1 when all eight carry the
    same information.
    """
    X = np.column_stack([res[v].ravel() for v in VARIABLES])
    sp = stats.spearmanr(X).statistic
    pe = np.corrcoef(X, rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(pe)[::-1], 0, None)
    m_eff = float(lam.sum() ** 2 / (lam ** 2).sum())
    return dict(spearman=sp, pearson=pe, eigenvalues=lam, m_eff=m_eff,
                first_pc_share=float(lam[0] / lam.sum()))
