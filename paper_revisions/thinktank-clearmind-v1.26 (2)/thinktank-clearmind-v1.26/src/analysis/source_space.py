"""
Self-contained source imaging for the synthetic-vs-real consistency analysis.

Why this exists
---------------
The 3-D "electrical activity" figures use MNE-Python with the fsaverage BEM
(src/viz/visualizer.py).  MNE-Python, the fsaverage template and network access
were not available in the environment used to prepare the v1.1.0 numerical
results, so the *consistency analysis* (Reviewer #3, point 2) uses this
dependency-free equivalent instead, which mirrors the settings of the
visualizer wherever a sphere model allows:

  =====================  ==================================  ===========================
  setting                visualizer.py (MNE / fsaverage)     this module
  =====================  ==================================  ===========================
  head model             3-layer BEM, ico-3                  homogeneous sphere, analytic
  source space           ico4 cortical surface (2 x 2562)    ~400 sources, cortical shell
  orientation            surface normal, loose = 0.2 (auto)  radial (fixed) dipoles
  reference              average (projection)                average (projection)
  depth weighting        0.8 (MNE default)                   0.8
  noise covariance       'shrunk' if n < 12000 else          Ledoit-Wolf if n < 12000
                         'empirical'                         else empirical
  inverse                dSPM, lambda2 = 1/9                 dSPM, lambda2 = 1/9
  =====================  ==================================  ===========================

Absolute source maps therefore differ from MNE's; comparisons *between* synthetic
and real recordings, computed with the same operator, are what this module is for.

The homogeneous-sphere surface potential of a current dipole is the classical
series (Brody et al., 1973; Mosher et al., 1999):

    V(theta) = 1/(4 pi sigma R^2) * sum_n (2n+1)/n * f^(n-1) *
               [ n p_r P_n(cos theta) + p_t sin(theta) P_n'(cos theta) ]      f = r0 / R

validated in tests/test_distance_and_source.py against (i) the centre-dipole closed form
3 p cos(theta) / (4 pi sigma R^2) and (ii) the boundary-doubling limit for a
near-surface dipole (potential -> 2x the infinite-medium value).
"""
from __future__ import annotations

import numpy as np

# Unit-sphere 10-20 coordinates used by the visualisation module (EEG_Plotting.py):
# x anterior, y left, z up.
ELECTRODE_XYZ = {
    'Fp1': (0.95, 0.309, -0.0349), 'Fp2': (0.95, -0.309, -0.0349),
    'F7': (0.587, 0.809, -0.0349), 'F3': (0.673, 0.545, 0.5),
    'F4': (0.673, -0.545, 0.5), 'F8': (0.587, -0.809, -0.0349),
    'T3': (0.0, 0.999, -0.0349), 'C3': (0.0, 0.719, 0.695),
    'C4': (0.0, -0.719, 0.695), 'T4': (0.0, -0.999, -0.0349),
    'T5': (-0.587, 0.809, -0.0349), 'P3': (-0.672, 0.545, 0.5),
    'P4': (-0.673, -0.545, 0.5), 'T6': (-0.587, -0.809, -0.0349),
    'O1': (-0.95, 0.309, -0.0349), 'O2': (-0.95, -0.309, -0.0349),
}
CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'T5', 'T6', 'O1', 'O2',
            'F7', 'F8', 'C3', 'C4', 'T3', 'T4', 'P3', 'P4']


def electrode_positions(R=1.0):
    xyz = np.array([ELECTRODE_XYZ[c] for c in CHANNELS], float)
    return R * xyz / np.linalg.norm(xyz, axis=1, keepdims=True)


def _legendre_and_derivative(x, n_max):
    """P_n(x) and P_n'(x) for n = 0..n_max via the standard recurrences."""
    x = np.atleast_1d(x).astype(float)
    P = np.zeros((n_max + 1,) + x.shape)
    dP = np.zeros_like(P)
    P[0] = 1.0
    if n_max >= 1:
        P[1] = x
        dP[1] = 1.0
    for n in range(2, n_max + 1):
        P[n] = ((2 * n - 1) * x * P[n - 1] - (n - 1) * P[n - 2]) / n
        dP[n] = n * P[n - 1] + x * dP[n - 1]      # P_n' = n P_{n-1} + x P_{n-1}'
    return P, dP


# Sign convention of the tangential term, fixed by tests/test_distance_and_source.py so that the
# potential rises on the side of the electrode the tangential moment points towards.
_TANGENTIAL_SIGN = 1.0


def sphere_potential(src_pos, p_radial, p_tangential, elec_pos, R=1.0, n_terms=300):
    """Surface potential (units of 1/(4 pi sigma R^2) = 1) at ``elec_pos`` for one
    dipole at ``src_pos`` with radial / tangential moments.  The tangential
    direction is the unit vector, in the plane containing source and electrode,
    that points from the source's radial axis towards the electrode."""
    src_pos = np.asarray(src_pos, float)
    r0 = np.linalg.norm(src_pos)
    f = r0 / R
    u0 = src_pos / r0
    e = np.asarray(elec_pos, float)
    ue = e / np.linalg.norm(e, axis=-1, keepdims=True)
    cos_t = np.clip(ue @ u0, -1, 1)
    sin_t = np.sqrt(1 - cos_t ** 2)
    P, dP = _legendre_and_derivative(cos_t, n_terms)
    n = np.arange(1, n_terms + 1)[:, None]
    w = (2 * n + 1) / n * f ** (n - 1)
    radial = (w * n * P[1:]).sum(axis=0)                  # sum (2n+1) f^(n-1) P_n(cos t)
    tangential = _TANGENTIAL_SIGN * (w * sin_t * dP[1:]).sum(axis=0)
    return p_radial * radial + p_tangential * tangential


def cortical_shell_sources(n_points=700, radius=0.85, z_min=-0.35):
    """Quasi-uniform (Fibonacci) points on a shell, restricted to z >= z_min."""
    i = np.arange(n_points) + 0.5
    phi = np.arccos(1 - 2 * i / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * i
    xyz = np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])
    xyz = xyz[xyz[:, 2] >= z_min]
    return radius * xyz


def gain_matrix(src_xyz, elec_xyz=None, n_terms=200):
    """(n_elec, n_src) lead field for radial dipoles."""
    elec = electrode_positions() if elec_xyz is None else elec_xyz
    G = np.empty((elec.shape[0], src_xyz.shape[0]))
    for j, s in enumerate(src_xyz):
        G[:, j] = sphere_potential(s, 1.0, 0.0, elec, n_terms=n_terms)
    return G


class SourceImager:
    """Average-reference, depth-weighted dSPM (lambda2 = 1/9) on a sphere model."""

    def __init__(self, n_sources=700, lambda2=1.0 / 9.0, depth=0.8):
        self.src = cortical_shell_sources(n_sources)
        self.G = gain_matrix(self.src)
        n = self.G.shape[0]
        self.P = np.eye(n) - np.ones((n, n)) / n          # average-reference projector
        self.Gp = self.P @ self.G
        self.lambda2 = lambda2
        self.depth = depth
        norms = np.linalg.norm(self.Gp, axis=0)
        w = norms ** (-2.0 * depth)
        self.r_diag = w / w.mean()                        # relative source variances

    @staticmethod
    def noise_cov(x, method=None):
        """Covariance of mean-subtracted data (rows = samples)."""
        xc = x - x.mean(axis=0, keepdims=True)
        if method is None:
            method = "shrunk" if x.shape[0] < 12000 else "empirical"
        if method == "shrunk":
            from sklearn.covariance import LedoitWolf
            return LedoitWolf().fit(xc).covariance_
        return np.cov(xc, rowvar=False)

    def operator(self, cov):
        """dSPM inverse operator (n_src, n_ch) for average-referenced data."""
        cp = self.P @ cov @ self.P
        lam, V = np.linalg.eigh(cp)
        keep = lam > lam.max() * 1e-10                    # rank n_ch - 1 after re-referencing
        Wh = (V[:, keep] / np.sqrt(lam[keep])).T          # whitener, (m, n_ch)
        Gw = Wh @ self.Gp                                 # whitened lead field, (m, n_src)
        m = Gw.shape[0]
        # relative source variances, scaled so that trace(Gw R Gw') = m (MNE convention)
        Rd = self.r_diag * m / float((self.r_diag * (Gw ** 2).sum(axis=0)).sum())
        A = (Gw * Rd[None, :]) @ Gw.T + self.lambda2 * np.eye(m)
        M = (Rd[:, None] * Gw.T) @ np.linalg.inv(A)       # (n_src, m) minimum-norm map
        noise_sens = np.sqrt((M ** 2).sum(axis=1))        # whitened noise has identity covariance
        return (M @ Wh) / noise_sens[:, None]             # dSPM

    def power_map(self, x, remove_dc_per_block=None, cov=None):
        """Mean squared dSPM value per source for data ``x`` (n_samples, 16)."""
        xa = x.copy()
        if remove_dc_per_block:
            b = remove_dc_per_block
            n = (xa.shape[0] // b) * b
            xa = xa[:n]
            blocks = xa.reshape(-1, b, xa.shape[1])
            xa = (blocks - blocks.mean(axis=1, keepdims=True)).reshape(-1, xa.shape[1])
        C = self.noise_cov(xa) if cov is None else cov
        W = self.operator(C)
        xr = xa @ self.P.T                                 # average reference
        s = xr @ W.T                                       # (n_samples, n_src)
        return (s ** 2).mean(axis=0)
