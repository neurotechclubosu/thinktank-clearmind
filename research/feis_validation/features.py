"""Fixed spectral features and training-only augmentation with donor receipts."""
from __future__ import annotations

import numpy as np
from scipy import signal

BANDS = [('delta', 1, 4), ('theta', 4, 8), ('alpha', 8, 13), ('beta', 13, 30), ('low_gamma', 30, 45)]
METHODS = ['real_only', 'duplicate_real', 'scale_jitter', 'same_label_feature_mixup', 'same_label_splice', 'random_label_splice', 'shuffle_blocks']


def band_features(x: np.ndarray, fs: float) -> np.ndarray:
    f, psd = signal.welch(x, fs=fs, nperseg=int(fs), noverlap=0, axis=-1)
    step = f[1] - f[0]
    powers = np.stack([psd[:, :, (f >= lo) & (f < hi)].sum(axis=-1) * step for _, lo, hi in BANDS], axis=-1)
    return np.log(np.maximum(powers, 1e-12)).reshape(len(x), -1)


def feature_columns(family: str, channels: int = 14) -> np.ndarray:
    grid = np.arange(channels * len(BANDS)).reshape(channels, len(BANDS))
    if family == 'multiband':
        return grid.ravel()
    if family == 'lower_bands':
        return grid[:, :4].ravel()
    if family == 'gamma_only':
        return grid[:, 4]
    raise ValueError(f'Unknown family {family}')


def training_additions(x: np.ndarray, y: np.ndarray, features: np.ndarray,
                       method: str, fs: float, seed: int) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Indices in receipts refer exclusively to the passed training array."""
    rng = np.random.default_rng(seed)
    n, _, samples = x.shape
    if method == 'real_only':
        return features[:0], y[:0], []
    if method in ('duplicate_real', 'shuffle_blocks'):
        if method == 'duplicate_real':
            return features.copy(), y.copy(), [[i] for i in range(n)]
        blocks = x.reshape(n, x.shape[1], samples // int(fs), int(fs))
        shuffled = np.empty_like(blocks)
        for i in range(n):
            shuffled[i] = np.take(blocks[i], rng.permutation(blocks.shape[2]), axis=1)
        return band_features(shuffled.reshape(x.shape), fs), y.copy(), [[i] for i in range(n)]
    if method == 'scale_jitter':
        scales = x.std(axis=(0, 2), keepdims=True)
        gain = rng.uniform(0.9, 1.1, size=(n, 1, 1))
        synthetic = gain * x + rng.normal(size=x.shape) * scales * 0.01
        return band_features(synthetic, fs), y.copy(), [[i] for i in range(n)]
    receipts = []
    if method == 'same_label_feature_mixup':
        synthetic_features = np.empty_like(features)
        for i, label in enumerate(y):
            pool = np.flatnonzero(y == label)
            if len(pool) < 2:
                raise ValueError('Fewer than two same-label donors')
            donors = rng.choice(pool, size=2, replace=False)
            weight = rng.beta(0.4, 0.4)
            synthetic_features[i] = weight * features[donors[0]] + (1-weight) * features[donors[1]]
            receipts.append(donors.tolist())
        return synthetic_features, y.copy(), receipts
    if method not in ('same_label_splice', 'random_label_splice'):
        raise ValueError(f'Unknown augmentation {method}')
    width = int(fs)
    if samples % width:
        raise ValueError('Noninteger number of blocks')
    nblocks = samples // width
    synthetic = np.empty_like(x)
    for i, label in enumerate(y):
        pool = np.flatnonzero(y == label) if method == 'same_label_splice' else np.arange(n)
        if len(pool) < 2:
            raise ValueError('Fewer than two donors')
        donors = rng.choice(pool, size=nblocks, replace=True)
        donors[:2] = rng.choice(pool, size=2, replace=False)
        receipts.append(donors.tolist())
        for block, donor in enumerate(donors):
            first = block * width
            synthetic[i, :, first:first+width] = x[donor, :, first:first+width]
    return band_features(synthetic, fs), y.copy(), receipts
