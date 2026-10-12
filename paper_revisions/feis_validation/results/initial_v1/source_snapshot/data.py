"""Validate FEIS CSV trial boundaries and prepare fixed signal transformations."""
from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

CHANNELS = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
LABELS = sorted(['fleece', 'goose', 'trap', 'thought', 'm', 'n', 'ng', 'f', 's', 'sh', 'v', 'z', 'zh', 'p', 't', 'k'])


@dataclass
class Recording:
    x: np.ndarray
    y: np.ndarray
    trial_ids: np.ndarray
    starts: np.ndarray
    flagged: np.ndarray
    fs: float
    audit: dict


def load_recording(path: Path, phase: str) -> Recording:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    match = re.fullmatch(r'Time:([0-9.]+)Hz', header[0])
    if match is None or header[2:16] != CHANNELS:
        raise ValueError(f'Unrecognized time/channel header: {path}')
    fs = float(match.group(1))
    samples = int(5 * fs)
    if fs != 256:
        raise ValueError('This frozen experiment requires the verified 256 Hz format')
    frame = pd.read_csv(path, dtype={name: 'float32' for name in CHANNELS})
    if set(frame.Stage.dropna().unique()) != {phase}:
        raise ValueError('Unexpected task phase')
    ids = frame.Epoch.to_numpy()
    cuts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    observed = ids[cuts[:-1]]
    if len(np.unique(observed)) != len(observed) or np.any(np.diff(observed) <= 0):
        raise ValueError('Trial IDs repeat, interleave or are unordered')
    if len(observed) % 16:
        raise ValueError('Incomplete chronological 16-trial block')
    metadata = []
    xs, ys, kept_ids, starts, flagged = [], [], [], [], []
    block_labels = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        part = frame.iloc[start:end]
        labels = part.Label.unique()
        if len(labels) != 1 or labels[0] not in LABELS:
            raise ValueError('Missing or changing trial label')
        label = labels[0]
        block_labels.append(label)
        times = part[header[0]].to_numpy(dtype='float64')
        values = part[CHANNELS].to_numpy(dtype='float32').T
        flags = sorted(set(part.Flag.dropna().astype(str)) - {''})
        problems = []
        if len(part) != samples:
            problems.append('incorrect_samples')
        elif not np.allclose(np.diff(times), 1 / fs, rtol=0, atol=1e-7):
            problems.append('nonuniform_time')
        if not np.isfinite(values).all():
            problems.append('nonfinite')
        flat = int(np.sum(np.ptp(values, axis=-1) <= 1e-6)) if np.isfinite(values).all() else 14
        if flat == 14:
            problems.append('completely_flat')
        entry = {'trial_id': int(part.Epoch.iloc[0]), 'label': label, 'samples': len(part),
                 'start': float(times[0]), 'end': float(times[-1]), 'flag_values': flags,
                 'flat_channels': flat, 'excluded_reasons': problems}
        metadata.append(entry)
        if not problems:
            xs.append(values)
            ys.append(LABELS.index(label))
            kept_ids.append(entry['trial_id'])
            starts.append(times[0])
            flagged.append(bool(flags))
    for first in range(0, len(block_labels), 16):
        if sorted(block_labels[first:first+16]) != LABELS:
            raise ValueError(f'Unbalanced/mismapped block at ordinal {first}')
    original_ids = observed.astype(int)
    id_to_block = {int(item): i // 16 for i, item in enumerate(original_ids)}
    audit = {'source': str(path), 'phase': phase, 'sampling_rate_hz': fs, 'channels': CHANNELS,
             'physical_units': 'Not independently verified; native numeric units retained',
             'original_trials': len(observed), 'kept_trials': len(xs), 'blocks': len(observed)//16,
             'label_counts': {label: int(sum(x == label for x in block_labels)) for label in LABELS},
             'id_to_block': id_to_block, 'trials': metadata}
    return Recording(np.stack(xs), np.array(ys), np.array(kept_ids), np.array(starts), np.array(flagged), fs, audit)


def preprocess(x: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(x, dtype='float64')
    x = x - x.mean(axis=-1, keepdims=True)
    x = x - x.mean(axis=1, keepdims=True)
    b, a = signal.iirnotch(50, 30, fs=fs)
    x = signal.filtfilt(b, a, x, axis=-1)
    sos = signal.butter(4, [1, 90], fs=fs, btype='bandpass', output='sos')
    x = signal.sosfiltfilt(sos, x, axis=-1)
    margin = int(0.5 * fs)
    return x[:, :, margin:-margin].astype('float32')


def chronological_split(recording: Recording) -> tuple[np.ndarray, np.ndarray]:
    nblocks = recording.audit['blocks']
    first_test = int(np.floor(0.7 * nblocks))
    if first_test < 3 or nblocks - first_test < 2:
        raise ValueError('Insufficient complete blocks for the frozen evaluation')
    blocks = np.array([recording.audit['id_to_block'][int(i)] for i in recording.trial_ids])
    train = np.flatnonzero(blocks < first_test)
    test = np.flatnonzero(blocks >= first_test)
    for indices in (train, test):
        if len(np.unique(recording.y[indices])) != 16:
            raise ValueError('A class is missing from the training or test split')
    if len(set(recording.trial_ids[train]) & set(recording.trial_ids[test])):
        raise AssertionError('Trial leakage')
    return train, test


def strict_screen(x: np.ndarray, flagged: np.ndarray, train: np.ndarray) -> np.ndarray:
    ptp = np.ptp(x, axis=-1)
    logs = np.log(np.maximum(ptp, 1e-12))
    center = np.median(logs[train], axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(logs[train] - center), axis=0), 0.1)
    high = np.sum(logs > center + 6 * scale, axis=1) > 3
    flat = np.sum(ptp <= 1e-6, axis=1) >= 2
    return ~(high | flat | flagged)
