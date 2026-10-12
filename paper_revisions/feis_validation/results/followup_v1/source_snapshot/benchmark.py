"""Run the frozen FEIS experiment and retain trial-level evidence on D:."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import Recording, LABELS, load_recording, preprocess, chronological_split
from .download import DEFAULT_ROOT, PROJECT_ROOT, sha256_file
from .features import METHODS, band_features, feature_columns, training_additions

SEEDS = [17, 29, 43]
PROTOCOL = Path(__file__).with_name('PROTOCOL.md')
RESULTS = PROJECT_ROOT / 'paper_revisions/feis_validation/results/initial_v1'


def classifier():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))


def prepare(root: Path, participant: str, phase: str) -> tuple[Recording, np.ndarray, np.ndarray]:
    source = root / f'extracted/experiments/{participant}/{phase}/{phase}.csv'
    checksum = sha256_file(source)
    manifest = json.loads((root / 'download_manifest.json').read_text(encoding='utf-8'))
    entry = manifest['files'][f'experiments/{participant}/{phase}.zip']
    expected = next(item['sha256'] for item in entry['extracted'] if Path(item['path']).name == f'{phase}.csv')
    if checksum != expected:
        raise ValueError('Source file changed after acquisition')
    version = hashlib.sha256((Path(__file__).with_name('data.py').read_bytes() +
                             Path(__file__).with_name('features.py').read_bytes())).hexdigest()
    cache = root / 'processed' / f'{participant}_{phase}.npz'
    audit_path = cache.with_suffix('.json')
    if cache.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if audit.get('source_sha256') == checksum and audit.get('processing_code_sha256') == version:
            with np.load(cache, allow_pickle=False) as loaded:
                rec = Recording(loaded['x'], loaded['y'], loaded['trial_ids'], loaded['starts'], loaded['flagged'],
                                float(loaded['fs']), audit)
                audit['id_to_block'] = {int(k): v for k,v in audit['id_to_block'].items()}
                return rec, loaded['features'], loaded['raw_ptp']
    rec = load_recording(source, phase)
    raw_ptp = np.ptp(rec.x, axis=-1)
    rec.x = preprocess(rec.x, rec.fs)
    features = band_features(rec.x, rec.fs)
    rec.audit.update(participant=participant, source_sha256=checksum, processing_code_sha256=version,
                     processed_shape=list(rec.x.shape), raw_ptp_median=float(np.median(raw_ptp)),
                     raw_ptp_max=float(raw_ptp.max()), provided_flagged_trials=int(rec.flagged.sum()),
                     raw_flat_trial_count=int(np.sum(np.sum(raw_ptp <= 1e-6, axis=1) >= 2)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=rec.x, y=rec.y, trial_ids=rec.trial_ids, starts=rec.starts,
                        flagged=rec.flagged, fs=rec.fs, features=features, raw_ptp=raw_ptp)
    audit_path.write_text(json.dumps(rec.audit, indent=2), encoding='utf-8')
    return rec, features, raw_ptp


def strict_mask(ptp: np.ndarray, flagged: np.ndarray, train: np.ndarray) -> np.ndarray:
    logs = np.log(np.maximum(ptp, 1e-12))
    median = np.median(logs[train], axis=0)
    spread = np.maximum(1.4826 * np.median(np.abs(logs[train] - median), axis=0), 0.1)
    high = np.sum(logs > median + 6 * spread, axis=1) > 3
    flat = np.sum(ptp <= 1e-6, axis=1) >= 2
    return ~(high | flat | flagged)


def score_one(participant: str, phase: str, rec: Recording, features: np.ndarray,
              train: np.ndarray, test: np.ndarray, screening: str, prediction_stream,
              receipt_stream) -> list[dict]:
    families = ['multiband', 'lower_bands', 'gamma_only'] if screening == 'primary' else ['multiband']
    records = []
    train_ids = set(rec.trial_ids[train].tolist())
    test_ids = set(rec.trial_ids[test].tolist())
    assert not train_ids & test_ids
    for method in METHODS:
        for seed in SEEDS:
            additional, labels, parents = training_additions(rec.x[train], rec.y[train], features[train],
                                                              method, rec.fs, seed)
            assert len(additional) in (0, len(train))
            for target, donor_indices in enumerate(parents):
                ids = rec.trial_ids[train[np.array(donor_indices)]]
                if set(ids.tolist()) & test_ids or not set(ids.tolist()).issubset(train_ids):
                    raise AssertionError('Donor leakage')
                if method in ('same_label_splice', 'same_label_feature_mixup'):
                    if not np.all(rec.y[train[np.array(donor_indices)]] == labels[target]):
                        raise AssertionError('Same-label augmentation used a different label')
                receipt_stream.write(json.dumps({'participant': participant, 'phase': phase, 'screening': screening,
                                                  'method': method, 'seed': seed, 'target_label': LABELS[int(labels[target])],
                                                  'parent_trial_ids': ids.tolist()}) + '\n')
            fit_features = np.concatenate([features[train], additional])
            fit_y = np.concatenate([rec.y[train], labels])
            for family in families:
                columns = feature_columns(family)
                model = classifier().fit(fit_features[:, columns], fit_y)
                predicted = model.predict(features[test][:, columns])
                entry = {'participant': participant, 'phase': phase, 'screening': screening,
                         'feature_family': family, 'method': method, 'seed': seed,
                         'n_real_train': len(train), 'n_synthetic_train': len(additional), 'n_real_test': len(test),
                         'balanced_accuracy': float(balanced_accuracy_score(rec.y[test], predicted)),
                         'accuracy': float(accuracy_score(rec.y[test], predicted)),
                         'macro_f1': float(f1_score(rec.y[test], predicted, labels=np.arange(16), average='macro', zero_division=0))}
                records.append(entry)
                output = pd.DataFrame({'participant': participant, 'phase': phase, 'screening': screening,
                                       'feature_family': family, 'method': method, 'seed': seed,
                                       'trial_id': rec.trial_ids[test], 'true_label': [LABELS[i] for i in rec.y[test]],
                                       'predicted_label': [LABELS[i] for i in predicted]})
                output.to_csv(prediction_stream, index=False, header=False)
    return records


def permutation_null(participant: str, phase: str, rec: Recording, features: np.ndarray,
                     train: np.ndarray, test: np.ndarray, count: int) -> list[dict]:
    rng = np.random.default_rng(20261003 + int(participant))
    blocks = np.array([rec.audit['id_to_block'][int(item)] for item in rec.trial_ids[train]])
    records = []
    for draw in range(count):
        y = rec.y[train].copy()
        for block in np.unique(blocks):
            selected = np.flatnonzero(blocks == block)
            y[selected] = rng.permutation(y[selected])
        predicted = classifier().fit(features[train], y).predict(features[test])
        records.append({'participant': participant, 'phase': phase, 'draw': draw,
                        'balanced_accuracy': float(balanced_accuracy_score(rec.y[test], predicted))})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=RESULTS)
    parser.add_argument('--participants', nargs='+', default=[f'{i:02}' for i in range(1,22)])
    parser.add_argument('--permutations', type=int, default=100)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    large_outputs = args.root / 'analysis' / output.name
    large_outputs.mkdir(parents=True, exist_ok=True)
    stamp = {'started_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha256_file(PROTOCOL),
             'protocol_path': str(PROTOCOL), 'data_root': str(args.root), 'participants': args.participants,
             'permutations_per_participant_phase': args.permutations, 'seeds': SEEDS,
             'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                          'pandas': pd.__version__, 'sklearn': sklearn.__version__},
             'source_code_sha256': {p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
             'predictions_path': str(large_outputs/'predictions.csv'),
             'donor_receipts_path': str(large_outputs/'donor_receipts.jsonl'),
             'status': 'running', 'audit_only': args.audit_only}
    (output/'run_manifest.json').write_text(json.dumps(stamp, indent=2), encoding='utf-8')
    scores, null, audits, issues = [], [], [], []
    predictions_path = large_outputs/'predictions.csv'
    receipts_path = large_outputs/'donor_receipts.jsonl'
    with threadpool_limits(limits=1), predictions_path.open('w', encoding='utf-8', newline='') as predictions, receipts_path.open('w', encoding='utf-8') as receipts:
        predictions.write('participant,phase,screening,feature_family,method,seed,trial_id,true_label,predicted_label\n')
        for participant in args.participants:
            pair = {}
            for phase in ('thinking', 'resting'):
                rec, features, ptp = prepare(args.root, participant, phase)
                train, test = chronological_split(rec)
                if pair:
                    previous = pair['thinking']
                    previous_train, previous_test = chronological_split(previous)
                    if previous.audit['id_to_block'] != rec.audit['id_to_block']:
                        raise ValueError('Phases disagree on original trial/block identities')
                    if (set(previous.trial_ids[previous_train]) & set(rec.trial_ids[test]) or
                            set(previous.trial_ids[previous_test]) & set(rec.trial_ids[train])):
                        raise AssertionError('Related task phases cross the holdout boundary')
                pair[phase] = rec
                keep = strict_mask(ptp, rec.flagged, train)
                rec.audit.update(train_trial_ids=rec.trial_ids[train].tolist(), test_trial_ids=rec.trial_ids[test].tolist(),
                                 strict_train_kept=int(keep[train].sum()), strict_test_kept=int(keep[test].sum()))
                audits.append(rec.audit)
                print(f'{participant} {phase}: {len(rec.y)} trials, {rec.audit["blocks"]} blocks, {len(train)} train/{len(test)} test; strict {keep[train].sum()}/{keep[test].sum()}', flush=True)
                if not args.audit_only:
                    scores.extend(score_one(participant, phase, rec, features, train, test, 'primary', predictions, receipts))
                    null.extend(permutation_null(participant, phase, rec, features, train, test, args.permutations))
                    strict_train, strict_test = train[keep[train]], test[keep[test]]
                    if len(np.unique(rec.y[strict_train])) == 16 and len(np.unique(rec.y[strict_test])) == 16 and min(np.bincount(rec.y[strict_train], minlength=16)) >= 2:
                        scores.extend(score_one(participant, phase, rec, features, strict_train, strict_test, 'strict', predictions, receipts))
                    else:
                        issues.append({'participant': participant, 'phase': phase, 'scope': 'strict sensitivity', 'reason': 'Insufficient class coverage after fixed screening; no fallback tuning'})
                pd.DataFrame(scores).to_csv(output/'scores.csv', index=False)
                pd.DataFrame(null).to_csv(output/'permutation_scores.csv', index=False)
                (output/'data_audit.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
                (output/'issues.json').write_text(json.dumps(issues, indent=2), encoding='utf-8')
            for trial in set(pair['thinking'].trial_ids) & set(pair['resting'].trial_ids):
                left = np.flatnonzero(pair['thinking'].trial_ids == trial)[0]
                right = np.flatnonzero(pair['resting'].trial_ids == trial)[0]
                if pair['thinking'].y[left] != pair['resting'].y[right]:
                    raise ValueError('Thinking/rest labels disagree for a shared trial')
    stamp.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                 score_rows=len(scores), null_rows=len(null), issue_count=len(issues),
                 predictions_sha256=sha256_file(predictions_path), donor_receipts_sha256=sha256_file(receipts_path))
    (output/'run_manifest.json').write_text(json.dumps(stamp, indent=2), encoding='utf-8')
    print(f'Completed: {output}', flush=True)


if __name__ == '__main__':
    main()
