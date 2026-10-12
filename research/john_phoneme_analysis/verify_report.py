"""Independently replay the frozen evaluation, summarize it, and draw figures.

Does not import the scoring implementation or fit any additional model family.
Raw epochs, prediction tables, and full nulls stay on D:. No public release.
"""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import argparse
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = Path('D:/ResearchData/thinktank-clearmind/John/2026-10-05')
RUN = DATA / 'analysis/phoneme_repeatability_v1'
OUT = REPO / 'paper_revisions/john_phoneme_analysis'
SUBJECTS = ['GTP000', 'GTP003', 'GTP004', 'GTP007']
CHANCE = 1 / 44
EXPECTED = {'GTP000': [1, 3, 4, 5, 6, 7], 'GTP003': [1, 2],
            'GTP004': [1, 3], 'GTP007': [1, 2, 3]}
KEYS = {('task', 'spectral'): 'task_spectral',
        ('task', 'covariance'): 'task_covariance',
        ('wait', 'spectral'): 'wait_spectral',
        ('wait', 'covariance'): 'wait_covariance',
        ('no_car', 'spectral'): 'no_car_spectral',
        ('no_car', 'covariance'): 'no_car_covariance',
        ('harmonic', 'spectral'): 'harmonic_spectral',
        ('timing', 'timing'): 'timing',
        ('screened', 'spectral'): 'task_spectral',
        ('screened', 'covariance'): 'task_covariance',
        ('different_day', 'spectral'): 'task_spectral',
        ('different_day', 'covariance'): 'task_covariance'}


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def independent_centroids(train, labels, queries):
    # Explicit population SD, zero-variance handling, and broadcast distances.
    # No call to analyze.prediction, its scaler, or its distance routine.
    avg = train.mean(axis=0)
    sd = train.std(axis=0, ddof=0)
    sd[sd == 0] = 1
    z = (train - avg) / sd
    centers = np.stack([z[labels == k].mean(axis=0) for k in range(44)])
    q = (queries - avg) / sd
    distance = ((q[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return distance.argmin(axis=1), distance


def verify(run):
    ext = json.loads((run / 'extraction.json').read_text())
    evaluation = json.loads((run / 'evaluation.json').read_text())
    assert ext['status'] == evaluation['status'] == 'PASS'
    for key, filename in [('features_sha256', 'features.npz'), ('meta_sha256', 'epochs.csv'),
                          ('source_manifest_sha256', 'complete_session_recordings.csv')]:
        assert sha(run / filename) == ext[key]
    assert ext['code_sha256'] == evaluation['code_sha256'] == sha(run / 'analyze_snapshot.py')
    assert ext['protocol_sha256'] == evaluation['protocol_sha256'] == sha(run / 'PROTOCOL.md')
    assert sha(REPO / 'research/john_phoneme_analysis/analyze.py') == ext['code_sha256']
    assert sha(REPO / 'research/john_phoneme_analysis/PROTOCOL.md') == ext['protocol_sha256']
    sources = pd.read_csv(run / 'complete_session_recordings.csv')
    receipts = pd.read_csv(run / 'source_receipts.csv').set_index('file')
    m = pd.read_csv(run / 'epochs.csv')
    scores = pd.read_csv(run / 'scores.csv')
    p = pd.read_csv(run / 'predictions.csv')
    status = pd.read_csv(run / 'model_status.csv')
    f = np.load(run / 'features.npz')
    nulls = np.load(run / 'alignment_nulls.npz')
    perms = np.load(run / 'label_permutations.npz')
    assert len(sources) == len(receipts) == 572
    assert len(m) == 2860 and len(scores) == 96 and len(p) == 21120
    assert m.full_task_hash.nunique() == m.central_task_hash.nunique() == 2860
    assert not p.duplicated(['condition', 'family', 'epoch_index']).any()
    assert len(status) == 40 and (status.status == 'PASS').all()
    assert (m.task_end - m.task_start == 250).all()
    assert (m.wait_end - m.wait_start == 250).all()
    assert (m.task_start - m.wait_end == 125).all()
    assert (m.wait_start >= 0).all()
    for sub, sessions in EXPECTED.items():
        g = m[m.subject == sub]
        assert sorted(g.session.unique()) == sessions
        for ses in sessions:
            block = g[g.session == ses]
            assert len(block) == 220 and block.file.nunique() == 44
            assert sorted(block['index'].unique()) == list(range(44))
            assert (block.groupby('index').size() == 5).all()
        tr, te = g[g.session == 1], g[g.session != 1]
        assert not set(tr.file) & set(te.file)
        assert not set(tr.full_task_hash) & set(te.full_task_hash)
    assert (m.session == 1).sum() == 880
    assert (m.session != 1).sum() == 1980
    # Reverify every original source against both extraction receipts.
    for row in sources.itertuples(index=False):
        assert sha(DATA / row.path) == row.sha256 == receipts.loc[row.file, 'sha256']
    print('Verified every source and all ownership/count/receipt invariants.', flush=True)

    # Direct sample/window/filter/feature checks: first and last file of each
    # person's first and last session, selected by index before feature inspection.
    source_samples = sources[
        sources.apply(lambda r: int(r.block) in [1, EXPECTED[r.subject][-1]], axis=1)
        & sources.phoneme_index.isin([0, 43])]
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    filter_sos = signal.butter(4, [1, 45], fs=250, btype='bandpass', output='sos')
    checked_epochs = 0
    for r in source_samples.itertuples(index=False):
        a = np.loadtxt(DATA / r.path)
        for i in m.index[m.file == r.file]:
            row = m.loc[i]
            start = int(row.wait_end)
            end = start + int(row.samples)
            raw = np.ascontiguousarray(a[start:end, 1:17], dtype='<f8')
            assert hashlib.sha256(raw.tobytes()).hexdigest() == row.full_task_hash
            raw_central = np.ascontiguousarray(a[int(row.task_start):int(row.task_end), 1:17], dtype='<f8')
            assert hashlib.sha256(raw_central.tobytes()).hexdigest() == row.central_task_hash
            whole = signal.detrend(raw.T, axis=-1, type='linear')
            whole -= whole.mean(axis=0, keepdims=True)
            x = signal.sosfiltfilt(filter_sos, whole, axis=-1)[:, 125:375]
            freq, psd = signal.welch(x, fs=250, window='hann', nperseg=250,
                                     noverlap=0, detrend='constant', axis=-1)
            denominator = psd[:, (freq >= 1) & (freq < 45)].sum(axis=1)
            powers = np.column_stack([
                np.log(psd[:, (freq >= lo) & (freq < hi)].sum(axis=1) / denominator)
                for lo, hi in bands]).ravel()
            np.testing.assert_allclose(powers, f['task_spectral'][i], atol=1e-10)
            centered = x - x.mean(axis=1, keepdims=True)
            cov = centered @ centered.T / 249
            cov = .9 * cov / (np.trace(cov) / 16) + .1 * np.eye(16)
            value, vector = np.linalg.eigh(cov)
            log_cov = (vector * np.log(value)) @ vector.T
            ii, jj = np.triu_indices(16)
            upper = log_cov[ii, jj] * np.where(ii == jj, 1, np.sqrt(2))
            np.testing.assert_allclose(upper, f['task_covariance'][i], atol=1e-10)
            checked_epochs += 1
    assert checked_epochs == 80
    print('Re-extracted80 fixed source task windows and both feature families.', flush=True)

    recomputed_scores = []
    for (condition, family, sub), pred_rows in p.groupby(['condition', 'family', 'subject']):
        if condition == 'different_day':
            train_ids = m.index[(m.subject == sub) & m.session.isin([1, 3, 4])].to_numpy()
            expected_test_ids = m.index[(m.subject == sub) & m.session.isin([5, 6, 7])].to_numpy()
        else:
            train_ids = m.index[(m.subject == sub) & (m.session == 1)].to_numpy()
            expected_test_ids = m.index[(m.subject == sub) & (m.session != 1)].to_numpy()
        if condition == 'screened':
            mid = np.median(f['quality'][train_ids], axis=0)
            mad = np.median(np.abs(f['quality'][train_ids] - mid), axis=0)
            upper = mid + 6 * 1.4826 * np.maximum(mad, 1e-8)
            keep = (f['quality'] <= upper).all(axis=1)
            train_ids = train_ids[keep[train_ids]]
            expected_test_ids = expected_test_ids[keep[expected_test_ids]]
        pred_rows = pred_rows.sort_values('epoch_index')
        test_ids = pred_rows.epoch_index.to_numpy()
        np.testing.assert_array_equal(test_ids, np.sort(expected_test_ids))
        query_meta = m.loc[test_ids]
        for pred_name, meta_name in [('subject', 'subject'), ('session', 'session'),
                                     ('file', 'file'), ('marker', 'marker'), ('true', 'index')]:
            np.testing.assert_array_equal(pred_rows[pred_name].to_numpy(), query_meta[meta_name].to_numpy())
        labels = m.loc[train_ids, 'index'].to_numpy()
        features = f[KEYS[(condition, family)]]
        rebuilt, distances = independent_centroids(features[train_ids], labels, features[test_ids])
        np.testing.assert_array_equal(rebuilt, pred_rows.prediction.to_numpy())
        for ses in sorted(pred_rows.session.unique()):
            mask = pred_rows.session.to_numpy() == ses
            truths = pred_rows.true.to_numpy()[mask]
            guesses = rebuilt[mask]
            # Recover each saved BA from class-specific recall, independently.
            recalls = [np.mean(guesses[truths == k] == k) for k in range(44)]
            accuracy = float(np.mean(recalls))
            file_predictions = np.array([
                distances[mask & (pred_rows.true.to_numpy() == k)].mean(axis=0).argmin()
                for k in range(44)])
            file_score = float(np.mean(file_predictions == np.arange(44)))
            saved = scores[(scores.condition == condition) & (scores.family == family)
                           & (scores.subject == sub) & (scores.session == ses)]
            assert len(saved) == 1
            np.testing.assert_allclose(accuracy, saved.balanced_accuracy.iloc[0], atol=1e-12)
            np.testing.assert_allclose(file_score, saved.file_mean_accuracy.iloc[0], atol=1e-12)
            assert len(train_ids) == saved.train.iloc[0] and mask.sum() == saved.test.iloc[0]
            recomputed_scores.append((condition, family, sub, int(ses)))
        if condition == 'task':
            key = f'{family}_{sub}'
            draw = perms[key]
            assert draw.shape == (999, 44)
            np.testing.assert_array_equal(np.sort(draw, axis=1), np.tile(np.arange(44), (999, 1)))
            conditional = []
            for ses in sorted(pred_rows.session.unique()):
                q = pred_rows.session.to_numpy() == ses
                assert np.all(np.bincount(pred_rows.true.to_numpy()[q], minlength=44) == 5)
                conditional.append(np.mean(draw[:, rebuilt[q]] == pred_rows.true.to_numpy()[q], axis=1))
            np.testing.assert_allclose(np.mean(conditional, axis=0), nulls[key], atol=1e-12)
    assert len(recomputed_scores) == len(scores)
    print('Independently replayed all21120 predictions,96 scores and8 conditional nulls.', flush=True)
    receipt = dict(status='PASS',verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   source_hashes=572, unique_full_task_hashes=2860, independent_feature_checks=80,
                   replayed_predictions=len(p), replayed_session_scores=len(scores), null_arrays=8,
                   train_recalls=880, heldout_recalls=1980, participants=4, independent_test_sessions=9,
                   code_sha256=sha(Path(__file__)),
                   files={name: sha(run / name) for name in ['features.npz', 'epochs.csv', 'scores.csv',
                       'predictions.csv', 'alignment_nulls.npz', 'label_permutations.npz', 'evaluation.json']})
    return m, scores, p, nulls, receipt


def summarize(scores, predictions, nulls):
    participant = scores.groupby(['condition', 'family', 'subject'], as_index=False).agg(
        balanced_accuracy=('balanced_accuracy', 'mean'), file_mean_accuracy=('file_mean_accuracy', 'mean'),
        test_sessions=('session', 'size'), scored_recalls=('test', 'sum'))
    primary = []
    group_nulls = {}
    rng = np.random.default_rng(20261006)
    draws = rng.integers(0, 4, size=(10000, 4))
    for family in ['spectral', 'covariance']:
        group = participant[(participant.condition == 'task') & (participant.family == family)].set_index('subject')
        value = group.loc[SUBJECTS, 'balanced_accuracy'].to_numpy()
        null = np.stack([nulls[f'{family}_{sub}'] for sub in SUBJECTS]).mean(axis=0)
        group_nulls[family] = null
        observed = float(value.mean())
        ci = np.percentile(value[draws].mean(axis=1), [2.5, 97.5]).tolist()
        # Small tolerance counts mathematically equal scores as ties.
        probability = float((1 + (null >= observed - 1e-12).sum()) / 1000)
        per_sub = {}
        for sub, v in zip(SUBJECTS, value):
            per_sub[sub] = dict(accuracy=float(v), conditional_p=float(
                (1 + (nulls[f'{family}_{sub}'] >= v - 1e-12).sum()) / 1000))
        primary.append(dict(family=family, mean_balanced_accuracy=observed,
                            bootstrap_ci=ci, conditional_p=probability,
                            above_chance_participants=int((value > CHANCE + 1e-12).sum()),
                            file_mean_accuracy=float(group.file_mean_accuracy.mean()),
                            participants=per_sub))
    order = np.argsort([r['conditional_p'] for r in primary])
    running = 0
    for position, index in enumerate(order):
        running = max(running, min(1., (2 - position) * primary[index]['conditional_p']))
        primary[index]['holm_p'] = running
    for row in primary:
        row['alignment_gate_pass'] = bool(row['mean_balanced_accuracy'] >= CHANCE + .05
            and row['above_chance_participants'] >= 3 and row['holm_p'] < .05)
    means = participant.groupby(['condition', 'family'], as_index=False).agg(
        mean_balanced_accuracy=('balanced_accuracy', 'mean'), file_mean_accuracy=('file_mean_accuracy', 'mean'),
        participants=('subject', 'size'))
    differences = []
    for family in ['spectral', 'covariance']:
        task = participant[(participant.condition == 'task') & (participant.family == family)].set_index('subject').loc[SUBJECTS, 'balanced_accuracy'].to_numpy()
        for control, control_family in [('wait', family), ('timing', 'timing')]:
            compare = participant[(participant.condition == control) & (participant.family == control_family)].set_index('subject').loc[SUBJECTS, 'balanced_accuracy'].to_numpy()
            delta = task - compare
            differences.append(dict(family=family, control=control,
                mean_task_minus_control_pp=float(100 * delta.mean()),
                bootstrap_ci_pp=(100 * np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])).tolist(),
                participant_differences_pp={sub: float(100 * v) for sub, v in zip(SUBJECTS, delta)}))
    # Descriptive class collapse; no extra fitted baseline or selected features.
    collapse = []
    for (family, sub, ses), g in predictions[predictions.condition == 'task'].groupby(['family', 'subject', 'session']):
        counts = g.prediction.value_counts()
        collapse.append(dict(family=family, subject=sub, session=int(ses), predicted_classes=len(counts),
                             most_frequent_index=int(counts.index[0]),
                             largest_prediction_fraction=float(counts.iloc[0] / len(g))))
    summary = dict(exploratory=True, chance=CHANCE, gate_accuracy=CHANCE + .05,
                   participant_count=4, primary=primary, group_conditions=means.to_dict('records'),
                   task_control_differences=differences, bootstrap_draws=10000, conditional_draws=999,
                   interpretation='Neither fixed task family meets the limited alignment gate. No augmentation/composition experiment is justified by this run.')
    return summary, participant, means, pd.DataFrame(collapse), group_nulls


def figures(out, scores, participant, means):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    colors = {'task': '#155c8d', 'wait': '#b85c38', 'timing': '#656b74'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, family in zip(axes, ['spectral', 'covariance']):
        for j, condition in enumerate(['task', 'wait', 'timing']):
            sf = 'timing' if condition == 'timing' else family
            g = participant[(participant.condition == condition) & (participant.family == sf)].set_index('subject').loc[SUBJECTS]
            ax.plot(np.arange(4), g.balanced_accuracy.to_numpy() * 100,
                    'o-', color=colors[condition], label=condition.capitalize(), lw=1.2)
        ax.axhline(CHANCE * 100, color='black', ls='--', lw=1, label='Chance (2.27%)')
        ax.set_xticks(range(4), SUBJECTS)
        ax.set_ylim(0, 6)
        ax.set_title(f'{family.capitalize()} features')
        ax.grid(axis='y', alpha=.18)
        ax.set_xlabel('Participant (session means within each person)')
    axes[0].set_ylabel('Held-out 44-class balanced accuracy (%)')
    axes[1].legend(loc='upper right', frameon=False, fontsize=9)
    fig.suptitle('Task alignment does not consistently exceed waits or timing', fontsize=13)
    fig.text(.5, .015, 'Four participants; train session1, test nine later complete sessions. No cue-only or EOG/EMG ground truth.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .05, 1, .93])
    fig.savefig(out / 'task_controls.png', dpi=180)
    fig.savefig(out / 'task_controls.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    for ax, family in zip(axes, ['spectral', 'covariance']):
        conditions = ['task', 'no_car', 'screened'] + (['harmonic'] if family == 'spectral' else [])
        labels = ['Primary', 'No CAR', 'Screened'] + (['Harmonic bins\nomitted'] if family == 'spectral' else [])
        for j, condition in enumerate(conditions):
            values = participant[(participant.condition == condition) & (participant.family == family)].set_index('subject').loc[SUBJECTS, 'balanced_accuracy'].to_numpy() * 100
            ax.scatter(j + np.linspace(-.12, .12, 4), values, s=24, color='#9aa8b5')
            ax.plot(j, values.mean(), 'D', color='#155c8d', markersize=7)
        ax.axhline(CHANCE * 100, color='black', ls='--', lw=1)
        ax.axhline((CHANCE + .05) * 100, color='#8b3939', ls=':', lw=1)
        ax.set_ylim(0, 8)
        ax.set_xticks(range(len(conditions)), labels)
        ax.set_title(f'{family.capitalize()} features')
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Participant-equal balanced accuracy (%)')
    fig.suptitle('Fixed reference, screening and harmonic checks remain near chance', fontsize=13)
    fig.text(.5, .025, 'Dots: participant means. Diamonds: equal-person means. Dashed: 2.27% chance. Dotted: 7.27% pilot floor.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .07, 1, .93])
    fig.savefig(out / 'sensitivities.png', dpi=180)
    fig.savefig(out / 'sensitivities.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
    sessions = [(sub, int(s)) for sub in SUBJECTS for s in EXPECTED[sub] if s != 1]
    for ax, family in zip(axes, ['spectral', 'covariance']):
        g = scores[(scores.condition == 'task') & (scores.family == family)].set_index(['subject', 'session'])
        v = np.array([g.loc[key, 'balanced_accuracy'] for key in sessions]) * 100
        ax.bar(range(9), v, color='#155c8d', width=.7)
        ax.axhline(CHANCE * 100, color='black', ls='--', lw=1)
        ax.set_xticks(range(9), [f'{sub}\ns{s}' for sub, s in sessions], rotation=55, ha='right', fontsize=8)
        ax.set_ylim(0, 6)
        ax.set_title(f'{family.capitalize()} features')
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Held-out session balanced accuracy (%)')
    fig.suptitle('All nine held-out sessions, including adverse outcomes', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(out / 'session_results.png', dpi=180)
    fig.savefig(out / 'session_results.pdf')
    plt.close(fig)


def report(out, run, summary, participant, means, collapse, receipt):
    def pct(x):
        return f'{100 * x:.2f}%'
    lines = [
        '# Independent-session analysis of John’s EEG delivery', '',
        '6 October 2026. Exploratory author-review analysis; no manuscript replacement, submission or public deposit.', '',
        '**The fixed analysis did not find useful repeatability of the 44 recorded stimulus indices across sessions. Both primary pipelines remain near the 2.27% chance level and fail the predeclared pilot criterion. These results do not support moving into composition/augmentation testing with this bank under this design.**', '',
        'The recordings provide a stronger test of the bank than the original single-session demonstration: four reported people, 13 structurally complete sessions, 572 source files and 2,860 unique marked recall intervals. They do not provide continuous imagined sentences, cue-only recordings, separate EOG/EMG controls or independent metabolic measurements.', '',
        '## Primary result', '',
        '| Fixed family | Mean balanced accuracy | Descriptive participant-bootstrap 95% interval | Conditional alignment p | Holm p | People above chance | Pilot gate |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in summary['primary']:
        lo, hi = r['bootstrap_ci']
        lines.append(f"| {r['family'].capitalize()} | {pct(r['mean_balanced_accuracy'])} | [{pct(lo)}, {pct(hi)}] | {r['conditional_p']:.3f} | {r['holm_p']:.3f} | {r['above_chance_participants']}/4 | {'Pass' if r['alignment_gate_pass'] else 'Fail'} |")
    lines += ['',
        'Each participant’s first complete session supplies 220 training recalls. All remaining complete sessions are held out: 1,980 real recalls in nine sessions. Five recalls from a file always stay together. Preprocessing and class centroids fit no test observations. All 44 labels are retained. Results first average sessions within a person, then weight the four people equally; John’s extra sessions therefore do not dominate the mean.', '',
        'The locally frozen criterion requires at least 7.27% group accuracy, at least three of four people above chance, and Holm p<0.05 across the two primary families. The five-percentage-point practical floor was chosen for this pilot; it is not a publication standard or clinical utility threshold. The bootstrap resamples four people 10,000 times, so its interval is discrete and unstable. It is not population validation. The 999-permutation test is conditional on exchangeability of class identities, not a test that a signal is neural or caused by imagination.', '',
        '| Participant | Held-out sessions | Spectral task | Covariance task | Spectral wait | Covariance wait | Timing only |',
        '|---|---|---:|---:|---:|---:|---:|']
    lookup = participant.set_index(['condition', 'family', 'subject'])
    for sub in SUBJECTS:
        vals = [lookup.loc[(c, f, sub), 'balanced_accuracy'] for c, f in [('task', 'spectral'), ('task', 'covariance'), ('wait', 'spectral'), ('wait', 'covariance'), ('timing', 'timing')]]
        lines.append(f"| {sub} | {', '.join(str(s) for s in EXPECTED[sub] if s != 1)} | " + ' | '.join(map(pct, vals)) + ' |')
    lines += ['', '![Task and controls](task_controls.png)', '',
        '## Controls and fixed sensitivities', '',
        '| Condition | Family | Equal-person accuracy | File-mean accuracy |',
        '|---|---|---:|---:|']
    for r in means.itertuples(index=False):
        if r.condition != 'different_day':
            lines.append(f'| {r.condition} | {r.family} | {pct(r.mean_balanced_accuracy)} | {pct(r.file_mean_accuracy)} |')
    lines += ['', 'File-mean accuracy is the same fixed centroid model applied to the mean feature vector of each held-out file; it is not an additional biological replicate.', '']
    for d in summary['task_control_differences']:
        low, high = d['bootstrap_ci_pp']
        lines.append(f"- {d['family'].capitalize()} task minus {d['control']}: {d['mean_task_minus_control_pp']:+.2f} percentage points; descriptive participant-bootstrap interval [{low:+.2f}, {high:+.2f}].")
    lines += ['',
        'The task does not outperform these controls on the participant-equal mean. Waits may contain carryover or an early visual cue and are not a separately collected cue-only baseline. Timing features use acquisition rank, marker ordinal, interval duration, sample count and packet counter. Their near-chance outcome does not prove absence of every timing/order confound.', '',
        'Dropping CAR, omitting predefined harmonic neighborhoods and applying the training-only amplitude/derivative screen do not reach the pilot floor. The screen rejects **zero** recalls; its threshold is a relative outlier diagnostic, not evidence that these recordings are artifact-free. Removing selected spectral bins is also not proof of removal of interference. No filters or thresholds were tuned after results.', '',
        '![Fixed sensitivities](sensitivities.png)', '',
        'John’s different-day diagnostic trains sessions 1,3,4 from 21 March and tests 5,6,7 from 23 March. It is a separate single-person result, not a fifth person or an independent cohort:', '']
    for r in means[means.condition == 'different_day'].itertuples(index=False):
        lines.append(f'- {r.family.capitalize()}: {pct(r.mean_balanced_accuracy)} recall accuracy; {pct(r.file_mean_accuracy)} file-mean accuracy.')
    lines += ['', '## Signal and provenance limitations', '',
        'Original units, gain, reference, processing history and exact executed stimulus-deck identity remain unverified. The earlier fixed raw traces show pronounced periodic waveforms and harmonic peaks near multiples of 10 Hz, as well as substantial common signal. Their cause has not been diagnosed. Relative spectral power and scale-normalized covariance reduce dependence on absolute gain but do not establish neural signal quality.', '',
        'The covariance model often assigns most or all test recalls in a session to one index. This is documented in `prediction_class_use.csv`; no classes were removed or model family substituted to hide it. Such predictions can achieve nominal chance by guessing one class in a balanced session. A near-chance aggregate should not be mistaken for a weak but validated decoder.', '']
    constant = collapse[(collapse.family == 'covariance') & (collapse.predicted_classes == 1)]
    lines.append(f'{len(constant)}/9 covariance test sessions predict exactly one class. All per-session scores are shown below.')
    lines += ['', '![All sessions](session_results.png)', '',
        'Exact duplicated/misplaced files and truncated files were excluded through the author-confirmation manifest before fitting. The entire affected incomplete session is omitted, rather than replacing its absent index with another session’s copy. All raw archives and files remain preserved. Duplicate ownership is supported by recorded timestamps, not by an original acquisition log.', '',
        'The primary label is the recorded integer index, not a fully validated distinct phoneme interpretation. The supplied 16-slide object deck and recovered 91-slide phoneme candidate are still not established as the same executed stimulus version. Even a correct phoneme mapping would leave visual-cue and instructed-imagination effects confounded.', '',
        '## Consequence for the paper', '',
        'This run supplies an independent-session check that the original bank lacked. Its contribution is a transparent test and a limit on the bank’s demonstrated usefulness, not improved biological reconstruction performance. There is no evidence here that a correctly ordered phoneme splice resembles a genuinely imagined sentence, or that composition improves prediction of held-out real EEG.', '',
        'The appropriate present conclusion is: **“In a fixed exploratory analysis of four participants, relative spectral and covariance centroid classifiers did not establish reliable 44-index alignment across held-out sessions. These results limit the demonstrated utility of the recorded bank for label-dependent EEG composition under the tested design.”** Do not write “phonemes contain no EEG information” or generalize this failure to all imagined-speech methods.', '',
        'The prior FEIS and Kara analyses remain separate adverse results; this delivery does not supersede them or create a successful replication. Any revised paper should acknowledge the sequence of exploratory analyses and avoid presenting the newest dataset as a prospectively independent confirmation of a favorable effect.', '',
        'The Rothman/DiNuzzo physiology framework remains theoretical context for an illustrative model. More EEG recordings do not calibrate the imposed gamma-to-CMRO2 input or independently validate CBF, oxygen extraction, pH, gases or lactate. The downstream variables share one deterministic driver and cannot be counted as independent measured physiological outcomes.', '',
        'The current v1.27 source-space outputs also use an inherited channel assignment that differs from John’s confirmed order in 14 of 16 positions. Regenerate those outputs and electrode labels before relying on them; this index-level analysis does not repair or endorse those manuscript figures.', '',
        '## Recommended next action', '',
        '1. **Review acquisition quality before collecting or modeling more.** Resolve units/gain/reference/processing and investigate the periodic raw pattern against the equipment and acquisition logs. These results cannot distinguish poor recording quality from weak or poorly captured task information. Do not start another open-ended classifier search to obtain a positive score.',
        '2. **Choose a defensible manuscript question with the authors.** The existing data support a construction demonstration with a candid independent-session feasibility failure and explicit model assumptions. That is narrower than biological speech or metabolism reconstruction, and its publication value still needs a concrete reproducibility or simulation use. No acceptance guarantee follows from a careful negative analysis.',
        '3. **If biological reconstruction remains the aim, design the required study.** Use matched isolated units and independently recorded continuous sequences, randomized order, verified event timing, cue-only and eye/muscle controls, repeats on separate days and a justified participant/precision plan. Test acquisition and a fixed real-data baseline first. Metabolic calibration requires independent physiological measurements in a separate justified study.',
        '4. **Resolve the remaining submission requirements.** Exact stimulus crosswalk, four-person demographics/consent/reuse, approved data/code release and corrected spatial figures remain outstanding. Neither the manuscript nor reviewer response was silently upgraded by this analysis.', '',
        '## Reproduction and verification', '',
        'Protocol: `research/john_phoneme_analysis/PROTOCOL.md`, saved before classifier scores on this delivery. This is a local exploratory plan, not external preregistration; original session1, raw signal examples and prior public-dataset results had already been inspected. Two fixed feature families and a transparent nearest-centroid classifier were used, with no hyperparameter search or results-based sample selection.', '',
        f'Detailed data, epochs, feature cache, predictions and nulls: `{run.as_posix()}`. Original archives and source recordings remain on D:. Only summaries and figures are in this repository.', '',
        '```powershell',
        'python -m unittest discover -s research/john_phoneme_analysis -p test_analysis.py -v',
        '# First two stages refuse overwriting a completed run:',
        'python research/john_phoneme_analysis/analyze.py extract',
        'python research/john_phoneme_analysis/analyze.py evaluate',
        '# Replay verification and regenerate derivative reports:',
        'python research/john_phoneme_analysis/verify_report.py',
        '```', '',
        f"Independent verification: **{receipt['source_hashes']} source SHA-256 receipts**, **{receipt['unique_full_task_hashes']} unique full-task signals**, **{receipt['independent_feature_checks']} fixed source windows re-extracted**, **{receipt['replayed_predictions']:,} saved predictions replayed**, **{receipt['replayed_session_scores']} session scores and eight conditional null arrays recovered**. A labeled positive fixture, train-only scaling isolation, actual label-refit/null equivalence and feature-scale/filter invariants pass the five unit tests.", '',
        'Artifacts: `summary.json`, `participant_results.csv`, `session_results.csv`, `condition_summary.csv`, `prediction_class_use.csv`, `screen_thresholds.csv`, `verification.json`, three PNG/PDF figure pairs, and this report. Saved predictions in repeated controls are not additional biological samples.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    m, scores, p, nulls, receipt = verify(args.run)
    summary, participant, means, collapse, group_nulls = summarize(scores, p, nulls)
    assert not any(r['alignment_gate_pass'] for r in summary['primary']), 'Report prose is specific to this frozen failed run.'
    args.out.mkdir(parents=True, exist_ok=True)
    participant.to_csv(args.out / 'participant_results.csv', index=False)
    scores.to_csv(args.out / 'session_results.csv', index=False)
    means.to_csv(args.out / 'condition_summary.csv', index=False)
    collapse.to_csv(args.out / 'prediction_class_use.csv', index=False)
    pd.read_csv(args.run / 'screen_thresholds.csv').to_csv(args.out / 'screen_thresholds.csv', index=False)
    write_json(args.out / 'summary.json', summary)
    # Full null distribution remains with the data, not in a submission bundle.
    np.savez_compressed(args.run / 'group_nulls.npz', **group_nulls)
    figures(args.out, scores, participant, means)
    report(args.out, args.run, summary, participant, means, collapse, receipt)
    receipt['figure_inspection'] = 'PENDING: inspect all three PNGs before marking work complete'
    receipt['report_sha256'] = sha(args.out / 'REPORT.md')
    write_json(args.out / 'verification.json', receipt)
    write_json(args.run / 'verification.json', receipt)
    print(json.dumps(summary['primary'], indent=2), flush=True)
    print('Report and figures generated; visual inspection remains.', flush=True)


if __name__ == '__main__':
    main()
