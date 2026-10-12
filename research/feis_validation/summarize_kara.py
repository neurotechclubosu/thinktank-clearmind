"""Independently recover Kara real-trial scores and keep controls visible."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import balanced_accuracy_score
from .kara_benchmark import OUTPUT
from .kara_download import ROOT
from .download import sha256_file
from .summarize_followup import bootstrap,holm

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    output=args.output;root=args.root;manifest=json.loads((output/'run_manifest.json').read_text())
    if manifest['status']!='complete' or len(manifest['completed'])!=13:raise ValueError('Complete predefined13-person independent run required')
    for name,digest in manifest['source_sha256'].items():
        if sha256_file(output/'source_snapshot'/name)!=digest:raise ValueError('Frozen run snapshot mismatch')
    scores=pd.concat([pd.read_csv(output/f'{p}_scores.csv') for p in manifest['completed']],ignore_index=True)
    null=pd.concat([pd.read_csv(output/f'{p}_conditional_null.csv') for p in manifest['completed']],ignore_index=True)
    recovered=[];checks=[];prediction_count=0
    for participant in manifest['completed']:
        audit=json.loads((output/f'{participant}_audit.json').read_text());frame=pd.read_csv(root/'analysis'/output.name/f'{participant}_predictions.csv')
        train=set(audit['train_ids']);test=set(audit['test_ids'])
        if train&test:raise AssertionError('Primary Kara real trial leakage')
        for fold in audit['inner_folds']:
            fit=set(fold['train_ids']);val=set(fold['validation_ids'])
            if fit&val or (fit|val)-train or (fit|val)&test:raise AssertionError('Primary nested leakage')
        from .kara_data import WORDS
        native={r['trial_id']:WORDS.index(r['prompt']) for r in audit['trials'] if r['used_four_word_task']}
        for key,group in frame.groupby(['participant','model','seed','screening']):
            screening=key[-1];allowed=test
            if screening=='global_chronology':
                global_plan=audit['global_chronology'];gtrain=set(global_plan['train_ids']);allowed=set(global_plan['test_ids'])
                if gtrain&allowed or max(gtrain)>=min(allowed):raise AssertionError('Global chronology failed')
                for fold in global_plan['inner_folds']:
                    f=set(fold['train_ids']);v=set(fold['validation_ids'])
                    if f&v or (f|v)-gtrain or (f|v)&allowed:raise AssertionError('Global inner leakage')
            elif screening=='strict':allowed=set(audit['strict']['test_ids'])
            if set(group.trial_id)!=allowed or group.trial_id.duplicated().any():raise AssertionError('Missing/duplicate/misassigned real test predictions')
            if any(native[int(r.trial_id)]!=int(r.true_label) for r in group.itertuples()):raise AssertionError('Published real trial labels differ')
            recovered.append(dict(zip(['participant','model','seed','screening'],key),balanced_accuracy=float(balanced_accuracy_score(group.true_label,group.predicted_label))))
        prediction_count+=len(frame)
        cache=root/'processed'/f'{participant}_four_word.npz'
        if sha256_file(cache)!=audit['cache_sha256']:raise AssertionError('EEG cache changed')
        coefficients=root/'analysis'/output.name/f'{participant}_eye_regression.npz'
        if sha256_file(coefficients)!=audit['eye_regression_coefficients_sha256']:raise AssertionError('Eye regression receipt changed')
        checks.append({'participant':participant,'real_labels_verified':True,'primary_and_global_trials_disjoint':True,
                       'inner_selection_train_only':True,'cache_and_regression_hashes_verified':True})
    merged=scores.merge(pd.DataFrame(recovered),on=['participant','model','seed','screening'],suffixes=('_saved','_recovered'),validate='one_to_one')
    error=float(np.max(np.abs(merged.balanced_accuracy_saved-merged.balanced_accuracy_recovered)))
    if len(merged)!=len(scores) or error>1e-12:raise AssertionError('Recovered scores differ')
    summary=[]
    for (model,seed,screening),group in scores.groupby(['model','seed','screening']):
        lo,hi=bootstrap(group.balanced_accuracy)
        summary.append({'model':model,'seed':int(seed),'screening':screening,'participants':len(group),
            'mean':float(group.balanced_accuracy.mean()),'ci_low':lo,'ci_high':hi,'real_test_predictions':int(group.n_test.sum())})
    table=pd.DataFrame(summary);table.to_csv(output/'cohort_summary.csv',index=False)
    scores.to_csv(output/'scores.csv',index=False);null.to_csv(output/'conditional_null.csv',index=False)
    decisions=[]
    for model in ['covariance','eegnet_style']:
        row=table[(table.model==model)&(table.seed==17)&(table.screening=='primary')].iloc[0]
        draws=null[null.model==model].groupby('draw').balanced_accuracy.mean().to_numpy()
        decisions.append({'model':model,'mean':float(row['mean']),'participant_ci':[float(row.ci_low),float(row.ci_high)],
            'conditional_permutation_p':float((1+np.sum(draws>=row['mean']-1e-15))/(len(draws)+1))})
    adjusted=holm([d['conditional_permutation_p'] for d in decisions])
    for d,q in zip(decisions,adjusted):
        d['holm_p']=float(q);d['signal_gate']=bool(d['mean']>=.30 and d['participant_ci'][0]>.25 and q<=.05)
        sensitivities={}
        for screening in ['eye_regression','global_chronology','strict']:
            group=table[(table.model==d['model'])&(table.seed==17)&(table.screening==screening)]
            if len(group):sensitivities[screening]=group.iloc[0].to_dict()
        d['sensitivities']=sensitivities
    eye=table[(table.model=='eye_only')&(table.screening=='primary')].iloc[0].to_dict()
    nuisance=table[(table.model=='amplitude_nuisance')&(table.screening=='primary')].iloc[0].to_dict()
    decision={'status':'complete','format_only_pilot_excluded':'MM05','participants':13,'nominal_chance':.25,
        'families':decisions,'eye_only_control':eye,'amplitude_nuisance_control':nuisance,
        'any_signal_gate_passed':any(d['signal_gate'] for d in decisions),
        'additional_splicing_test':'not_run_because_both_signal_gates_failed' if not any(d['signal_gate'] for d in decisions) else 'requires_separate_prospective_augmentation_plan_and_specificity_review',
        'needs_specificity_review_before_augmentation':True,
        'no_sentence_or_metabolism_validation':True,'interpretation':'Numerical gate indicates bounded method feasibility only. Eye/mouth artifacts and cue effects remain separate questions.'}
    (output/'decision.json').write_text(json.dumps(decision,indent=2))
    integrity={'all_checks_pass':True,'score_rows':len(scores),'real_prediction_rows':prediction_count,
        'max_score_reconstruction_error':error,'source_snapshot_verified':True,'checks':checks}
    (output/'integrity_verification.json').write_text(json.dumps(integrity,indent=2))
    lines=['# Kara One independent four-word baseline evaluation','',
        'MM05 was a format-only pilot, excluded.13 new people, four words: gnaw, knew, pat, pot; chance25%. All recordings are public secondary data on D:. This is a different task from FEIS, not exact replication and not full-sentence reconstruction.','',
        '| Model | Condition | Seed | People | Balanced accuracy | Participant95% interval |','|---|---|---:|---:|---:|---|']
    for r in summary:
        lines.append(f'| {r["model"]} | {r["screening"]} | {r["seed"]} | {r["participants"]} | {100*r["mean"]:.2f}% | {100*r["ci_low"]:.2f}-{100*r["ci_high"]:.2f}% |')
    lines+=['','Primary decisions use seed17 and Holm correction across two families; other seeds/controls/sensitivities are secondary. The primary split is later within each class and can interleave across classes; the global chronological sensitivity has genuinely later real test epochs and its own training-only selection.','']
    for d in decisions:lines.append(f'{d["model"]}: conditional permutation p={d["conditional_permutation_p"]:.3f}, Holm p={d["holm_p"]:.3f}; provisional>=30%/CI-above25% gate **{"passed" if d["signal_gate"] else "failed"}**.\n')
    lines+=['Conditional permutations assume exchangeable labels within the held-out participant portion; they do not establish neural specificity. Eye-only and raw-amplitude controls, training-fitted eye regression and chronology sensitivities remain visible. A negative eye-only result cannot rule out other artifacts; no direct muscle channel is verified. The channel called EMG is a task-colour sensor and is excluded from predictors.','',
        'Acquisition/label checks were conducted before classifier scores, including colour-sensor corroboration where repeats made text alignment ambiguous. Supplied imagery intervals are task periods, not measured internal speech onsets. Source/calibration assumptions and omissions are in participant audits.','',
        f'Independent verification recovered {len(scores)} scores from {prediction_count} real predictions, checked native labels, trial/fold isolation, cached EEG and eye-regression receipts, and source snapshots.','',
        'No results support calibrated CMRO2 or continuous sentence biology. No paper, collaborator message, journal submission or public deposit was changed by this experiment.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(decision,indent=2));print('Verified',len(scores),'scores and',prediction_count,'real predictions')

if __name__=='__main__':main()
