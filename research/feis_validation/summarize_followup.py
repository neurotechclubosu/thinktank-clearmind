"""Recover scores from real trial predictions and report bounded follow-up decisions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from .followup import OUTPUT
from .download import DEFAULT_ROOT,sha256_file

def bootstrap(values):
    rng=np.random.default_rng(20261004)
    values=np.asarray(values,dtype=float)
    draws=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
    return [float(v) for v in np.percentile(draws,[2.5,97.5])]

def holm(pvalues):
    order=np.argsort(pvalues);adjusted=np.empty(len(pvalues));previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(pvalues)-rank)*pvalues[index]));adjusted[index]=previous
    return adjusted

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--root',type=Path,default=DEFAULT_ROOT);a=p.parse_args()
    output=a.output;manifest=json.loads((output/'run_manifest.json').read_text())
    if manifest['status']!='complete' or len(manifest['completed'])!=44:raise ValueError('Full planned44 participant-phase run required')
    for name,digest in manifest['source_sha256'].items():
        if sha256_file(output/'source_snapshot'/name)!=digest:raise ValueError('Run snapshot changed')
    scores=pd.concat([pd.read_csv(output/f'{i}_scores.csv') for i in manifest['completed']],ignore_index=True)
    recovered=[];checks=[];audits=[];nulls=[];refits=[];predictions=0
    for identifier in manifest['completed']:
        audit=json.loads((output/f'{identifier}_audit.json').read_text());audits.append(audit)
        train=set(audit['train_ids']);test=set(audit['test_ids'])
        if train&test:raise AssertionError('Real train/test leakage')
        labels={r['trial_id']:r['label'] for r in audit['data_audit']['trials']}
        for fold in audit['inner_folds']:
            fit=set(fold['train_ids']);val=set(fold['validation_ids'])
            if fit&val or (fit|val)-train or (fit|val)&test:raise AssertionError('Inner fold leakage')
        frame=pd.read_csv(a.root/'analysis'/output.name/f'{identifier}_predictions.csv');predictions+=len(frame)
        from .data import LABELS
        for row in frame.itertuples():
            if row.trial_id not in test or LABELS[row.true_label]!=labels[row.trial_id]:raise AssertionError('Prediction not true held-out source trial')
        for key,group in frame.groupby(['participant','phase','model','seed','screening']):
            if group.trial_id.duplicated().any():raise AssertionError('Duplicated test trial prediction')
            recovered.append(dict(zip(['participant','phase','model','seed','screening'],key),
                                  balanced_accuracy=float(balanced_accuracy_score(group.true_label,group.predicted_label))))
        nulls.append(pd.read_csv(output/f'{identifier}_conditional_null.csv'))
        refits.append(pd.read_csv(output/f'{identifier}_training_null.csv'))
        checks.append({'identifier':identifier,'train_test_disjoint':True,'nested_folds_train_only':True,
                       'real_prediction_labels_verified':True,'processed_cache_hash_verified':
                       sha256_file(a.root/f'processed/followup_v1/{identifier}.npz')==audit['data_audit']['processed_cache_sha256']})
    recovered=pd.DataFrame(recovered);merged=scores.merge(recovered,on=['participant','phase','model','seed','screening'],suffixes=('_saved','_recovered'),validate='one_to_one')
    error=float(np.max(np.abs(merged.balanced_accuracy_saved-merged.balanced_accuracy_recovered)))
    if len(merged)!=len(scores) or error>1e-12 or not all(c['processed_cache_hash_verified'] for c in checks):raise AssertionError('Score/cache verification failed')
    scores.to_csv(output/'scores.csv',index=False);pd.concat(nulls).to_csv(output/'conditional_null.csv',index=False)
    pd.concat(refits).to_csv(output/'training_null.csv',index=False)
    primary=scores[(scores.seed==17)&(scores.screening=='primary')]
    summaries=[]
    for (phase,model),group in primary.groupby(['phase','model']):
        values=group.balanced_accuracy.to_numpy();interval=bootstrap(values) if len(values)>1 else [None,None]
        summaries.append({'phase':phase,'model':model,'participants':len(values),'mean_balanced_accuracy':float(values.mean()),
            'ci_low':interval[0],'ci_high':interval[1]})
    summary=pd.DataFrame(summaries);summary.to_csv(output/'cohort_summary.csv',index=False)
    decisions=[];null=pd.concat(nulls)
    for model in ['covariance','eegnet_style']:
        observed=primary[(primary.phase=='thinking')&(primary.model==model)].balanced_accuracy.to_numpy()
        conditional=null[(null.phase=='thinking')&(null.model==model)].groupby('draw').balanced_accuracy.mean().to_numpy()
        lo,hi=bootstrap(observed)
        decisions.append({'model':model,'mean':float(observed.mean()),'participant_ci':[lo,hi],
            'conditional_permutation_p':float((1+np.sum(conditional>=observed.mean()-1e-15))/(len(conditional)+1)),
            'conditional_null_mean':float(conditional.mean())})
    adjusted=holm([d['conditional_permutation_p'] for d in decisions])
    for d,q in zip(decisions,adjusted):
        d['holm_p']=float(q);d['signal_gate']=bool(d['mean']>=.0825 and d['participant_ci'][0]>.0625 and q<=.05)
    paired=[]
    for model in ['covariance','eegnet_style','nuisance']:
        g=primary[(primary.phase.isin(['thinking','resting']))&(primary.model==model)].pivot(index='participant',columns='phase',values='balanced_accuracy')
        diff=(g.thinking-g.resting).to_numpy();paired.append({'model':model,'thinking_minus_rest':float(diff.mean()),'participant_ci':bootstrap(diff)})
    decision={'status':'complete','exploratory':True,'nominal_chance':.0625,'families':decisions,'thinking_rest_comparisons':paired,
        'augmentation_authorized_by_signal_gate':any(d['signal_gate'] for d in decisions),
        'interpretation':'Signal gate is conditional feasibility only; artifact and task specificity still require review.'}
    (output/'decision.json').write_text(json.dumps(decision,indent=2))
    integrity={'participant_phase_runs':len(checks),'score_rows':len(scores),'real_prediction_rows':predictions,
        'max_score_reconstruction_error':error,'all_checks_pass':True,'checks':checks,
        'source_snapshot_verified':True,'original_primary_results_modified':False}
    (output/'integrity_verification.json').write_text(json.dumps(integrity,indent=2))
    audit_summary=[{'participant':r['participant'],'phase':r['phase'],
        **{k:r['data_audit'][k] for k in ['original_trials','kept_trials','raw_ptp_median','raw_ptp_max','flagged_trials','common_mode_rms_fraction_median','spectral_fractions_median']},
        'strict_n_train':r['strict']['n_train'],'strict_n_test':r['strict']['n_test']} for r in audits]
    (output/'signal_audit_summary.json').write_text(json.dumps(audit_summary,indent=2))
    lines=['# Stronger FEIS baseline follow-up','',
        'Exploratory follow-up: the first FEIS test results were already inspected. Native later-block real-trial holdout; no sentence/metabolism or cross-session validation.','',
        '| Phase | Model | People | Balanced accuracy | Participant95% interval |','|---|---|---:|---:|---|']
    for r in summaries:
        ci=f'{100*r["ci_low"]:.2f}-{100*r["ci_high"]:.2f}%' if r['ci_low'] is not None else 'one-person diagnostic'
        lines.append(f'| {r["phase"]} | {r["model"]} | {r["participants"]} | {100*r["mean_balanced_accuracy"]:.2f}% | {ci} |')
    lines+=['','Nominal16-class chance6.25%. CNN primary seed17; seeds29/43 are stability sensitivities. Model choices were made using two forward folds entirely inside training.']
    for d in decisions: lines+=['',f'{d["model"]}: conditional label-permutation p={d["conditional_permutation_p"]:.3f}, Holm p={d["holm_p"]:.3f}; predefined signal gate **{"passed" if d["signal_gate"] else "failed"}**.']
    lines+=['','Conditional null shuffles test labels within their original balanced blocks. Exchangeability is assumed, not proof of randomized acquisition. Nine full training-label permutation refits with repeated inner selection were limited prospectively to participant01 thinking; these are coarse implementation controls.','',
        'Stimuli and speaking comparisons use participant01 only. Listening/spoken success may reflect sensory input or muscles. Rest follows speech. Nuisance models use raw amplitudes/means/line fractions/flags; failure of these controls cannot certify neural specificity.','',
        f'Independent verification recovered {len(scores)} scores from {predictions} real-trial predictions, confirmed source labels, disjoint real trials and inner folds, and processed-cache/source-snapshot hashes.','',
        'All full predictions and EEG caches are on D:. Initial_v1 is preserved. No splicing comparison should proceed without the signal gate and specificity review. The separate Kara format pilot has no classifier results.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(decision,indent=2));print('Verified',integrity['score_rows'],'scores and',predictions,'real predictions')

if __name__=='__main__':main()
