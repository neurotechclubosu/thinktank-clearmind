"""Summarize participant-level results without treating synthetic draws as people."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .benchmark import RESULTS, PROTOCOL
from .download import sha256_file
from .features import METHODS

NAMES = {'real_only':'Real only', 'duplicate_real':'Duplicate real', 'scale_jitter':'Scale and jitter',
         'same_label_feature_mixup':'Feature mixup', 'same_label_splice':'Phoneme splice',
         'random_label_splice':'Random-label splice', 'shuffle_blocks':'Shuffle blocks'}


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261003)
    means = values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
    lo,hi = np.quantile(means,[.025,.975])
    return float(values.mean()),float(lo),float(hi)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=RESULTS)
    args=parser.parse_args()
    root=args.results
    manifest=json.loads((root/'run_manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete' or manifest['audit_only']:
        raise ValueError('Only a complete classification run may be summarized')
    if manifest['protocol_sha256'] != sha256_file(PROTOCOL):
        raise ValueError('The protocol changed after the recorded run')
    scores=pd.read_csv(root/'scores.csv',dtype={'participant':str})
    null=pd.read_csv(root/'permutation_scores.csv',dtype={'participant':str})
    audits=json.loads((root/'data_audit.json').read_text(encoding='utf-8'))
    participant=scores.groupby(['participant','phase','screening','feature_family','method'],as_index=False).balanced_accuracy.mean()
    participant.to_csv(root/'participant_scores.csv',index=False)
    rows=[]
    for keys,group in participant.groupby(['phase','screening','feature_family','method']):
        mean,lo,hi=mean_interval(group.balanced_accuracy)
        rows.append(dict(zip(['phase','screening','feature_family','method'],keys),participants=len(group),mean=mean,ci_low=lo,ci_high=hi))
    summary=pd.DataFrame(rows)
    summary.to_csv(root/'cohort_summary.csv',index=False)
    comparisons=[]
    for keys,group in participant.groupby(['phase','screening','feature_family']):
        pivot=group.pivot(index='participant',columns='method',values='balanced_accuracy')
        for comparator in ['real_only','duplicate_real','scale_jitter','same_label_feature_mixup','random_label_splice','shuffle_blocks']:
            delta=(pivot.same_label_splice-pivot[comparator]).dropna()
            mean,lo,hi=mean_interval(delta)
            comparisons.append(dict(zip(['phase','screening','feature_family'],keys),comparator=comparator,
                                    participants=len(delta),mean_difference=mean,ci_low=lo,ci_high=hi,
                                    participants_improved=int((delta>1e-12).sum()),participants_worse=int((delta < -1e-12).sum())))
    comparisons=pd.DataFrame(comparisons)
    comparisons.to_csv(root/'paired_comparisons.csv',index=False)
    real=participant.query("screening=='primary' and feature_family=='multiband' and method=='real_only'")
    null_summary=[]
    for _,row in real.iterrows():
        values=null.query('participant==@row.participant and phase==@row.phase').balanced_accuracy.to_numpy()
        null_summary.append({'participant':row.participant,'phase':row.phase,'observed_balanced_accuracy':row.balanced_accuracy,
                             'null_mean':float(values.mean()),'null_q025':float(np.quantile(values,.025)),
                             'null_q975':float(np.quantile(values,.975)),
                             'monte_carlo_exceedance':float((1+np.sum(values>=row.balanced_accuracy))/(len(values)+1)),
                             'permutations':len(values)})
    null_summary=pd.DataFrame(null_summary)
    null_summary.to_csv(root/'permutation_summary.csv',index=False)
    primary=comparisons.query("phase=='thinking' and screening=='primary' and feature_family=='multiband' and comparator=='same_label_feature_mixup'").iloc[0]
    gate=bool(primary.mean_difference >= .02 and primary.ci_low > 0)
    duplicate=scores.query("method=='duplicate_real'").merge(scores.query("method=='shuffle_blocks'"),on=['participant','phase','screening','feature_family','seed'],suffixes=('_duplicate','_shuffle'))
    shuffle_max=float(np.max(np.abs(duplicate.balanced_accuracy_duplicate-duplicate.balanced_accuracy_shuffle)))
    result={'status':'complete', 'design':'Within-person later-trial-block evaluation; no verified session separation',
            'primary_comparison':primary.to_dict(), 'provisional_two_percentage_point_gate_passed':gate,
            'chance_balanced_accuracy':1/16, 'seed_replicates_are_not_independent_participants':True,
            'shuffle_vs_duplicate_max_balanced_accuracy_difference':shuffle_max,
            'total_imagined_original_trials':int(sum(x['original_trials'] for x in audits if x['phase']=='thinking')),
            'total_imagined_kept_trials':int(sum(x['kept_trials'] for x in audits if x['phase']=='thinking')),
            'total_imagined_test_trials':int(sum(len(x['test_trial_ids']) for x in audits if x['phase']=='thinking')),
            'limitations':['Phoneme augmentation, not sentence reconstruction', 'No verified session separation or new-person transfer',
                           'No independent eye/mouth artifact measurements in this analysis', 'Rest follows spoken trials and may retain label carryover',
                           'No metabolic calibration', 'One fixed feature/classifier design; negative results do not rule out every algorithm']}
    (root/'decision.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    figure_dir=root/'figures'
    figure_dir.mkdir(exist_ok=True)
    selected=summary.query("phase=='thinking' and screening=='primary' and feature_family=='multiband'").set_index('method').loc[METHODS]
    fig,ax=plt.subplots(figsize=(9,5))
    locations=np.arange(len(METHODS))
    for i,(method,row) in enumerate(selected.iterrows()):
        values=participant.query("phase=='thinking' and screening=='primary' and feature_family=='multiband' and method==@method").balanced_accuracy.to_numpy()*100
        ax.scatter(np.full(len(values),i)+np.linspace(-.13,.13,len(values)),values,color='#59666d',s=14,alpha=.45)
        ax.errorbar(i,row['mean']*100,yerr=[[100*(row['mean']-row.ci_low)],[100*(row.ci_high-row['mean'])]],fmt='o',color='#164b72',capsize=4,markersize=7)
    ax.axhline(6.25,color='#999999',linestyle='--',label='Nominal chance (16 classes)')
    ax.set_xticks(locations,[NAMES[m] for m in METHODS],rotation=24,ha='right')
    ax.set_ylabel('Balanced accuracy on real held-out trials (%)')
    ax.set_title('FEIS: participant results and mean with 95% bootstrap interval')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,loc='upper left')
    fig.tight_layout(); fig.savefig(figure_dir/'thinking_accuracy.png',dpi=180); plt.close(fig)
    pivot=participant.query("phase=='thinking' and screening=='primary' and feature_family=='multiband'").pivot(index='participant',columns='method',values='balanced_accuracy')
    delta=(pivot.same_label_splice-pivot.same_label_feature_mixup)*100
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(delta.index,delta.to_numpy(),color='#164b72',s=35)
    ax.axhline(0,color='#888888',linewidth=1)
    ax.axhline(primary.mean_difference*100,color='#164b72',linestyle='--',label='Mean difference')
    ax.axhspan(primary.ci_low*100,primary.ci_high*100,color='#164b72',alpha=.12,label='95% interval for mean')
    ax.set_xlabel('Public participant identifier'); ax.set_ylabel('Splice minus feature mixup (percentage points)')
    ax.set_title('Primary comparison: one effect per participant')
    ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figure_dir/'primary_paired_effect.png',dpi=180); plt.close(fig)
    for phase in ['thinking','resting']:
        line=summary.query("phase==@phase and screening=='primary' and method=='real_only'")
        print(phase, line[['feature_family','mean','ci_low','ci_high']].to_string(index=False))
    print('PRIMARY COMPARISON',primary.to_dict(),'GATE',gate,flush=True)
    print('Shuffle/duplicate difference',shuffle_max,flush=True)

    report=[]
    report.append('# FEIS validation: first external experiment\n')
    report.append('This is an exploratory external-data experiment following the prospective local [protocol](C:/Personal_Coding/thinktank-clearmind/research/feis_validation/PROTOCOL.md). It evaluates an adapted phoneme augmentation method, using real recorded trials as the test set. It does not validate sentence reconstruction or metabolism.\n')
    verdict='The provisional improvement criterion was met.' if gate else 'The provisional improvement criterion was not met.'
    report.append(f'**{verdict}** Same-phoneme splicing changed balanced accuracy by **{primary.mean_difference*100:+.2f} percentage points** relative to feature mixup; the participant-bootstrap 95% interval is **[{primary.ci_low*100:+.2f}, {primary.ci_high*100:+.2f}]**. {int(primary.participants_improved)} participants improved and {int(primary.participants_worse)} worsened.\n')
    report.append('## Recorded data and checks\n')
    report.append(f"The [FEIS v1.1 release](https://zenodo.org/records/3554128) was acquired on D:. Our file audit found {result['total_imagined_original_trials']:,} original imagined trials, of which {result['total_imagined_kept_trials']:,} passed basic format/finite-value screening. Across {int(primary.participants)} participants, {result['total_imagined_test_trials']:,} real later trials were reserved for testing. Labels and timing were checked from native CSV columns. Nominal chance is 6.25% for 16 classes.\n")
    report.append('The merged CSV files do not establish original session identities. This is a within-person later-trial-block test, not cross-session or new-person validation. Every augmentation donor was confined to the real training set, with parent receipts retained on D:. Features/model scaling used training examples only. Three augmentation seeds were averaged within a person.\n')
    reasons=Counter(reason for a in audits for trial in a['trials'] for reason in trial['excluded_reasons'])
    report.append(f'Basic exclusions across both task phases: {dict(reasons)}. The stricter artifact sensitivity uses training-only amplitude thresholds and available flags. It is not a validated eye/mouth cleanup. Full trial reasons and strict-screening counts are in `data_audit.json`.\n')
    sensitivity_path=root.parent/'timestamp_rounding_sensitivity'/'run_manifest.json'
    if sensitivity_path.exists():
        sensitivity=json.loads(sensitivity_path.read_text(encoding='utf-8'))
        if sensitivity.get('status')=='complete':
            report.append('## Timestamp precision sensitivity\n')
            report.append('The 19 excluded trials of participant 04 in each phase have six-decimal timestamp rounding, with a maximum step deviation of 0.75 microseconds and grid deviation of 0.5 microseconds. The initial tolerance of 0.1 microseconds was overly conservative; these are not established dropped-sample events. A separate audit-informed sensitivity used a tolerance of one microsecond, retained all 160 trials and kept exactly the same 48 real test trials. Original recordings and the primary results remain unchanged.\n')
            report.append(f"With this change to the admission tolerance, the cohort splice-minus-feature-mixup difference is {sensitivity['cohort_primary_difference']*100:+.2f} percentage points, with a 95% interval [{sensitivity['ci_low']*100:+.2f}, {sensitivity['ci_high']*100:+.2f}]. The provisional criterion remains unmet. See [sensitivity results](C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/results/timestamp_rounding_sensitivity/run_manifest.json). This sensitivity is supplemental, not a replacement prespecified outcome.\n")
    report.append('## Primary imagined-phoneme comparison\n')
    report.append('| Training condition | Mean balanced accuracy | Participant-bootstrap 95% interval |\n| --- | ---: | ---: |')
    for method,row in selected.iterrows():
        report.append(f"| {NAMES[method]} | {row['mean']*100:.2f}% | {row.ci_low*100:.2f}–{row.ci_high*100:.2f}% |")
    report.append('\n![Imagined-phoneme accuracy](C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/results/initial_v1/figures/thinking_accuracy.png)\n')
    report.append('![Participant differences](C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/results/initial_v1/figures/primary_paired_effect.png)\n')
    report.append('## Controls and limits\n')
    for phase in ['thinking','resting']:
        row=summary.query("phase==@phase and screening=='primary' and feature_family=='multiband' and method=='real_only'").iloc[0]
        permutation=null_summary.query('phase==@phase')
        report.append(f"- Real-only {phase}: {row['mean']*100:.2f}% balanced accuracy; shuffled-training-label mean {permutation.null_mean.mean()*100:.2f}%. Per-person permutation results are retained without treating permutation draws as new participants.")
    report.append(f'- Maximum difference between block shuffling and exact-copy augmentation: {shuffle_max*100:.6f} percentage points. The chosen spectral averages cannot establish correct phoneme order.')
    report.append('- Rest follows spoken trials and can retain phoneme/task carryover. It is a diagnostic comparison, not proof of a label-independent baseline. Native units and eye/mouth contamination remain limitations.')
    report.append('- The primary comparison was specified before classification. Secondary bands, strict screening and other comparators are exploratory. Confidence intervals describe variability across the supplied participants; they do not remove acquisition or protocol confounds.')
    report.append('- Results use one fixed, simple feature/classifier design. A weak result here does not prove that no other imagined-speech algorithm can work. A positive result here would need independent replication and artifact review.\n')
    report.append('## What this means for the paper\n')
    if gate:
        report.append('The adapted method has an initial advantage over a strong simple comparator in this design. Review the stricter-screening and task controls before planning an independent Kara One word replication. Describe this as augmentation utility on real held-out phoneme trials; do not transfer it to sentence or physiological claims.\n')
    else:
        report.append('These results do not support adding a performance claim for phoneme splicing to the manuscript. Keep the negative finding and controls. Review the phoneme-information and artifact results before investing in a larger sentence model. A Kara One study would be a separately justified exploratory experiment, rather than confirmation of an improvement established here.\n')
    report.append('The round-two author-review manuscript was not changed by this experiment. Author verification, ethics/reuse checks for the original donor files and the public-deposit decision remain separate requirements.\n')
    report.append('## Reproducibility\n')
    report.append(f"Data root: `{manifest['data_root']}`. Full real-trial predictions and donor receipts are saved under its `analysis/initial_v1` directory. Small tables, data audit, the decision record and software/protocol hashes are alongside this report. Dataset citation and ODC-BY terms are retained with the downloaded source README.\n")
    report.append('Commands from the project root:\n\n```powershell\npython -m pytest research/feis_validation/test_validation.py -q\npython -m research.feis_validation.benchmark\npython -m research.feis_validation.summarize\n```\n')
    (root/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')


if __name__ == '__main__':
    main()
