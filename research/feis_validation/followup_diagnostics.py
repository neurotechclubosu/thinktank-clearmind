"""Supplementary scheduling audit and rendered research figure, explicitly exploratory."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .download import DEFAULT_ROOT
from .followup import OUTPUT
from .summarize_followup import bootstrap

def main():
    rows=[];sequences={};common={}
    for source in sorted((DEFAULT_ROOT/'extracted/experiments').glob('*/thinking/thinking.csv')):
        d=pd.read_csv(source,usecols=['Epoch','Label']).drop_duplicates();ids=d.Epoch.to_numpy();y=d.Label.to_numpy()
        cut=int(np.floor(len(y)/16*.7))*16;participant=source.parts[-3];row={'participant':participant,'trials':len(y)}
        sequences[participant]={int(i):str(label) for i,label in zip(ids,y)}
        for modulo in [16,32]:
            lookup={k:pd.Series(y[:cut][ids[:cut]%modulo==k]).mode().iloc[0] for k in np.unique(ids[:cut]%modulo)}
            row[f'order_only_mod{modulo}_accuracy']=float(np.mean([lookup[int(i)%modulo]==label for i,label in zip(ids[cut:],y[cut:])]))
        rows.append(row)
    reference=sequences['01']
    for participant,sequence in sequences.items():common[participant]=all(reference.get(i)==label for i,label in sequence.items())
    receipt={'status':'supplementary audit after baseline run; not a selected primary classifier',
        'same_trial_id_to_label_mapping_as_participant01':common,'order_only_controls':rows,
        'caution':'Shared scheduling across participants is not independent biological evidence. Modulo lookups use training labels only; chosen diagnostic periods16/32 are exploratory. Conditional permutation assumes within-block exchangeability.'}
    (OUTPUT/'task_order_audit.json').write_text(json.dumps(receipt,indent=2))
    scores=pd.read_csv(OUTPUT/'scores.csv');extra=[]
    for (phase,model,screening,seed),group in scores.groupby(['phase','model','screening','seed']):
        if len(group)>1:
            lo,hi=bootstrap(group.balanced_accuracy)
            extra.append({'phase':phase,'model':model,'screening':screening,'seed':int(seed),'people':len(group),
                          'mean':float(group.balanced_accuracy.mean()),'ci_low':lo,'ci_high':hi})
    pd.DataFrame(extra).to_csv(OUTPUT/'sensitivity_summary.csv',index=False)
    summary=pd.read_csv(OUTPUT/'cohort_summary.csv');thinking=summary[summary.phase=='thinking'].set_index('model')
    order=['covariance','eegnet_style','mean_power_lda','nuisance'];labels=['Spatial covariance','Temporal-spatial CNN','Mean band power','Amplitude/noise proxies']
    fig,ax=plt.subplots(figsize=(9,5));means=100*thinking.loc[order,'mean_balanced_accuracy'].to_numpy()
    lower=100*thinking.loc[order,'ci_low'].to_numpy();upper=100*thinking.loc[order,'ci_high'].to_numpy()
    ax.errorbar(means,np.arange(4),xerr=np.vstack([means-lower,upper-means]),fmt='o',capsize=4,color='#245875')
    ax.axvline(6.25,color='#555555',ls='--',label='16-class chance: 6.25%');ax.axvline(8.25,color='#9b493d',ls=':',label='Provisional feasibility target: 8.25%')
    ax.set_yticks(np.arange(4),labels);ax.invert_yaxis();ax.set_xlabel('Balanced accuracy on later real imagined trials (%)')
    ax.set_xlim(0,12);fig.suptitle('FEIS: no stronger baseline clears the signal gate',fontsize=12,y=.96)
    fig.text(.64,.90,'21 people; bars are participant-bootstrap 95% intervals',ha='center',fontsize=10)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2,fontsize=8);ax.grid(axis='x',alpha=.2)
    fig.subplots_adjust(left=.29,right=.98,bottom=.28,top=.83)
    figures=OUTPUT/'figures';figures.mkdir(exist_ok=True);fig.savefig(figures/'stronger_baselines.png',dpi=180);plt.close(fig)
    report=OUTPUT/'REPORT.md'
    base=report.read_text(encoding='utf-8').split('\n## Prespecified sensitivities and scheduling audit')[0]
    report.write_text(base,encoding='utf-8')
    with report.open('a',encoding='utf-8') as stream:
        stream.write('\n## Prespecified sensitivities and scheduling audit\n\n')
        for e in extra:
            if e['phase']=='thinking' and (e['screening']=='strict' or (e['model']=='eegnet_style' and e['screening']=='primary')):
                stream.write(f"- {e['model']}, {e['screening']}, seed{e['seed']}: {100*e['mean']:.2f}% [{100*e['ci_low']:.2f}, {100*e['ci_high']:.2f}]%; {e['people']} people.\n")
        stream.write('\nAll21 participants share the same trial-ID-to-prompt mapping wherever their recordings overlap. A supplementary training-only modulo16 scheduling lookup reaches10.42% in the20 full recordings and12.50% in the shorter participant12 recording, without EEG. Modulo32 yields4.17% and14.58%, respectively. These periods were examined after the baseline experiment and are not tuned primary models. The common schedule prevents interpreting repeated order-control scores across people as independent evidence; it further limits imagery-specific inference from a small above-chance score.\n')
        stream.write('\nThe rest-phase CNN mean7.74% [6.55,8.83]% is exploratory and rest follows spoken speech. Thinking does not outperform it. One-person listening/spoken models remain near chance. None of these observations demonstrates that all imagined-speech EEG lacks information.\n')
    print('Saved task-order audit, sensitivities and research figure')

if __name__=='__main__':main()
