"""Render the independent result and chronology/eye diagnostic context."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd,numpy as np
from .kara_benchmark import OUTPUT

def main():
    data=pd.read_csv(OUTPUT/'cohort_summary.csv')
    entries=[('covariance','primary','Spatial covariance: primary'),('eegnet_style','primary','Temporal-spatial CNN: primary'),
        ('covariance','global_chronology','Spatial covariance: later trials'),('eegnet_style','global_chronology','Temporal-spatial CNN: later trials'),
        ('eye_only','primary','Eye channels only (diagnostic)')]
    fig,ax=plt.subplots(figsize=(9.5,5.4))
    for i,(model,condition,label) in enumerate(entries):
        row=data[(data.model==model)&(data.screening==condition)&(data.seed==17)].iloc[0]
        value=float(row['mean'])*100;lo=float(row.ci_low)*100;hi=float(row.ci_high)*100
        color='#245875' if model=='covariance' else '#9b493d' if model=='eegnet_style' else '#666666'
        ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',color=color,capsize=4)
    ax.set_yticks(range(len(entries)),[e[2] for e in entries]);ax.invert_yaxis();ax.set_xlim(0,45)
    ax.axvline(25,ls='--',color='#555555',label='Four-word chance: 25%');ax.axvline(30,ls=':',color='#8c6f36',label='Provisional feasibility target: 30%')
    ax.set_xlabel('Balanced accuracy on held-out real imagined-word trials (%)');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Kara One: no model clears the predefined signal gate',y=.96,fontsize=12)
    fig.text(.67,.90,'13 people; participant-bootstrap 95% intervals',ha='center',fontsize=10)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.21),ncol=2,fontsize=8)
    fig.subplots_adjust(left=.36,right=.98,bottom=.25,top=.83)
    target=OUTPUT/'figures';target.mkdir(exist_ok=True);fig.savefig(target/'word_decoding_and_controls.png',dpi=180);plt.close(fig)
    print('Saved scientific figure')
if __name__=='__main__':main()
