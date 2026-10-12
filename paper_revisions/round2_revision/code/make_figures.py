from pathlib import Path
import json,csv,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from reanalyse import ROOT,DATA,load_bank,gather
ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=DATA);args=ap.parse_args();DATA=args.data

OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
r=json.loads((ROOT/'results/results.json').read_text())
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight');fig.savefig(OUT/(name+'.tiff'),bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(7.2,3.6));ax.axis('off')
boxes=[(.02,.7,'Frozen text\nCMUdict + declared overrides'),(.39,.7,'Stimulus crosswalk\nOne recorded block per phoneme'),(.76,.7,'Insert recorded waits\n1 or 2 one-second blocks'),(.02,.12,'Trial 1\nCalibration + common inverse'),(.39,.12,'Trial 2\nConstruct composites'),(.76,.12,'Trials 3–5\nHeld-out within-session reference')]
for x,y,t in boxes:ax.text(x,y,t,va='center',ha='left',fontsize=9,bbox=dict(boxstyle='round,pad=.5',fc='#f1f4f6',ec='#667580'),transform=ax.transAxes)
for x in [.33,.70]:ax.annotate('',xy=(x+.045,.7),xytext=(x,.7),arrowprops=dict(arrowstyle='->'),xycoords='axes fraction')
ax.text(.5,.42,'Windowed spectrum → normalized gamma index → illustrative physiological transform',ha='center',fontsize=9,transform=ax.transAxes)
fig.tight_layout();save(fig,'Fig1_workflow')

fig,axs=plt.subplots(1,2,figsize=(7.2,3.9),gridspec_kw={'width_ratios':[1,1.6]})
primary=[x for x in r['summary'] if x['config']=='primary' and x['variable']=='CMRO2']
for j,x in enumerate(primary):
    axs[0].errorbar(x['mean'],j,xerr=[[x['mean']-x['lower']],[x['upper']-x['mean']]],fmt='o',color='#186a8a',capsize=4)
axs[0].set(yticks=range(3),yticklabels=['S1','S2','S3'],xlabel='Change in normalized gamma index',title='A  Primary simulation interval')
axs[0].axvline(0,color='grey',lw=.8);axs[0].invert_yaxis()
cfg=['primary','own_scale','word_final','auxiliary_speech','old_file_order','low_gamma']
labels=['Recorded waits','Own scaling','Last phoneme only','Auxiliary speech','Old file order','30–45 Hz']
for k,sn in enumerate(['S1','S2','S3']):
    means=[next(x['mean'] for x in r['summary'] if x['config']==c and x['sentence']==sn and x['variable']=='CMRO2') for c in cfg]
    axs[1].plot(means,np.arange(6)+(k-1)*.16,'o',ms=5,label=sn)
axs[1].set(yticks=range(6),yticklabels=labels,xlabel='Mean change in normalized gamma index',title='B  Dependence on analysis choices');axs[1].axvline(0,color='grey',lw=.8);axs[1].invert_yaxis();axs[1].legend(fontsize=8)
fig.tight_layout();save(fig,'Fig2_simulation')

sp=np.load(ROOT/'results/spectra.npz');sm=np.load(ROOT/'results/source_maps.npz')
fig=plt.figure(figsize=(7.2,3.0),layout='constrained')
gs=fig.add_gridspec(1,4,width_ratios=[1.3,1,1,.07]);axs=[fig.add_subplot(gs[0,i]) for i in range(3)];cbax=fig.add_subplot(gs[0,3])
for key,label in [('reference','Held-out reference'),('S1_no_gap','S1 no insertion'),('S1_wait_inserted','S1 with waits')]:axs[0].plot(sp['frequency'],10*np.log10(sp[key]),label=label,lw=1)
axs[0].set(xlabel='Frequency (Hz)',ylabel='PSD (dB; stored units²/Hz)',title='A  Mean spectra');axs[0].legend(fontsize=6)
for ax,key,title in zip(axs[1:],['reference','S1_wait_inserted'],['B  Held-out\nreference','C  S1 with\nwaits']):
    z=sm[key]/sm[key].mean();sc=ax.scatter(sm['xyz'][:,0],sm['xyz'][:,1],c=z,s=10,vmin=0,vmax=3,cmap='viridis');ax.set(title=title,aspect='equal',xlabel='Sphere x',ylabel='Sphere y')
fig.colorbar(sc,cax=cbax,label='Relative modeled source power');save(fig,'Fig3_consistency')

task,wait,*_=load_bank(DATA);pools=np.stack([task[:,1],wait[:,1]])
plans=json.loads((ROOT/'results/assembly_manifests_seed0.json').read_text())['primary_S1']['inserted'];synt=gather(pools,plans).reshape(-1,16)
raw=np.loadtxt(DATA/'DLR_0_1.txt')[:,1:17]
fig,axs=plt.subplots(2,1,figsize=(7.2,4.4),sharex=True,layout='constrained')
arr=[]
for x in [raw[:5000],synt[:5000]]:
    f,t,p=spectrogram(x,fs=250,window='hann',nperseg=125,noverlap=62,axis=0,scaling='density',detrend='constant')
    arr.append((f,t,10*np.log10(np.maximum(p.mean(axis=1),1e-20))))
vmin=min(np.percentile(a[2],5) for a in arr);vmax=max(np.percentile(a[2],95) for a in arr)
for ax,(f,t,p),title in zip(axs,arr,['A  Original continuous source fragment (file 0; mixed task and wait)','B  S1 composite with inserted waits (seed 0)']):
    mesh=ax.pcolormesh(t,f,p,shading='auto',vmin=vmin,vmax=vmax,cmap='viridis');ax.set(ylim=(2,100),ylabel='Hz');ax.set_title(title,fontsize=9)
axs[1].set_xlabel('Time within displayed fragment (s)');fig.colorbar(mesh,ax=axs,label='PSD (dB; stored units²/Hz)',shrink=.85);save(fig,'Fig4_time_frequency')
print('Created four figure sets')
