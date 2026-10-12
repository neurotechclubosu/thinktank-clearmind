"""Round-two replacement analysis. Run from any directory; no API calls.

This is an exploratory, single-recording-source software demonstration.
Monte Carlo draws are NOT participants or a null distribution.
"""
from pathlib import Path
import argparse, csv, hashlib, json, platform, re, sys
import numpy as np
import scipy
from scipy.signal import welch
from scipy.stats import spearmanr
from model import neurovascular_from_cmro2, VARIABLES, CONSTANTS
from sphere import SourceImager

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / 'data/eeg_recordings'
FS = 250
# Crosswalk from the archived stimulus slides. Dialect/stress approximations
# are explicit: AA -> dark/are; AO -> roar/morning; AH0 -> again/under.
MAP = dict(AA=10, AE=8, AH=9, AO=7, AW=19, AY=18, B=28, CH=24,
           D=30, DH=31, EH=4, ER=6, EY=13, F=21, G=35, HH=36,
           IH=1, IY=0, JH=32, K=27, L=41, M=37, N=38, NG=39,
           OW=16, OY=15, P=20, R=40, S=25, SH=26, T=22, TH=23,
           UH=2, UW=3, V=29, W=42, Y=43, Z=33, ZH=34)
OLD_MAP = dict(zip('AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split(), range(39)))

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def csvwrite(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def words(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*", text.replace('’', "'"))

def indices(text, lex, unit='phoneme', old=False):
    out=[]
    for word in words(text):
        ph=lex[word.lower()]  # Missing words fail; never silently disappear.
        if unit=='word_final': ph=ph[-1:]
        ids=[(OLD_MAP[p.rstrip('012')] if old else (5 if p=='AH0' else MAP[p.rstrip('012')])) for p in ph]
        out.append(ids)
    return out

def load_bank(data_dir):
    task, wait, audit, events, manifest = [], [], [], [], []
    for k in range(44):
        path=data_dir/f'DLR_{k}_1.txt'; b=path.read_bytes(); a=np.loadtxt(path)
        if a.shape[1]!=32 or not np.isfinite(a).all(): raise ValueError(path)
        pairs=[]
        for marker in range(1,6):
            ix=np.flatnonzero(a[:,31]==marker)
            if len(ix)!=2: raise ValueError(f'{path}: marker {marker} not paired')
            s,e=map(int,ix); duration=float(a[e,30]-a[s,30])
            if not (1.8<duration<2.2 and 450<e-s<550): raise ValueError(f'{path}: invalid event')
            pairs.append((s,e)); events.append(dict(file=path.name,marker=marker,start_row=s,end_row_exclusive=e,samples=e-s,seconds=duration,rate=(e-s)/duration))
        task.append(np.stack([a[s:s+250,1:17] for s,e in pairs]))
        ws=[]
        for j,(s,e) in enumerate(pairs[:4]):
            # Exclude 200 ms after cue offset; stay entirely in instructed wait.
            if e+300>pairs[j+1][0]: raise ValueError('Wait block overlaps next cue')
            ws.append(a[e+50:e+300,1:17])
        wait.append(np.stack(ws))
        first=int(np.flatnonzero(a[:,0]==0)[0]); audit.append(dict(file=path.name,rows=len(a),first_counter_zero=first,first_task_start=pairs[0][0],old_block_entirely_before_first_task=first+512<=pairs[0][0],marker_pairs=len(pairs)))
        manifest.append(dict(file=path.name,sha256=hashlib.sha256(b).hexdigest(),lf_git_blob_sha1=hashlib.sha1(b'blob '+str(len(b.replace(b'\r\n',b'\n'))).encode()+b'\0'+b.replace(b'\r\n',b'\n')).hexdigest()))
    return np.asarray(task),np.asarray(wait),audit,events,manifest

def features(blocks):
    """Two non-overlapping periodic-Hann periodograms per one-second block."""
    x=np.asarray(blocks); w=x.reshape(-1,2,125,16)
    f,p=welch(w, fs=FS, window='hann',nperseg=125,noverlap=0,detrend='constant',axis=2,scaling='density')
    gamma=p[:,:, (f>=30)&(f<100),:].sum(axis=2)*(f[1]-f[0])
    return f,p,gamma

def normalized(g,bounds):
    lo,hi=bounds
    if hi<=lo: raise ValueError('Degenerate gamma calibration')
    return np.clip((g-lo)/(hi-lo),0,1)

def summary(x):
    x=np.asarray(x); sd=float(x.std(ddof=1)); q=np.quantile(x,[.025,.975])
    return dict(mean=float(x.mean()),sd=sd,lower=float(q[0]),upper=float(q[1]),mcse_mean=sd/np.sqrt(len(x)),fraction_positive=float(np.mean(x>0)))

def plan(word_ids, rng, gap_mode, unit):
    # Element is (pool, file index); every block has exactly 250 samples.
    out=[]; control=[]
    for j,ids in enumerate(word_ids):
        out.extend((0,k) for k in ids); control.extend((0,k) for k in ids)
        if j==len(word_ids)-1 or gap_mode=='none': continue
        count=int(rng.integers(1,3))
        for _ in range(count):
            out.append((1,int(rng.integers(0,44))) if gap_mode=='wait' else (0,int(rng.integers(39,44))))
            control.append((0,ids[-1]))  # same duration, repeated preceding task block
    return out,control

def gather(pool, plan):
    return np.stack([pool[p,k] for p,k in plan])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--draws',type=int,default=2000);ap.add_argument('--data',type=Path,default=DATA);a=ap.parse_args()
    if a.draws<2: raise ValueError('Need at least two draws')
    out=ROOT/'results';out.mkdir(exist_ok=True)
    trials=json.loads((ROOT/'inputs/sentences.json').read_text(encoding='utf-8'))
    lexdata=json.loads((ROOT/'inputs/pronunciations.json').read_text()); lex=lexdata['lexicon']
    task,wait,audit,events,manifest=load_bank(a.data)
    csvwrite(out/'data_audit.csv',audit);csvwrite(out/'events.csv',events);dump(out/'input_hashes.json',manifest)
    # Trial 1 calibration, trial 2 construction, trials 3-5 reference.
    cal=np.concatenate([task[:,0],wait[:,0]])
    f,pc,gc=features(cal);bounds=np.quantile(gc,[.01,.99])
    pools=np.stack([task[:,1],wait[:,1]])
    _,_,gp=features(pools.reshape(-1,250,16));gp=gp.reshape(2,44,2,16)
    np.savez_compressed(out/'feature_arrays.npz',calibration_gamma=gc,construction_gamma=gp,task=task,wait=wait)
    draws=[]; rows=[]; checks=[]; manifests={}; consistency=[]
    default_gp=gp.copy(); default_bounds=bounds.copy()
    configs=[('primary','phoneme','wait','fixed',False),('own_scale','phoneme','wait','own',False),('word_final','word_final','wait','fixed',False),('auxiliary_speech','phoneme','aux','fixed',False),('old_file_order','phoneme','wait','fixed',True),('low_gamma','phoneme','wait','fixed',False)]
    for name,unit,mode,scale,old in configs:
        gp=default_gp;bounds=default_bounds
        if name=='low_gamma':
            _,pp,_=features(pools.reshape(-1,250,16)); mask=(f>=30)&(f<45)
            gp=(pp[:,:,mask,:].sum(axis=2)*2).reshape(2,44,2,16)
            bounds=np.quantile(pc[:,:,mask,:].sum(axis=2)*2,[.01,.99])
        for sentence,t in trials.items():
            ids=indices(t['response'],lex,unit,old)
            base,_=plan(ids,np.random.default_rng(0),'none',unit)
            bg=gather(gp,base).reshape(-1,16)
            bnd=bounds if scale=='fixed' else (bg.min(),bg.max())
            bn=normalized(bg,bnd);bv=neurovascular_from_cmro2(2.1+bn)
            bmeans=np.array([bv[v].mean() for v in VARIABLES])
            vals=[];ctrl=[];lengths=[]
            count=a.draws if name=='primary' else min(a.draws,500)
            for seed in range(count):
                rng=np.random.default_rng(seed); pp,cc=plan(ids,rng,mode,unit)
                g=gather(gp,pp).reshape(-1,16);cg=gather(gp,cc).reshape(-1,16)
                bd=bounds if scale=='fixed' else (g.min(),g.max()); cbd=bounds if scale=='fixed' else (cg.min(),cg.max())
                n=normalized(g,bd); cn=normalized(cg,cbd);nv=neurovascular_from_cmro2(2.1+n);cv=neurovascular_from_cmro2(2.1+cn)
                vv=np.array([nv[v].mean() for v in VARIABLES]); cvv=np.array([cv[v].mean() for v in VARIABLES]); vals.append(vv-bmeans);ctrl.append(vv-cvv);lengths.append(len(pp))
                draw=dict(config=name,sentence=sentence,seed=seed,blocks=len(pp),base_blocks=len(base),delta_index=float(n.mean()-bn.mean()),wait_minus_duration_control=float(n.mean()-cn.mean()))
                draw.update({'delta_'+v:float(vv[j]-bmeans[j]) for j,v in enumerate(VARIABLES)});draws.append(draw)
                if seed==0:
                    manifests[name+'_'+sentence]={'word_ids':ids,'no_gap':base,'inserted':pp,'duration_control':cc}
                    # Exact control: reverse already selected windows. Same multiset.
                    rev=neurovascular_from_cmro2(2.1+normalized(g[::-1],bd))
                    err=max(abs(float(rev[v].mean()-nv[v].mean())) for v in VARIABLES)
                    checks.append(dict(check='window_order_invariance',config=name,sentence=sentence,max_abs_error=err,passed=bool(err<1e-10)))
                    x=gather(pools,pp).reshape(-1,16)
                    _,dp,direct=features(x.reshape(-1,250,16))
                    if name=='low_gamma': direct=dp[:,:,(f>=30)&(f<45),:].sum(axis=2)*2
                    checks.append(dict(check='cached_equals_materialized',config=name,sentence=sentence,max_abs_error=float(np.max(np.abs(direct.reshape(-1,16)-g))),passed=bool(np.allclose(direct.reshape(-1,16),g))))
            vals=np.asarray(vals);ctrl=np.asarray(ctrl)
            for j,v in enumerate(VARIABLES):
                row=dict(config=name,sentence=sentence,variable=v,draws=count,base=float(bmeans[j]),inserted_mean=float(bmeans[j]+vals[:,j].mean()),base_blocks=len(base),mean_inserted_blocks=float(np.mean(lengths)))
                row.update(summary(vals[:,j]));row['duration_control_delta']=float(ctrl[:,j].mean());rows.append(row)
            print(name,sentence,'done',flush=True)
    gp=default_gp;bounds=default_bounds
    csvwrite(out/'simulation_draws.csv',draws);csvwrite(out/'simulation_summary.csv',rows);dump(out/'assembly_manifests_seed0.json',manifests)
    quality=[]
    _,pq,gq=features(task.reshape(-1,250,16)); line=pq[:,:,(f>=58)&(f<=62),:].sum(axis=2)*2
    for ch in range(16):
        quality.append(dict(channel_index=ch+1,median_task_peak_to_peak=float(np.median(np.ptp(task[:,:,:,ch],axis=2))),median_task_sd=float(np.median(task[:,:,:,ch].std(axis=2))),median_gamma_fraction_58_62hz=float(np.median(line[:,:,ch]/gq[:,:,ch])),calibration_clipped_fraction=float(np.mean((gq[:,:,ch]<bounds[0])|(gq[:,:,ch]>bounds[1])))))
    csvwrite(out/'signal_quality.csv',quality)
    # Same covariance, centering and inverse for ALL source comparisons.
    im=SourceImager(n_sources=300)
    calc=cal-cal.mean(axis=1,keepdims=True); cov=np.cov(calc.reshape(-1,16),rowvar=False)
    W=im.operator(cov)
    def feat(x):
        x=x.reshape(-1,250,16); ff,pp,gg=features(x)
        psd=pp.mean(axis=(0,1,3)); keep=(ff>=2)&(ff<=100)
        xc=x-x.mean(axis=1,keepdims=True); c=np.einsum('bti,btj->ij',xc,xc)/(len(x)*250)
        pr=im.P@c@im.P; sm=np.einsum('ij,jk,ik->i',W,pr,W)
        return dict(spectrum=psd[keep],source=sm,index=normalized(gg,bounds).mean(axis=(0,1)))
    refs=[feat(task[k,j][None]) for k in range(44) for j in range(2,5)]
    reference={key:np.mean([x[key] for x in refs],axis=0) for key in refs[0]}
    def compare(fe,ref):
        aa=10*np.log10(fe['spectrum']);bb=10*np.log10(ref['spectrum'])
        return dict(logpsd_r=float(np.corrcoef(aa,bb)[0,1]),logpsd_rmse_db=float(np.sqrt(np.mean((aa-bb)**2))),source_rho=float(spearmanr(fe['source'],ref['source']).statistic),index_profile_rho=float(spearmanr(fe['index'],ref['index']).statistic),index_profile_mae=float(np.mean(np.abs(fe['index']-ref['index']))))
    for k in range(44):
        own=refs[k*3:(k+1)*3];others=refs[:k*3]+refs[(k+1)*3:]
        fe={key:np.mean([x[key] for x in own],axis=0) for key in own[0]}; ref={key:np.mean([x[key] for x in others],axis=0) for key in others[0]}
        consistency.append(dict(kind='reference_leave_file_out',sentence=f'file{k}',condition='task',**compare(fe,ref)))
    spectra={'frequency':f[(f>=2)&(f<=100)],'reference':reference['spectrum']}; source_arrays={'xyz':im.src,'reference':reference['source']}
    for sn in trials:
        m=manifests['primary_'+sn]
        for cond,key in [('no_gap','no_gap'),('wait_inserted','inserted')]:
            x=gather(pools,m[key]); fe=feat(x)
            consistency.append(dict(kind='composite',sentence=sn,condition=cond,**compare(fe,reference)))
            spectra[sn+'_'+cond]=fe['spectrum']; source_arrays[sn+'_'+cond]=fe['source']
    csvwrite(out/'consistency.csv',consistency);np.savez_compressed(out/'spectra.npz',**spectra);np.savez_compressed(out/'source_maps.npz',**source_arrays)
    # State all implemented parameters and tests, without claiming biological validity.
    checks.append(dict(check='corrected_map_B',config='all',sentence='all',max_abs_error=0,passed=MAP['B']==28))
    checks.append(dict(check='source_operator_rank',config='all',sentence='all',max_abs_error=0,passed=bool(np.linalg.matrix_rank(W@im.P)<=15)))
    if not all(c['passed'] for c in checks): raise AssertionError('Analysis checks failed')
    dump(out/'checks.json',checks)
    results=dict(version='round2-v1.27-working',status='author verification required',draws=a.draws,fs=FS,window=125,overlap=0,gamma_band_hz=[30,100],bounds=bounds.tolist(),normalization='calibration 1st/99th percentiles, clipped',source_count=len(im.src),source_covariance='trial1 task+wait, common empirical covariance',source_model='homogeneous sphere, radial dipoles; not fsaverage or validated anatomy',constants=CONSTANTS,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,audit=dict(files=44,paired_events=len(events),rate_median=float(np.median([x['rate'] for x in events])),rate_range=[min(x['rate'] for x in events),max(x['rate'] for x in events)],old_blocks_before_task=sum(x['old_block_entirely_before_first_task'] for x in audit)),summary=rows,consistency=consistency)
    dump(out/'results.json',results)
    print('PASS',len(checks),'checks; all outputs finite')

if __name__=='__main__': main()
