"""Bounded covariance and EEGNet-style real-trial baselines."""
from __future__ import annotations
import copy
import random
import numpy as np
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

CS = (0.01, 0.1, 1.0)
EPOCHS = (30, 60)
SEEDS = (17, 29, 43)
BANDS = ((1,4),(4,8),(8,13),(13,30),(30,45))

def signal_transform(x, fs=256., line=50.):
    x=np.asarray(x,dtype=np.float64)
    x=x-x.mean(axis=-1,keepdims=True)
    x=x-x.mean(axis=1,keepdims=True)
    b,a=signal.iirnotch(line,30,fs=fs)
    x=signal.filtfilt(b,a,x,axis=-1)
    sos=signal.butter(4,[1,45],fs=fs,btype='bandpass',output='sos')
    x=signal.sosfiltfilt(sos,x,axis=-1)
    margin=int(round(fs*0.5)); x=x[...,margin:-margin]
    from fractions import Fraction
    ratio=Fraction(128/int(fs)).limit_denominator(10000)
    x=signal.resample_poly(x,ratio.numerator,ratio.denominator,axis=-1)
    if x.shape[-1] !=512: raise ValueError('Expected central four-second epoch')
    if not np.isfinite(x).all(): raise ValueError('Nonfinite processed signal')
    return x.astype(np.float32)

def covariance_features(x,fs=128.):
    x=np.asarray(x,dtype=np.float64)
    n,c,_=x.shape; upper=np.triu_indices(c)
    weights=np.where(upper[0]==upper[1],1.,np.sqrt(2.))
    features=[]
    for lo,hi in BANDS:
        z=signal.sosfiltfilt(signal.butter(4,[lo,hi],fs=fs,btype='bandpass',output='sos'),x,axis=-1)
        z=z-z.mean(axis=-1,keepdims=True)
        cov=np.einsum('nct,ndt->ncd',z,z)/(z.shape[-1]-1)
        scale=np.maximum(np.trace(cov,axis1=1,axis2=2)/c,1e-12)
        cov=.9*cov+.1*scale[:,None,None]*np.eye(c)
        val,vec=np.linalg.eigh(cov)
        logged=np.einsum('nci,ni,ndi->ncd',vec,np.log(np.maximum(val,1e-15)),vec)
        features.append(logged[:,upper[0],upper[1]]*weights)
    return np.concatenate(features,axis=1)

def covariance_classifier(c):
    return make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000,solver='lbfgs'))

def inner_block_folds(train,blocks):
    ordered=np.unique(blocks[train])
    if len(ordered)<4: raise ValueError('At least four training blocks required')
    folds=[]
    for validation in ordered[-2:]:
        fit=train[blocks[train]<validation]; val=train[blocks[train]==validation]
        if len(set(fit)&set(val)): raise AssertionError('Inner leakage')
        folds.append((fit,val))
    return folds

def select_covariance(features,y,folds):
    from sklearn.metrics import balanced_accuracy_score
    scores=[]
    for c in CS:
        fold_scores=[]
        for fit,val in folds:
            if set(y[val])-set(y[fit]): raise ValueError('Missing class in inner fit')
            prediction=covariance_classifier(c).fit(features[fit],y[fit]).predict(features[val])
            fold_scores.append(float(balanced_accuracy_score(y[val],prediction)))
        scores.append({'C':c,'fold_scores':fold_scores,'mean':float(np.mean(fold_scores))})
    choice=CS[int(np.argmax([entry['mean'] for entry in scores]))]
    return choice,scores

class EEGNetStyle(nn.Module):
    def __init__(self,channels,classes,samples=512):
        super().__init__()
        self.temporal=nn.Sequential(nn.Conv2d(1,8,(1,64),padding='same',bias=False),nn.BatchNorm2d(8))
        self.spatial_conv=nn.Conv2d(8,16,(channels,1),groups=8,bias=False)
        self.spatial=nn.Sequential(self.spatial_conv,nn.BatchNorm2d(16),nn.ELU(),nn.AvgPool2d((1,4)),nn.Dropout(.5))
        self.separable=nn.Sequential(nn.Conv2d(16,16,(1,16),groups=16,padding='same',bias=False),
                                   nn.Conv2d(16,16,(1,1),bias=False),nn.BatchNorm2d(16),nn.ELU(),
                                   nn.AvgPool2d((1,8)),nn.Dropout(.5))
        self.classifier=nn.Linear(16*(samples//32),classes)
    def forward(self,x):
        return self.classifier(self.separable(self.spatial(self.temporal(x))).flatten(1))
    def constrain(self):
        with torch.no_grad():
            for weight,maximum in [(self.spatial_conv.weight,1.),(self.classifier.weight,.25)]:
                norms=torch.linalg.vector_norm(weight.flatten(1),dim=1)
                weight.mul_((maximum/norms.clamp_min(maximum)).reshape((-1,)+(1,)*(weight.ndim-1)))

def neural_predictions(train_x,train_y,query_x,epochs,seed=17,device=None):
    """No query labels or query-fitted scale. Return snapshots at selected epochs."""
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    scale=np.sqrt(np.mean(np.asarray(train_x,dtype=np.float64)**2,axis=(0,2),keepdims=True)).clip(1e-8)
    x=torch.as_tensor((train_x/scale).astype(np.float32)[:,None],device=device)
    query=torch.as_tensor((query_x/scale).astype(np.float32)[:,None],device=device)
    classes=np.unique(train_y)
    if not np.array_equal(classes,np.arange(len(classes))): raise ValueError('CNN labels must be contiguous and fully present')
    y=torch.as_tensor(train_y,dtype=torch.long,device=device)
    model=EEGNetStyle(x.shape[2],len(classes),x.shape[-1]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    predictions={}
    for epoch in range(1,max(epochs)+1):
        model.train()
        order=torch.randperm(len(x),generator=generator).to(device)
        for indices in order.split(32):
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(x[indices]),y[indices])
            if not torch.isfinite(loss): raise ValueError('Nonfinite CNN loss')
            loss.backward();optimizer.step();model.constrain()
        if epoch in epochs:
            model.eval()
            with torch.no_grad():
                predictions[epoch]=torch.cat([model(batch) for batch in query.split(64)]).argmax(1).cpu().numpy()
    return predictions

def select_neural(x,y,folds):
    from sklearn.metrics import balanced_accuracy_score
    scores={epoch:[] for epoch in EPOCHS}
    for fit,val in folds:
        prediction=neural_predictions(x[fit],y[fit],x[val],EPOCHS,17)
        for epoch in EPOCHS: scores[epoch].append(float(balanced_accuracy_score(y[val],prediction[epoch])))
    entries=[{'epochs':epoch,'fold_scores':scores[epoch],'mean':float(np.mean(scores[epoch]))} for epoch in EPOCHS]
    return EPOCHS[int(np.argmax([e['mean'] for e in entries]))],entries

def permute_within(y,indices,blocks,rng):
    out=y.copy()
    for block in np.unique(blocks[indices]):
        chosen=indices[blocks[indices]==block]
        out[chosen]=rng.permutation(out[chosen])
    return out
