"""Signal and leakage controls for the stronger baselines, without real scores."""
import numpy as np
import torch
from .followup_models import (covariance_features,covariance_classifier,inner_block_folds,
    neural_predictions,signal_transform,permute_within)

def test_covariance_detects_spatial_relationship_without_power_difference():
    rng=np.random.default_rng(70);n=160;c=4;s=512;y=np.tile([0,1],n//2)
    x=rng.normal(size=(n,c,s)); shared=rng.normal(size=(n,s))
    x[:,0]=shared+rng.normal(scale=.15,size=(n,s))
    x[:,1]=np.where(y[:,None]==0,1,-1)*shared+rng.normal(scale=.15,size=(n,s))
    features=covariance_features(x)
    fit=np.arange(112);test=np.arange(112,160)
    prediction=covariance_classifier(.1).fit(features[fit],y[fit]).predict(features[test])
    assert np.mean(prediction==y[test])>.95
    null=covariance_classifier(.1).fit(features[fit],rng.permutation(y[fit])).predict(features[test])
    assert .25<=np.mean(null==y[test])<=.75

def test_nested_folds_do_not_touch_test_and_are_forward():
    blocks=np.repeat(np.arange(10),16);train=np.arange(112);test=np.arange(112,160)
    for fit,val in inner_block_folds(train,blocks):
        assert set(fit).isdisjoint(val) and set(fit).isdisjoint(test) and set(val).isdisjoint(test)
        assert max(blocks[fit])<min(blocks[val])
    y=np.tile(np.arange(16),10);shuffled=permute_within(y,train,blocks,np.random.default_rng(6))
    assert np.array_equal(shuffled[test],y[test])
    for block in range(7): assert sorted(shuffled[blocks==block])==list(range(16))

def test_fixed_transform_keeps_trial_boundaries_and_common_reference():
    x=np.random.default_rng(7).normal(size=(5,6,1280))
    result=signal_transform(x)
    assert result.shape==(5,6,512)
    np.testing.assert_allclose(result[0],signal_transform(x[:1])[0],atol=1e-6)
    np.testing.assert_allclose(result.mean(axis=1),0,atol=1e-6)

def test_neural_temporal_positive_control_and_query_independence():
    rng=np.random.default_rng(12);y=np.tile([0,1],64);t=np.arange(512)/128
    x=rng.normal(scale=.2,size=(128,4,512)).astype(np.float32)
    for i,label in enumerate(y):
        x[i,0]+=np.sin(2*np.pi*10*t)*(t<(2 if label==0 else 0))
        if label==1: x[i,0]+=np.sin(2*np.pi*10*t)*(t>=2)
    fit=np.arange(96);test=np.arange(96,128)
    pred=neural_predictions(x[fit],y[fit],x[test],[30],17)[30]
    assert np.mean(pred==y[test])>=.9
    changed=x[test].copy();changed[-1]*=1e5
    again=neural_predictions(x[fit],y[fit],changed,[30],17)[30]
    assert np.array_equal(again[:-1],pred[:-1])
