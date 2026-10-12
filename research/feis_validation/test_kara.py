"""Native interval mapping, repeated labels, chronology and artifact-fit isolation."""
import numpy as np
import pytest
from .kara_data import epoch_pairs,word_split,class_inner_folds,fit_eye_regression,delivered_prompt_mapping,transform_variable_phase,sensor_prompt_mapping

def test_matlab_inclusive_indices_and_invalid_overlap():
    cells=np.empty((1,2),dtype=object);cells[0,0]=np.array([[1,5000]]);cells[0,1]=np.array([[6001,11000]])
    np.testing.assert_array_equal(epoch_pairs(cells,12000),[[0,5000],[6000,11000]])
    cells[0,1]=np.array([[4999,11000]])
    with pytest.raises(ValueError):epoch_pairs(cells,12000)

def test_per_class_chronology_and_nested_isolation():
    y=np.tile(np.arange(4),12);ids=np.arange(100,148)
    train,test,folds=word_split(y,ids)
    assert len(train)==32 and len(test)==16 and set(train).isdisjoint(test)
    for fit,val in folds:
        assert set(fit).isdisjoint(val) and set(fit).isdisjoint(test) and set(val).isdisjoint(test)
        for label in range(4):assert max(ids[fit[y[fit]==label]])<min(ids[val[y[val]==label]])

def test_repeated_delivery_alignment_requires_unique_mapping():
    assert delivered_prompt_mapping(['a','a','b'],['a','a','b'])==[0,1,2]
    assert delivered_prompt_mapping(['a','x','b'],['a','b'])==[0,2]
    with pytest.raises(ValueError):delivered_prompt_mapping(['a','a','b'],['a','b'])

def test_eye_regression_is_training_only():
    rng=np.random.default_rng(11);eye=rng.normal(size=(20,2,512));x=np.einsum('ce,net->nct',rng.normal(size=(5,2)),eye)+rng.normal(scale=.02,size=(20,5,512))
    train=np.arange(12);_,coef=fit_eye_regression(x,eye,train)
    x[-1]*=1000;eye[-1]*=10000
    _,after=fit_eye_regression(x,eye,train)
    np.testing.assert_array_equal(coef,after)

def test_variable_native_epoch_preserves_exact_central_four_seconds():
    rng=np.random.default_rng(8);x=rng.normal(size=(6,4912))
    result=transform_variable_phase(x)
    assert result.shape==(6,512)
    np.testing.assert_allclose(result.mean(axis=0),0,atol=1e-6)
    with pytest.raises(ValueError):transform_variable_phase(x[:,:3900])

def test_sensor_resolves_adjacent_same_prompt_and_rejects_wrong_label():
    sensor=np.zeros(60000);thinking=np.array([[10000,15000],[30000,35000],[40000,45000]])
    all_starts=[10000,20000,30000,40000]
    for start in all_starts:sensor[start:start+5000]=-2
    clearing=np.array([[5000,10000],[25000,30000],[35000,40000]])
    actual=['a','a','b','c'];accepted=['a','b','c']
    mapping,audit=sensor_prompt_mapping(sensor,thinking,clearing,actual,accepted)
    assert mapping==[0,2,3] and audit['unretained_attempt_indices']==[1]
    with pytest.raises(ValueError):sensor_prompt_mapping(sensor,thinking,clearing,actual,['a','b','a'])

def test_global_chronological_selection_never_uses_later_labels():
    y=np.tile(np.arange(4),12);ids=np.arange(48);train=np.arange(32);test=np.arange(32,48)
    for fit,val in class_inner_folds(y,ids,train):
        assert (set(fit)|set(val)).issubset(train) and (set(fit)|set(val)).isdisjoint(test)
    assert max(ids[train])<min(ids[test])

def test_covariance_regularization_is_not_dominated_by_volt_scale_floor():
    from .kara_benchmark import scaled_covariance_features
    from .followup_models import covariance_features
    x=np.random.default_rng(4).normal(scale=5e-6,size=(20,4,512))
    a=scaled_covariance_features(x);b=covariance_features(x*1e9)
    np.testing.assert_allclose(a-a.mean(axis=0),b-b.mean(axis=0),atol=1e-10)

def test_unused_phoneme_outlier_cannot_bypass_target_word_timing_limits():
    s=np.zeros(60000)
    for start in [10000,20000,30000,40000]:s[start:start+5000]=-2
    intervals=np.array([[10000,15000],[29800,34800],[40000,45000]])
    clears=np.array([[5000,10000],[25000,30000],[35000,40000]])
    mapping,audit=sensor_prompt_mapping(s,intervals,clears,['a','a','b','c'],['a','b','c'],[True,False,True])
    assert mapping==[0,2,3] and audit['unused_nonword_timing_outlier_indices']==[1]
    with pytest.raises(ValueError):sensor_prompt_mapping(s,intervals,clears,['a','a','b','c'],['a','b','c'],[True,True,True])

def test_sensor_levels_before_supplied_trial_span_do_not_shift_delivered_labels():
    s=np.zeros(70000)
    for start in [0,10000,20000,30000,40000]:s[start:start+5000]=-2
    thinking=np.array([[10000,15000],[30000,35000],[40000,45000]])
    clear=np.array([[5000,10000],[25000,30000],[35000,40000]])
    mapping,audit=sensor_prompt_mapping(s,thinking,clear,['a','a','b','c'],['a','b','c'])
    assert mapping==[0,2,3] and audit['unmatched_sensor_intervals_outside_supplied_task_bounds']==[[0,5000]]
