"""Check leakage, construction invariants and a known positive control."""
import numpy as np
import pandas as pd
import pytest

from .benchmark import classifier, strict_mask
from .data import CHANNELS, LABELS, Recording, chronological_split, load_recording
from .features import band_features, training_additions


def test_real_trials_never_enter_both_sides():
    ids = np.arange(160)
    y = np.tile(np.arange(16), 10)
    rec = Recording(np.zeros((160,14,1280)), y, ids, ids.astype(float), np.zeros(160,dtype=bool), 256,
                    {'blocks':10, 'id_to_block':{int(i):int(i//16) for i in ids}})
    train, test = chronological_split(rec)
    assert set(rec.trial_ids[train]).isdisjoint(rec.trial_ids[test])
    assert len(train) == 112 and len(test) == 48
    rng = np.random.default_rng(4)
    x = rng.normal(size=(len(train),14,1024)).astype('float32')
    features = band_features(x,256)
    for method in ('same_label_splice','random_label_splice','same_label_feature_mixup'):
        added, labels, parents = training_additions(x,y[train],features,method,256,17)
        assert added.shape == features.shape
        for label, donors in zip(labels,parents):
            assert set(rec.trial_ids[train[donors]]).isdisjoint(rec.trial_ids[test])
            if method.startswith('same_label'):
                assert all(y[train[donors]] == label)
                assert len(set(donors)) >= 2


def test_block_shuffle_preserves_the_selected_features():
    x = np.random.default_rng(5).normal(size=(20,14,1024)).astype('float32')
    f = band_features(x,256)
    added, _, _ = training_additions(x,np.repeat([0,1],10),f,'shuffle_blocks',256,17)
    np.testing.assert_allclose(added,f,atol=3e-6,rtol=1e-6)


def test_strict_rejection_does_not_fit_on_test_amplitudes():
    ptp = np.tile(np.linspace(2,3,14),(20,1))
    train = np.arange(12)
    flags = np.zeros(20,dtype=bool)
    before = strict_mask(ptp,flags,train)
    ptp[-1] = 1e8
    after = strict_mask(ptp,flags,train)
    assert np.array_equal(before[train],after[train])
    assert not after[-1]


def test_synthetic_frequency_positive_control_and_label_null():
    rng = np.random.default_rng(2)
    y = np.repeat([0,1],80)
    times = np.arange(1024)/256
    x = rng.normal(scale=0.1,size=(160,14,1024))
    x += np.sin(2*np.pi*np.where(y==0,10,20)[:,None,None]*times[None,None,:])
    f = band_features(x,256)
    train = np.r_[np.arange(60),np.arange(80,140)]
    test = np.r_[np.arange(60,80),np.arange(140,160)]
    predicted = classifier().fit(f[train],y[train]).predict(f[test])
    assert np.mean(predicted==y[test]) == 1
    scores=[]
    for _ in range(30):
        shuffled=rng.permutation(y[train])
        predicted=classifier().fit(f[train],shuffled).predict(f[test])
        scores.append(np.mean(predicted==y[test]))
    assert 0.3 < np.mean(scores) < 0.7


def test_parser_rejects_a_changing_trial_label(tmp_path):
    rows=[]
    for trial,label in enumerate(LABELS):
        for sample in range(1280):
            rows.append([trial*22+11+sample/256,trial,*([float(sample)]*14),label,'thinking',''])
    frame=pd.DataFrame(rows,columns=['Time:256Hz','Epoch',*CHANNELS,'Label','Stage','Flag'])
    frame.loc[1,'Label']='incorrect'
    file=tmp_path/'thinking.csv'
    frame.to_csv(file,index=False)
    with pytest.raises(ValueError,match='changing trial label'):
        load_recording(file,'thinking')
