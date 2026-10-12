import unittest
import numpy as np
from analyze import prediction, alignment_null, spectral, covariance, transform, ba

class AnalysisInvariants(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(27)
        centers=rng.normal(size=(44,80))*3
        y=np.repeat(np.arange(44),5)
        return centers[y]+rng.normal(size=(220,80))*.1,y,centers+ rng.normal(size=(44,80))*.1

    def test_positive_control_and_training_only_scale(self):
        x,y,q=self.fixture();p,d,_=prediction(x,y,q)
        self.assertGreater(ba(np.arange(44),p),.95)
        # Appending extreme unlabelled queries must not refit scaler or centers.
        p2,d2,_=prediction(x,y,np.concatenate([q,np.ones((3,80))*1e12]))
        np.testing.assert_array_equal(p,p2[:44]);np.testing.assert_allclose(d,d2[:44])

    def test_block_label_null_matches_actual_refit(self):
        x,y,q=self.fixture();p,_,_=prediction(x,y,q)
        rng=np.random.default_rng(88);perm=rng.permutation(44)
        refit,_,_=prediction(x,perm[y],q)
        np.testing.assert_array_equal(refit,perm[p])
        null,_=alignment_null(np.arange(44),p,np.ones(44),n=999)
        self.assertLess(abs(null.mean()-1/44),.004)
        self.assertLess(float((np.sum(null>=1)+1)/1000),.01)

    def test_covariance_scale_and_spectral_scale_invariance(self):
        rng=np.random.default_rng(19);x=rng.normal(size=(4,16,250))
        np.testing.assert_allclose(covariance(x),covariance(x*1000),atol=1e-10)
        np.testing.assert_allclose(spectral(x),spectral(x*.001),atol=1e-10)

    def test_reference_and_filter_are_independent_per_interval(self):
        rng=np.random.default_rng(3);a=rng.normal(size=(16,500));b=rng.normal(size=(16,500))
        baseline=transform(a)
        transform(b*1e6)
        np.testing.assert_array_equal(baseline,transform(a))
        np.testing.assert_allclose(baseline.mean(axis=0),0,atol=1e-10)

    def test_harmonic_sensitivity_preserves_defined_bands(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(3,16,250))
        f=spectral(x,True);self.assertEqual(f.shape,(3,80));self.assertTrue(np.isfinite(f).all())

if __name__=='__main__':unittest.main()
