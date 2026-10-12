import json, unittest, argparse, sys
from pathlib import Path
import numpy as np
from reanalyse import ROOT, MAP, words, indices, plan, normalized, load_bank, DATA
from model import neurovascular_from_cmro2
from sphere import sphere_potential

class ResearchChecks(unittest.TestCase):
    def test_contractions_and_pronunciations(self):
        lex=json.loads((ROOT/'inputs/pronunciations.json').read_text())['lexicon']
        self.assertEqual(words("It's humanity’s resilience."),["It's","humanity's","resilience"])
        self.assertEqual(indices('It’s',lex),indices("It's",lex))
        self.assertEqual(indices('the',lex)[0], [31,5])
        self.assertEqual(MAP['B'],28)
        with self.assertRaises(KeyError):indices('definitelymissingword',lex)

    def test_insertion_preserves_task_content(self):
        ids=[[1,2],[3],[4,5,6]]
        expected=[(0,k) for group in ids for k in group]
        for seed in range(50):
            pp,cc=plan(ids,np.random.default_rng(seed),'wait','phoneme')
            self.assertEqual([x for x in pp if x[0]==0],expected)
            self.assertEqual(len(pp),len(cc))
            self.assertTrue(len(expected)+2<=len(pp)<=len(expected)+4)

    def test_physical_balance_and_baseline(self):
        c=np.linspace(2.1,3.1,50);nv=neurovascular_from_cmro2(c)
        np.testing.assert_allclose(nv['CBF']*nv['OEF']*6.9,c)
        self.assertAlmostEqual(nv['OEF'][0],.42)
        self.assertTrue(np.all(np.diff(nv['CBF'])>0))
        self.assertTrue(np.all(np.diff(nv['OEF'])<0))
        for val in nv.values():self.assertTrue(np.isfinite(val).all())

    def test_fixed_normalization_and_clipping(self):
        np.testing.assert_allclose(normalized(np.array([-1,0,5,10,11]),(0,10)),[0,0,.5,1,1])
        with self.assertRaises(ValueError):normalized(np.ones(2),(1,1))

    def test_sphere_center_limit(self):
        e=np.array([[0,0,1],[1,0,0],[0,0,-1]],float)
        np.testing.assert_allclose(sphere_potential([0,0,1e-8],1,0,e),[3,0,-3],atol=1e-6)

    def test_real_marker_selections(self):
        task,wait,audit,events,_=load_bank(DATA)
        self.assertEqual(task.shape,(44,5,250,16))
        self.assertEqual(wait.shape,(44,4,250,16))
        self.assertEqual(len(events),220)
        self.assertTrue(all(x['old_block_entirely_before_first_task'] for x in audit))
        raw=np.loadtxt(DATA/'DLR_0_1.txt')
        first=events[0]['start_row']
        np.testing.assert_array_equal(task[0,0],raw[first:first+250,1:17])
        self.assertFalse(np.array_equal(task[0,0],task[0,1]))

if __name__=='__main__':
    ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--data',type=Path,default=DATA)
    args,rest=ap.parse_known_args();DATA=args.data
    unittest.main(argv=[sys.argv[0]]+rest,verbosity=2)
