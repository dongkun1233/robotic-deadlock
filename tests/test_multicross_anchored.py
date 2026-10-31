import importlib.util,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def module(name,file):
 s=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
search=module('anchored','scripts/run_multicross_anchored_search.py');evaluation=module('final_eval','scripts/evaluate_multicross_anchored_test.py')
class AnchoredTests(unittest.TestCase):
 def test_anchor_and_training_bound(self):
  initial=search.anchored_initial(101)
  np.testing.assert_array_equal(initial[0],[1,0,0,0]);np.testing.assert_allclose(np.linalg.norm(initial,axis=1),1)
  for alg in ('basic','bo','pso'):
   values=[]
   def f(v,source):
    value=float(np.sum((v-np.array([1.,0,0,0]))**2));values.append(value);return value
   search.search(alg,101,18,initial,f);self.assertEqual(len(values),18);self.assertEqual(min(values),0)
 def test_identical_policies_have_zero_paired_gap_and_unit_p(self):
  ss=[dict(scenario_id=str(i),case=i//2) for i in range(8)];rows=[]
  for alg in ('fcfs','basic','bo','pso'):
   for seed in (['baseline'] if alg=='fcfs' else [101,202]):
    for s in ss:rows.append(dict(algorithm=alg,optimizer_seed=seed,scenario_id=s['scenario_id'],total_steps=10+int(s['scenario_id'])))
  result=evaluation.analyze(rows,ss,[101,202],draws=100,permutations=1000)
  for r in result:
   self.assertEqual(r['mean_step_difference'],0);self.assertEqual(r['two_way95_lower'],0);self.assertEqual(r['two_way95_upper'],0);self.assertEqual(r['permutation_p_conditional'],1)
  self.assertEqual([r['holm_primary_p'] for r in result[:3]],[1,1,1])
if __name__=='__main__':unittest.main()
