"""Protocol independence, contamination guards, and episode resume tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts import run_agv_final_test as final
from scripts.analyze_agv_final_test import holm,stratified

class FinalProtocolTests(unittest.TestCase):
    def test_protocol_counts_independence_and_lock_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'final'
            final.prepare(out)
            m,signature=final.load_locked(out)
            self.assertEqual(len(m['scenarios']),2052)
            self.assertEqual(sum(final.COUNTS[l]*len(m['policies'][l]) for l in final.COUNTS),38988)
            self.assertEqual(len({s['seed'] for s in m['scenarios']}),2052)
            prior=json.loads(final.EXCLUSIONS.read_text())
            forbidden={(s['layout'],s['missions_sha256']) for s in prior['excluded_scenarios']}
            self.assertFalse(any((s['layout'],s['missions_sha256']) in forbidden for s in m['scenarios']))
            for layout,n in final.COUNTS.items():
                for cell in range(12):
                    self.assertEqual(sum(s['layout']==layout and s['cell']==cell for s in m['scenarios']),n//12)
            original=(out/'manifest.json').read_bytes()
            final.prepare(out)
            self.assertEqual((out/'manifest.json').read_bytes(),original)
            self.assertFalse((out/'episodes').exists())
            m['dt_s']=.02
            (out/'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):final.load_locked(out)

    def test_invalid_episode_rejected(self):
        case=dict(scenario_id='x',missions_sha256='mission',layout='single',robots=12,cell=0)
        row=dict(**case,status='completed',completed=12,makespan_s=100.,safety_checks_passed=True)
        final.validate_row(row,case,1200.)
        for update in [dict(scenario_id='foreign'),dict(missions_sha256='foreign'),
                       dict(safety_checks_passed=False),dict(completed=11),dict(makespan_s=float('nan'))]:
            with self.assertRaises(ValueError):final.validate_row(dict(row,**update),case,1200.)

    def test_worker_resume_skips_saved_and_keeps_timeout(self):
        policy=dict(label='fcfs',policy='fcfs',vector=None)
        cases=[dict(scenario_id=f'toy_{i}',missions_sha256=f'hash{i}',layout='single',robots=1,cell=0) for i in range(3)]
        m=dict(scenarios=cases,policies={'single':[policy]},max_time_s=1200.)
        calls=[]
        def fake_run(engine,case,manifest,name,vector):
            calls.append(case['scenario_id'])
            return dict(**case,status='timeout' if case['scenario_id']=='toy_2' else 'completed',
                completed=0 if case['scenario_id']=='toy_2' else 1,makespan_s=1200. if case['scenario_id']=='toy_2' else 1.,safety_checks_passed=True)
        with tempfile.TemporaryDirectory() as temp,patch.object(final,'load_locked',return_value=(m,'signature')),patch.object(final,'build_layout',return_value=None),patch.object(final,'run_case',side_effect=fake_run):
            out=Path(temp)
            final.worker((str(out),'single','fcfs'))
            self.assertEqual(len(calls),3)
            final.worker((str(out),'single','fcfs'))
            self.assertEqual(len(calls),3)
            rows=final.collect(out,dict(m,policies={'single':[policy],'double':[],'multi':[]}),'signature',True)
            self.assertEqual(sum(r['status']=='timeout' for r in rows),1)
            path=final.episode_path(out,'single','fcfs',cases[0]);record=json.loads(path.read_text());record['manifest_sha256']='other';path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):final.worker((str(out),'single','fcfs'))

    def test_holm_and_pairing(self):
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
        mean,se,df,p=stratified(np.array([1.,2.,3.,4.,5.,6.]),[np.array([0,1,2]),np.array([3,4,5])])
        self.assertAlmostEqual(mean,3.5)
        self.assertGreater(se,0)
        self.assertGreater(df,0)
        self.assertTrue(0<=p<=1)

if __name__=='__main__':unittest.main()
