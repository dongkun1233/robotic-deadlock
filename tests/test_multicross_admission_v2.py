"""Behavioral regression for bypassing blocked first candidates safely."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rware import agv_multicross_policy_simulation as v1
from rware import agv_multicross_policy_v2 as v2

class AdmissionTests(unittest.TestCase):
    def test_strict_mode_preserves_v1_trajectories_and_feature_scores(self):
        for policy in ('fcfs','heuristic'):
            for seed in (101,202,303,701001,703001):
                a=v1.run(robots=24,seed=seed,policy=policy)
                b=v2.run(robots=24,seed=seed,policy=policy,admission_rule='strict_first')
                self.assertEqual(a['robot_results'],b['robot_results'])
                self.assertEqual(a['total_steps'],b['total_steps'])
                self.assertEqual(a['decision_log'],[{k:v for k,v in d.items() if k!='eligible'} for d in b['decision_log']])

    def test_known_head_of_line_timeout_is_resolved(self):
        weights={'exit_competition':.6024239755731946,'maneuver':.6895162145259789,'path_conflict':-.22160649568377633,'waiting':-.335474745146498}
        # Fixed train_c4_001 replay: no external manifest needed.
        kwargs=dict(robots=48,seed=703001,spawn_window=0,weights=weights,policy='heuristic')
        a=v1.run(**kwargs);b=v2.run(**kwargs)
        self.assertTrue(a['timed_out']);self.assertFalse(b['timed_out']);self.assertEqual(b['completed'],48)
        for decision in b['decision_log']:
            if decision['ranked_first'] is not None:self.assertIn(decision['ranked_first'],decision['eligible'])
            self.assertFalse(decision['blocked'] and decision['eligible'])

    def test_both_policies_use_feasible_admission(self):
        for policy in ('fcfs','heuristic'):
            for seed in range(20):
                r=v2.run(robots=48,seed=seed,spawn_window=0,policy=policy)
                self.assertEqual(r['completed'],48)
                for d in r['decision_log']:
                    if d['ranked_first'] is not None:self.assertIn(d['ranked_first'],d['eligible'])

if __name__=='__main__':unittest.main()
