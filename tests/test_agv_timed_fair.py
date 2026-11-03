import math
import random
from types import SimpleNamespace
import unittest
from rware import agv_timed_simulation as plain
from rware import agv_timed_fair_simulation as fair

class FairPriorityTests(unittest.TestCase):
    def test_protection_only_uses_eligible_candidates(self):
        active={1:SimpleNamespace(zone_wait_since=0.),2:SimpleNamespace(zone_wait_since=25.)}
        chosen,overdue=fair.fair_choice([1,2],[2],active,{1:0,2:1},30.,.01,20.)
        self.assertEqual(chosen,2);self.assertEqual(overdue,[])

    def test_oldest_overdue_beats_score_and_ties_use_queue_order(self):
        active={1:SimpleNamespace(zone_wait_since=0.),2:SimpleNamespace(zone_wait_since=5.),3:SimpleNamespace(zone_wait_since=25.)}
        chosen,overdue=fair.fair_choice([1,2,3],[1,2,3],active,{1:-2,2:2,3:1},30.,.01,20.)
        self.assertEqual(chosen,1);self.assertEqual(overdue,[1,2])

    def test_no_overdue_uses_weighted_score(self):
        active={1:SimpleNamespace(zone_wait_since=25.),2:SimpleNamespace(zone_wait_since=26.)}
        chosen,overdue=fair.fair_choice([1,2],[1,2],active,{1:-2,2:1},30.,.01,20.)
        self.assertEqual(chosen,2);self.assertFalse(overdue)

    def test_matches_teammate_large_constant_in_normalized_score_domain(self):
        rng=random.Random(491)
        for threshold in [20.,40.]:
            for _ in range(500):
                now=100.;dt=.01;queue=list(range(5))
                active={rid:SimpleNamespace(zone_wait_since=now-rng.randrange(0,10001)*dt) for rid in queue}
                queue.sort(key=lambda rid:(active[rid].zone_wait_since,rid))
                eligible=[rid for rid in queue if rng.random()<.8]
                scores={rid:rng.uniform(-2,2) for rid in queue}
                def legacy_score(rid):
                    age=fair.queue_wait_s(active[rid],now,dt)
                    return 1000000.+age if age>=threshold else scores[rid]
                legacy=min(eligible,key=lambda rid:(-legacy_score(rid),active[rid].zone_wait_since,rid)) if eligible else None
                actual,_=fair.fair_choice(queue,eligible,active,scores,now,dt,threshold)
                self.assertEqual(actual,legacy)

    def test_exact_tick_threshold_boundary(self):
        active={1:SimpleNamespace(zone_wait_since=.03),2:SimpleNamespace(zone_wait_since=1.)}
        chosen,overdue=fair.fair_choice([1,2],[1,2],active,{1:-1,2:1},20.03,.01,20.)
        self.assertIn(1,overdue);self.assertEqual(chosen,1)

    def test_disabled_rule_matches_original_physics_and_admission(self):
        for name in ['single','double','multi']:
            layout=plain.build_layout(name)
            missions=plain.make_missions(layout,8,711,5)
            for policy in ['fcfs','heuristic','strict_fcfs']:
                a=plain.run(layout,missions,policy=policy)
                b=fair.run(layout,missions,policy=policy,wait_threshold_s=None)
                for field in a:self.assertEqual(a[field],b[field],(name,policy,field))

    def test_threshold_zero_reduces_to_fcfs_on_same_eligible_pool(self):
        for name in ['single','double','multi']:
            layout=plain.build_layout(name);missions=plain.make_missions(layout,12,101,0)
            a=plain.run(layout,missions,policy='fcfs')
            b=fair.run(layout,missions,policy='heuristic',weights=dict(zip(plain.FEATURES,[-1.,1.,1.,1.])),wait_threshold_s=0.)
            self.assertEqual(a['robots_detail'],b['robots_detail'])
            self.assertEqual(a['admission_log'],b['admission_log'])

    def test_cumulative_entry_wait_bound_certifies_inactive_protection(self):
        for name in ['single','double','multi']:
            layout=plain.build_layout(name);missions=plain.make_missions(layout,12,715,0)
            weights=dict(zip(plain.FEATURES,[-.5,.4,-.6,.2]))
            a=fair.run(layout,missions,policy='heuristic',weights=weights)
            bound=max(r['entry_wait_s'] for r in a['robots_detail'])+.01
            b=fair.run(layout,missions,policy='heuristic',weights=weights,wait_threshold_s=bound)
            self.assertEqual(a['robots_detail'],b['robots_detail'])
            self.assertEqual(a['admission_log'],b['admission_log'])
            self.assertEqual(b['fairness_protected_admissions'],0)

    def test_threshold_validation(self):
        layout=plain.build_layout('single');missions=plain.make_missions(layout,2,1,0)
        for threshold in [-1,float('inf'),float('nan')]:
            with self.assertRaises(ValueError):fair.run(layout,missions,wait_threshold_s=threshold)

if __name__=='__main__':unittest.main()
