import math,sys,unittest
from dataclasses import replace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rware.agv_layouts import AGVLayout
from rware.agv_topology import LaneGraph
from rware.agv_types import Node
from rware.agv_timed_simulation import run,build_layout,make_missions,ticks
from rware.agv_physical_time import AGVPhysicalTimeConfig

def lane(positions,bidirectional=False):
 g=LaneGraph()
 for i,p in enumerate(positions):g.add_node(Node(str(i),p,'normal_lane',['forward','wait']))
 for i in range(len(positions)-1):
  g.add_adjacent_edge(str(i),str(i+1))
  if bidirectional:g.add_adjacent_edge(str(i+1),str(i))
 return AGVLayout('fixture',(5,5),g,[], '',{},{},{})

class TimedSimulationTests(unittest.TestCase):
 def test_isolated_straight_duration_and_pending_spawn_accounting(self):
  l=lane([(0,i) for i in range(5)]);missions=[dict(id=0,path=['0','1','2','3','4'],planned_s=.03)]
  for dt in [.1,.05,.01]:
   r=run(l,missions,dt_s=dt,max_time_s=20);expected=ticks(.03,dt)*dt+4*ticks(1.875,dt)*dt
   self.assertAlmostEqual(r['makespan_s'],expected);self.assertAlmostEqual(r['robots_detail'][0]['stationary_wait_s'],0);self.assertAlmostEqual(r['robots_detail'][0]['spawn_delay_s'],ticks(.03,dt)*dt-.03)
 def test_spin_acceleration_and_phase_quantization(self):
  l=lane([(0,0),(0,1),(1,1)]);c=AGVPhysicalTimeConfig()
  for dt in [.1,.05,.01]:
   r=run(l,[dict(id=0,path=['0','1','2'],planned_s=0)],dt_s=dt,max_time_s=20);self.assertAlmostEqual(r['makespan_s'],(2*ticks(1.875,dt)+ticks(c.rotation_time_s(math.pi/2),dt))*dt);self.assertAlmostEqual(r['robots_detail'][0]['rotation_s'],c.rotation_time_s(math.pi/2));self.assertEqual(len([e for e in r['events'] if e['event']=='rotation_finished']),1)
 def test_initial_half_turn_is_timed(self):
  l=lane([(0,0),(0,1)]);c=AGVPhysicalTimeConfig();r=run(l,[dict(id=0,path=['0','1'],planned_s=0,initial_heading=(0,-1))],dt_s=.01,max_time_s=20)
  self.assertAlmostEqual(r['makespan_s'],(ticks(c.rotation_time_s(math.pi),.01)+ticks(1.875,.01))*.01)
  self.assertAlmostEqual(r['robots_detail'][0]['rotation_s'],c.rotation_time_s(math.pi))
 def test_head_on_cell_exchange_is_not_allowed(self):
  l=lane([(0,0),(0,1)],bidirectional=True);r=run(l,[dict(id=0,path=['0','1'],planned_s=0),dict(id=1,path=['1','0'],planned_s=0)],max_time_s=10)
  self.assertEqual(r['completed'],0);self.assertEqual(r['status'],'stalled');self.assertEqual(r['makespan_s'],10);self.assertTrue(r['waits_censored'])
 def test_same_source_spawns_without_stack(self):
  l=lane([(0,i) for i in range(3)]);r=run(l,[dict(id=i,path=['0','1','2'],planned_s=0) for i in range(3)],max_time_s=30)
  self.assertEqual(r['completed'],3);spawns=[x['spawned_s'] for x in r['robots_detail']];self.assertEqual(len(set(spawns)),3);self.assertGreater(r['mean_spawn_delay_s'],0)
 def test_common_eligibility_and_fcfs_waiting_equivalence(self):
  l=build_layout('multi');m=make_missions(l,12,101,5);a=run(l,m,policy='fcfs',record_decisions=True,max_time_s=400);b=run(l,m,policy='heuristic',weights=dict(waiting=1,maneuver=0,exit_competition=0,path_conflict=0),record_decisions=True,max_time_s=400)
  self.assertEqual(a['robots_detail'],b['robots_detail']);self.assertEqual(a['admission_log'],b['admission_log'])
  for r in [a,b]:
   for d in r['decision_log']:
    if d['admitted']:self.assertIn(d['selected'],d['eligible'])
 def test_strict_fcfs_is_separate_and_can_block_alternatives(self):
  l=build_layout('single');m=make_missions(l,8,101,10);a=run(l,m,policy='fcfs',max_time_s=400);b=run(l,m,policy='strict_fcfs',record_decisions=True,max_time_s=400)
  self.assertEqual(a['completed'],8);self.assertEqual(b['completed'],8);self.assertTrue(any(d['eligible'] and not d['admitted'] for d in b['decision_log']));self.assertGreater(b['makespan_s'],a['makespan_s'])
 def test_zone_release_after_actual_exit_and_no_readmission(self):
  l=build_layout('single');m=make_missions(l,8,101,10);r=run(l,m,policy='fcfs',max_time_s=400)
  for agent in r['robots_detail']:
   es=[e for e in r['events'] if e['robot']==agent['id']];ad=[e for e in es if e['event']=='admitted'];re=[e for e in es if e['event']=='zone_released'];self.assertEqual(len(ad),1);self.assertEqual(len(re),1)
   self.assertTrue(any(e['event']=='move_finished' and e['t']==re[0]['t'] for e in es));self.assertGreater(re[0]['t'],ad[0]['t'])
 def test_all_layout_pairs_are_reachable_with_protected_entries(self):
  from rware.agv_multicross_policy_simulation import shortest_path
  for name in ['single','double','multi']:
   l=build_layout(name);starts=[n for n in l.stations if n.endswith(('_IN','_START'))];goals=[n for n in l.stations if n.endswith(('_OUT','_EXIT'))]
   for start in starts:
    for goal in goals:run(l,[dict(id=0,path=shortest_path(l.graph,start,goal),planned_s=0)],max_time_s=400)
 def test_envelope_too_large_for_lane_rejected(self):
  l=lane([(0,0),(0,1)])
  with self.assertRaises(ValueError):run(l,[dict(id=0,path=['0','1'],planned_s=0)],config=replace(AGVPhysicalTimeConfig(),safety_margin_m=.4))
if __name__=='__main__':unittest.main()
