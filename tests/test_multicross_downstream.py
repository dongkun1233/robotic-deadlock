import sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rware.agv_multicross_downstream_features import downstream_nodes,occupancy
class DownstreamTests(unittest.TestCase):
 def test_corridor_starts_at_first_exit_and_stops_at_next_waiting_point(self):
  r=SimpleNamespace(index=0,path=['WAIT','CP1','CP2','EXIT','LANE','NEXT_WAIT','NEXT_CP','GOAL']);a=SimpleNamespace(conflict_zone_nodes={'CP1','CP2'})
  self.assertEqual(downstream_nodes(r,a,{'WAIT','NEXT_WAIT'}),['EXIT','LANE','NEXT_WAIT'])
 def test_terminal_corridor_and_union_without_double_counting(self):
  r=SimpleNamespace(index=0,path=['WAIT','CP','EXIT','GOAL']);a=SimpleNamespace(conflict_zone_nodes={'CP'})
  self.assertEqual(downstream_nodes(r,a,{'WAIT'}),['EXIT','GOAL'])
  self.assertEqual(occupancy(['EXIT','GOAL'],{'EXIT':1},{'EXIT','OUTSIDE'}),.5)
 def test_empty_corridor_rejected(self):
  with self.assertRaises(ValueError):occupancy([],{},set())
if __name__=='__main__':unittest.main()
