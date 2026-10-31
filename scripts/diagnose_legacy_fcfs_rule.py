"""Isolate legacy FCFS head-of-line admission on the original 400 training cases."""
import argparse,csv,hashlib,json,statistics,sys
from collections import deque
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rware.fcfs_cross_simulation import FCFSCrossExperiment

class FeasibleFCFS(FCFSCrossExperiment):
    bypasses=0
    def _admit_next_fcfs(self,area=None):
        area=area or self.area;queue=self.fcfs_queues[area.area_id]
        eligible=next((rid for rid in queue if rid in self.active and self.active[rid].current_node in area.waiting_points and self._compatible_with_shared_area(self.active[rid],area)),None)
        if eligible is not None and queue[0]!=eligible:
            self.bypasses+=1
            self.fcfs_queues[area.area_id]=deque([eligible]+[rid for rid in queue if rid!=eligible])
            if area.area_id==self.area.area_id:self.fcfs_queue=self.fcfs_queues[area.area_id]
        return super()._admit_next_fcfs(area)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scenarios',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    with a.scenarios.open() as f:scenarios=list(csv.DictReader(f))
    rows=[]
    for s in scenarios:
        kwargs=dict(robots_by_direction={d:int(s[d.lower()+'_agvs']) for d in ('NORTH','SOUTH','WEST','EAST')},random_seed=int(s['scenario_seed']),corridor_length=8,spawn_gap_steps=0,spawn_plan_by_direction=json.loads(s['spawn_plan_json']),goal_plan_by_direction=json.loads(s['goal_plan_json']),admission_window_steps=2,shared_area_capacity=2,policy_type='fcfs')
        a1=FCFSCrossExperiment(**kwargs);a2=FeasibleFCFS(**kwargs)
        def missions(ex):return [(r.robot_id,r.path,r.metadata) for pending in ex.pending_by_direction.values() for r in pending]
        assert missions(a1)==missions(a2)
        for rule,ex in [('strict_first',a1),('feasible_first',a2)]:
            r=ex.run(max_steps=1500);rows.append({'scenario':s['run_index'],'type':s['scenario_case_id'],'rule':rule,'robots':sum(kwargs['robots_by_direction'].values()),'completed':r['completed'],'total_steps':r['total_time'],'bypasses':getattr(ex,'bypasses',0)})
    with (a.output_dir/'legacy_fcfs_paired.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={rule:statistics.mean(r['total_steps'] for r in rows if r['rule']==rule) for rule in ('strict_first','feasible_first')}
    summary.update({'episodes':len(rows),'unique_training_scenarios':len(scenarios),'incomplete_runs':sum(r['completed']<r['robots'] for r in rows),'test_evaluated':False,'scenario_file_sha256':hashlib.sha256(a.scenarios.read_bytes()).hexdigest(),'purpose':'Diagnostic rule-only comparison, not reproduction of every historical reported experiment.'})
    (a.output_dir/'audit.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
