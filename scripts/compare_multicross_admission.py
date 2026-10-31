"""Compare admission versions on training only with fixed v1-selected weights."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rware import agv_multicross_policy_simulation as v1
from rware import agv_multicross_policy_v2 as v2
from rware.agv_multicross_experiments import digest

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--old-training-dir',type=Path,required=True)
    p.add_argument('--failures',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    m=json.loads(a.manifest.read_text()); scenarios={s['scenario_id']:s for s in m['scenarios'] if s['split']=='train'}
    rows=[]
    def execute(s,w,policy):
        r=v2.run(robots=s['robots'],seed=s['seed'],spawn_window=s['spawn_window'],capacity=m['capacity'],max_steps=m['max_steps'],weights=w,policy=policy)
        assert digest([{k:x[k] for k in ('id','start','goal','planned','path')} for x in r['robot_results']])==s['missions_sha256']
        assert all(d['ranked_first'] in d['eligible'] for d in r['decision_log'] if d['ranked_first'] is not None)
        assert not any(d['blocked'] and d['eligible'] for d in r['decision_log'])
        return r
    with a.failures.open() as f:
        for old in csv.DictReader(f):
            r=execute(scenarios[old['scenario_id']],json.loads(old['weights_json']),'heuristic')
            rows.append({**old,'v2_completed':r['completed'],'v2_steps':r['total_steps'],'v2_timed_out':r['timed_out']})
    def save(name,data):
        with (a.output_dir/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    save('v1_failure_replays.csv',rows)
    with (a.old_training_dir/'best_training_weights.csv').open() as f:best=list(csv.DictReader(f))
    policies=[('fcfs','baseline',None)]+[(r['algorithm'],r['seed'],json.loads(r['weights_json'])) for r in best]
    fixed=[]
    for alg,seed,w in policies:
        print(alg,seed,flush=True)
        for s in scenarios.values():
            r=execute(s,w,'fcfs' if w is None else 'heuristic')
            fixed.append({'algorithm':alg,'optimizer_seed':seed,'scenario_id':s['scenario_id'],'robots':s['robots'],'completed':r['completed'],'timed_out':r['timed_out'],'total_steps':r['total_steps']})
    save('fixed_v1_weights_on_v2_training.csv',fixed)
    audit={'split':'train','failed_v1_episodes':len(rows),'failure_replays_completed':sum(not r['v2_timed_out'] for r in rows),'fixed_weight_episodes':len(fixed),'fixed_weight_timeouts':sum(r['timed_out'] for r in fixed),'missions_verified':True,'feasible_admission_asserted':True,'test_evaluated':False}
    (a.output_dir/'audit.json').write_text(json.dumps(audit,indent=2));print(audit)
if __name__=='__main__':main()
