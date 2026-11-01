"""Fixed three-policy comparison on existing training cases only."""
import argparse,csv,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rware import agv_multicross_policy_v2 as v2
from rware import agv_multicross_downstream_pilot as pilot
from rware.agv_multicross_experiments import digest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--training-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);m=json.loads(a.manifest.read_text());state=json.loads((a.training_dir/'checkpoint.json').read_text());t=min([t for t in state['trials'] if t['algorithm']=='basic'],key=lambda t:(t['objective'],t['seed'],t['evaluation']));weights=t['weights'];rows=[]
    for s in [s for s in m['scenarios'] if s['split']=='train']:
        kwargs=dict(robots=s['robots'],seed=s['seed'],spawn_window=s['spawn_window'],capacity=m['capacity'],max_steps=m['max_steps'])
        for policy,w in [('fcfs',None),('heuristic',weights),('downstream',None)]:
            r=pilot.run(**kwargs,policy=policy,weights=w)
            assert digest([{k:x[k] for k in ('id','start','goal','planned','path')} for x in r['robot_results']])==s['missions_sha256']
            if policy!='downstream':
                old=v2.run(**kwargs,policy=policy,weights=w);assert old['robot_results']==r['robot_results'] and old['admission_log']==r['admission_log']
            rows.append(dict(scenario_id=s['scenario_id'],case=s['case'],policy=policy,robots=s['robots'],completed=r['completed'],timed_out=r['timed_out'],total_steps=r['total_steps'],mean_wait=r['mean_wait_steps_all_agents'],max_wait=r['max_wait_steps_all_agents'],mean_spawn_delay=r['mean_spawn_delay']))
    def save(name,rr):
        with (a.output_dir/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    save('training_raw.csv',rows);summary=[]
    for policy in ['fcfs','heuristic','downstream']:
        rr=[r for r in rows if r['policy']==policy];summary.append(dict(policy=policy,episodes=len(rr),mean_steps=statistics.mean(r['total_steps'] for r in rr),timeouts=sum(r['timed_out'] for r in rr),mean_wait=statistics.mean(r['mean_wait'] for r in rr),worst_wait=max(r['max_wait'] for r in rr)))
    save('summary.csv',summary);audit=dict(split='train',episodes=240,baseline_control_replays=160,baseline_trajectories_unchanged=True,validation_or_test_simulated=False,full_weights_selected_on_training_only=True,source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['rware/agv_multicross_downstream_pilot.py','rware/agv_multicross_downstream_features.py']},interpretation='Exploratory training comparison; full policy has already been fitted here. No held-out improvement claim.')
    (a.output_dir/'audit.json').write_text(json.dumps(audit,indent=2));print(summary)
if __name__=='__main__':main()
