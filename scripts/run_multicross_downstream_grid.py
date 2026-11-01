"""Small predeclared training-only grid, with zero-penalty baseline regression."""
import argparse,csv,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rware import agv_multicross_downstream_augmented as sim
from rware import agv_multicross_policy_v2 as v2
from rware.agv_multicross_experiments import digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['manifest','training-dir','output-dir']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);m=json.loads(a.manifest.read_text());st=json.loads((a.training_dir/'checkpoint.json').read_text());best=min([t for t in st['trials'] if t['algorithm']=='basic'],key=lambda t:(t['objective'],t['seed'],t['evaluation']));grid=[0.,.25,.5,1.,2.];protocol={'split':'train','penalties':grid,'families':['waiting_downstream','full_downstream'],'full_weights':best['weights'],'objective':'Completion then mean makespan; report wait separately','validation_or_test_allowed':False,'source_sha256':hashlib.sha256(Path(sim.__file__).read_bytes()).hexdigest()};(a.output_dir/'grid_fixed_before_run.json').write_text(json.dumps(protocol,indent=2));rows=[]
    for family in protocol['families']:
        for penalty in grid:
            for s in [s for s in m['scenarios'] if s['split']=='train']:
                kwargs=dict(robots=s['robots'],seed=s['seed'],spawn_window=s['spawn_window'],capacity=m['capacity'],max_steps=m['max_steps']);r=sim.run(**kwargs,policy=family,weights=best['weights'],downstream_penalty=penalty)
                assert digest([{k:x[k] for k in ('id','start','goal','planned','path')} for x in r['robot_results']])==s['missions_sha256']
                if penalty==0:
                    old=v2.run(**kwargs,policy='fcfs' if family=='waiting_downstream' else 'heuristic',weights=best['weights']);assert r['robot_results']==old['robot_results']
                rows.append(dict(family=family,penalty=penalty,scenario_id=s['scenario_id'],case=s['case'],robots=s['robots'],completed=r['completed'],timed_out=r['timed_out'],total_steps=r['total_steps'],mean_wait=r['mean_wait_steps_all_agents'],max_wait=r['max_wait_steps_all_agents']))
    def save(name,rr):
        with (a.output_dir/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    save('training_raw.csv',rows);summary=[]
    for family in protocol['families']:
        for penalty in grid:
            rr=[r for r in rows if r['family']==family and r['penalty']==penalty];summary.append(dict(family=family,penalty=penalty,mean_steps=statistics.mean(r['total_steps'] for r in rr),mean_wait=statistics.mean(r['mean_wait'] for r in rr),worst_wait=max(r['max_wait'] for r in rr),timeouts=sum(r['timed_out'] for r in rr)))
    save('summary.csv',summary);(a.output_dir/'audit.json').write_text(json.dumps(dict(episodes=len(rows),expected=800,duplicate_keys=len(rows)-len({(r['family'],r['penalty'],r['scenario_id']) for r in rows}),timeouts=sum(r['timed_out'] for r in rows),zero_penalty_trajectories_preserved=True,validation_or_test_simulated=False,scope='Exploratory training-only grid; no independent efficacy claim.'),indent=2));print(summary)
if __name__=='__main__':main()
