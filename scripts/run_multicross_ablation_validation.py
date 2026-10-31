"""Validate reoptimized leave-one-out models against preserved v2 full models."""
import argparse,csv,hashlib,json,statistics,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rware import agv_multicross_policy_v2 as sim
from rware import agv_multicross_policy_simulation as original
from rware import agv_multicross_layout as layout_module
from rware.agv_multicross_experiments import digest

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def save(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(rows,scenarios,seeds):
    models=['full']+['without_'+f for f in sim.FEATURES];summary=[];paired=[];cases=np.array([s['case'] for s in scenarios]);rng=np.random.default_rng(20261101)
    lookup={(r['model'],r['algorithm'],str(r['optimizer_seed']),r['scenario_id']):r for r in rows}
    for alg in ('basic','bo','pso'):
        for model in models:
            rr=[r for r in rows if r['algorithm']==alg and r['model']==model]
            summary.append({'algorithm':alg,'model':model,'episodes':len(rr),'unique_scenarios':len(scenarios),'mean_steps':statistics.mean(r['total_steps'] for r in rr),'timeouts':sum(r['timed_out'] for r in rr),'mean_wait':statistics.mean(r['mean_wait'] for r in rr),'worst_agent_wait':max(r['max_wait'] for r in rr)})
            if model=='full':continue
            d=[];incomplete=False
            for s in scenarios:
                pairs=[(lookup[(model,alg,str(seed),s['scenario_id'])],lookup[('full',alg,str(seed),s['scenario_id'])]) for seed in seeds]
                d.append(statistics.mean(a['total_steps']-b['total_steps'] for a,b in pairs));incomplete|=any(a['timed_out'] or b['timed_out'] for a,b in pairs)
            d=np.array(d);draws=np.concatenate([rng.choice(np.where(cases==case)[0],size=(5000,sum(cases==case)),replace=True) for case in sorted(set(cases))],axis=1);lo,hi=np.percentile(d[draws].mean(axis=1),[2.5,97.5])
            paired.append({'algorithm':alg,'model':model,'mean_difference_vs_full':float(d.mean()),'bootstrap95_lower':float(lo) if not incomplete else '', 'bootstrap95_upper':float(hi) if not incomplete else '', 'unique_scenarios':len(d),'timeout_affected':incomplete,'interval_note':'Exploratory, unadjusted; scenario bootstrap conditional on fitted optimizer seeds.'})
    return summary,paired

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','full-training-dir','ablation-training-dir','full-validation-dir','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);started=time.perf_counter();m=json.loads(a.manifest.read_text());scenarios=[s for s in m['scenarios'] if s['split']=='validation'];full=json.loads((a.full_training_dir/'checkpoint.json').read_text());ab=json.loads((a.ablation_training_dir/'checkpoint.json').read_text())
    for key in ('manifest_sha256','budget','initial_count','seeds','simulator','simulator_source_sha256','source_sha256','search_source_sha256','versions','objective','training_scenarios','capacity','max_steps'):assert full['config'][key]==ab['config'][key],key
    assert ab['config']['manifest_sha256']==hashlib.sha256(a.manifest.read_bytes()).hexdigest();assert ab['config']['simulator']=='v2';assert ab['config']['simulator_source_sha256']==hashlib.sha256(Path(sim.__file__).read_bytes()).hexdigest()
    for module in (layout_module,original):assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==m['source_sha256'][Path(module.__file__).name]
    assert set(ab['config']['models'])=={'without_'+f for f in sim.FEATURES}
    # Reuse the exact preserved full-model validation runs, not a favorable rerun.
    rows=[];fields=['model','algorithm','optimizer_seed','scenario_id','case','robots','completed','timed_out','total_steps','mean_wait','max_wait','mean_spawn_delay']
    for r in read(a.full_validation_dir/'validation_raw.csv'):
        rows.append({'model':'baseline' if r['algorithm']=='fcfs' else 'full',**{k:(r[k]=='True' if k=='timed_out' else int(r[k]) if k in ('case','robots','completed','total_steps') else float(r[k]) if k in ('mean_wait','max_wait','mean_spawn_delay') else r[k]) for k in fields if k!='model'}})
    full_audit=json.loads((a.full_validation_dir/'audit.json').read_text())
    assert full_audit['selected_on_training_only'] and full_audit['simulator']=='v2' and full_audit['duplicate_keys']==0
    assert len(rows)==400
    with (a.output_dir/'validation_raw.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        for best in read(a.ablation_training_dir/'best_training_weights.csv'):
            matching=[t for t in ab['trials'] if (t['model'],t['algorithm'],str(t['seed']))==(best['model'],best['algorithm'],best['seed'])];assert float(best['objective'])==min(t['objective'] for t in matching);weights=json.loads(best['weights_json']);assert weights[best['model'].removeprefix('without_')]==0
            print(best['model'],best['algorithm'],best['seed'],flush=True)
            for s in scenarios:
                r=sim.run(robots=s['robots'],seed=s['seed'],spawn_window=s['spawn_window'],capacity=m['capacity'],max_steps=m['max_steps'],weights=weights,policy='heuristic')
                assert digest([{k:x[k] for k in ('id','start','goal','planned','path')} for x in r['robot_results']])==s['missions_sha256']
                row=dict(model=best['model'],algorithm=best['algorithm'],optimizer_seed=best['seed'],scenario_id=s['scenario_id'],case=s['case'],robots=s['robots'],completed=r['completed'],timed_out=r['timed_out'],total_steps=r['total_steps'],mean_wait=r['mean_wait_steps_all_agents'],max_wait=r['max_wait_steps_all_agents'],mean_spawn_delay=r['mean_spawn_delay']);rows.append(row);writer.writerow(row)
            f.flush()
    expected=(1+5*3*len(ab['config']['seeds']))*len(scenarios);assert len(rows)==expected;assert len(rows)==len({(r['model'],r['algorithm'],str(r['optimizer_seed']),r['scenario_id']) for r in rows})
    summary,paired=summarize(rows,scenarios,ab['config']['seeds']);save(a.output_dir/'summary.csv',summary);save(a.output_dir/'paired_vs_full.csv',paired)
    audit={'expected_rows':expected,'actual_rows':len(rows),'reused_full_and_fcfs_rows':400,'new_ablation_rows':len(rows)-400,'duplicate_keys':0,'incomplete_runs':sum(r['timed_out'] for r in rows),'training_only_weight_selection':True,'test_evaluated':False,'validation_reuse_note':'Development set previously inspected; not a new independent test.','full_raw_sha256':hashlib.sha256((a.full_validation_dir/'validation_raw.csv').read_bytes()).hexdigest(),'elapsed_seconds':time.perf_counter()-started}
    (a.output_dir/'audit.json').write_text(json.dumps(audit,indent=2));print(audit)
    for r in summary:print(r)
    for r in paired:print(r)
if __name__=='__main__':main()
