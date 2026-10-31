"""Evaluate training-selected weights on validation only; never runs test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rware import agv_multicross_policy_simulation as sim
from rware import agv_multicross_layout as layout_module
from rware.agv_multicross_experiments import digest


def write_csv(path,rows):
    with path.open('x',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--training-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--simulator',choices=['v1','v2'],default='v1')
    args=p.parse_args()
    global sim
    original_sim=sim
    if args.simulator=='v2':
        from rware import agv_multicross_policy_v2
        sim=agv_multicross_policy_v2
    args.output_dir.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.manifest.read_text())
    checkpoint=json.loads((args.training_dir/'checkpoint.json').read_text())
    assert checkpoint['config'].get('simulator','v1')==args.simulator
    if args.simulator=='v2':
        assert checkpoint['config']['simulator_source_sha256']==hashlib.sha256(Path(sim.__file__).read_bytes()).hexdigest()
    assert checkpoint['config']['manifest_sha256']==hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    for module in (layout_module,original_sim):
        name=Path(module.__file__).name
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==manifest['source_sha256'][name]
    scenarios=[s for s in manifest['scenarios'] if s['split']=='validation']
    assert len(scenarios)==manifest['split_counts']['validation']
    with (args.training_dir/'best_training_weights.csv').open() as file:
        best=list(csv.DictReader(file))
    policies=[('fcfs','baseline',None)]
    for row in best:
        assert row['model']=='full','This validation stage is for the full model only'
        matches=[t for t in checkpoint['trials'] if t['model']==row['model'] and t['algorithm']==row['algorithm'] and t['seed']==int(row['seed'])]
        assert float(row['objective'])==min(t['objective'] for t in matches)
        policies.append((row['algorithm'],row['seed'],json.loads(row['weights_json'])))
    started=time.perf_counter()
    all_rows=[]
    raw_path=args.output_dir/'validation_raw.csv'
    fields=['algorithm','optimizer_seed','scenario_id','case','robots','completed','timed_out','total_steps','mean_wait','max_wait','mean_spawn_delay']
    with raw_path.open('x',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for algorithm,seed,weights in policies:
            print(algorithm,seed,flush=True)
            for s in scenarios:
                result=sim.run(robots=s['robots'],seed=s['seed'],spawn_window=s['spawn_window'],policy='fcfs' if weights is None else 'heuristic',weights=weights,capacity=manifest['capacity'],max_steps=manifest['max_steps'])
                replay=[{k:r[k] for k in ('id','start','goal','planned','path')} for r in result['robot_results']]
                assert digest(replay)==s['missions_sha256']
                row={'algorithm':algorithm,'optimizer_seed':seed,'scenario_id':s['scenario_id'],'case':s['case'],'robots':s['robots'],
                    'completed':result['completed'],'timed_out':result['timed_out'],'total_steps':result['total_steps'],
                    'mean_wait':result['mean_wait_steps_all_agents'],'max_wait':result['max_wait_steps_all_agents'],'mean_spawn_delay':result['mean_spawn_delay']}
                writer.writerow(row);all_rows.append(row)
            file.flush()
    summary=[]
    for algorithm,seed,_ in policies:
        rows=[r for r in all_rows if r['algorithm']==algorithm and r['optimizer_seed']==seed]
        summary.append({'algorithm':algorithm,'optimizer_seed':seed,'scenarios':len(rows),'incomplete_scenarios':sum(r['timed_out'] for r in rows),
            'mean_capped_steps':statistics.mean(r['total_steps'] for r in rows),'mean_agent_wait':statistics.mean(r['mean_wait'] for r in rows),
            'worst_agent_wait':max(r['max_wait'] for r in rows)})
    write_csv(args.output_dir/'validation_summary_by_seed.csv',summary)
    paired=[]
    baseline={r['scenario_id']:r for r in all_rows if r['algorithm']=='fcfs'}
    rng=np.random.default_rng(20261031)
    for algorithm in ('basic','bo','pso'):
        differences=[];cases=[];incomplete=0
        for s in scenarios:
            rows=[r for r in all_rows if r['algorithm']==algorithm and r['scenario_id']==s['scenario_id']]
            assert len(rows)==len(checkpoint['config']['seeds'])
            incomplete+=sum(r['timed_out'] for r in rows)+int(baseline[s['scenario_id']]['timed_out'])
            differences.append(statistics.mean(r['total_steps'] for r in rows)-baseline[s['scenario_id']]['total_steps'])
            cases.append(s['case'])
        # Average over optimizer seeds first; then resample scenario pairs within case.
        diffs=np.array(differences);cases=np.array(cases)
        samples=[]
        for _ in range(5000):
            draw=np.concatenate([rng.choice(np.where(cases==case)[0],size=sum(cases==case),replace=True) for case in sorted(set(cases))])
            samples.append(float(diffs[draw].mean()))
        lo,hi=np.percentile(samples,[2.5,97.5])
        paired.append({'algorithm':algorithm,'unique_validation_scenarios':len(scenarios),'mean_difference_vs_fcfs':float(diffs.mean()),
            'bootstrap95_lower':float(lo) if not incomplete else '', 'bootstrap95_upper':float(hi) if not incomplete else '',
            'timeout_affected':bool(incomplete),'interpretation':'Scenario uncertainty conditional on these three fitted seeds; optimizer uncertainty is not fully estimated.'})
    write_csv(args.output_dir/'validation_paired_vs_fcfs.csv',paired)
    learning=[]
    for algorithm in ('basic','bo','pso'):
        for seed in checkpoint['config']['seeds']:
            trace=sorted([t for t in checkpoint['trials'] if t['algorithm']==algorithm and t['seed']==seed],key=lambda t:t['evaluation'])
            for budget in (6,12,18,24):
                subset=trace[:budget];chosen=min(subset,key=lambda t:t['objective'])
                learning.append({'algorithm':algorithm,'optimizer_seed':seed,'budget':budget,'best_training_steps':chosen['mean_total_steps'],'incomplete':chosen['incomplete'],'selected_evaluation':chosen['evaluation']})
    write_csv(args.output_dir/'training_learning_checkpoints.csv',learning)
    audit={'split':'validation','episodes':len(all_rows),'expected':len(policies)*len(scenarios),
        'simulator':args.simulator,
        'duplicate_keys':len(all_rows)-len({(r['algorithm'],r['optimizer_seed'],r['scenario_id']) for r in all_rows}),
        'replay_verified':True,'selected_on_training_only':True,'test_evaluated':False,
        'scenario_bootstrap_note':'Resamples unique scenario pairs within four cases after averaging the three optimizer seeds.',
        'elapsed_seconds':time.perf_counter()-started}
    assert audit['episodes']==audit['expected'] and audit['duplicate_keys']==0
    (args.output_dir/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
    for r in summary:print(r)
    for r in paired:print(r)


if __name__=='__main__':main()
