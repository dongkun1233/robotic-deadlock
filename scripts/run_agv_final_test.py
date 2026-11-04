"""Frozen independent evaluation: prepare, run, status; no training or selection."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.agv_timed_main_common import (read_manifest,atomic_durable,build_layout,
    generate_missions,digest,run_case,FIXED4,runtime_hashes)

COUNTS={'single':804,'double':792,'multi':456}
BASE=ROOT/'outputs/agv_timed_main_v1'
OUT=ROOT/'outputs/agv_timed_final_v1'
EXCLUSIONS=ROOT/'configs/agv_final_prior_exclusions_20261104.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def frozen_inputs(base=BASE):
    m=read_manifest(base/'manifest.json')
    assert m['optimizer_seeds']==[101,202,303,404,505]
    assert m['methods']==['basic','bo','pso']
    selection=json.loads((base/'initial_weights.json').read_text())
    if selection['manifest_sha256']!=sha(base/'manifest.json'):
        raise ValueError('Initialization manifest differs')
    hashes={str((base/'manifest.json').relative_to(ROOT)):sha(base/'manifest.json'),
            str((base/'initial_weights.json').relative_to(ROOT)):sha(base/'initial_weights.json')}
    registry={}
    for layout in COUNTS:
        chosen=min(selection['selected_by_layout'][layout]['candidates'],key=lambda x:x['objective'])
        policies=[dict(label='fcfs',policy='fcfs',vector=None),
                  dict(label='strict_fcfs',policy='strict_fcfs',vector=None),
                  dict(label='fixed4_reference',policy='heuristic',vector=list(map(float,FIXED4))),
                  dict(label='warmstart_best',policy='heuristic',vector=chosen['vector'])]
        for seed in m['optimizer_seeds']:
            for method in m['methods']:
                job=base/'search'/f'{layout}_{seed}_{method}'
                state=json.loads((job/'checkpoint.json').read_text())
                complete=json.loads((job/'complete.json').read_text())
                fit=json.loads((job/'best.json').read_text())
                if len(state['trials'])!=72 or complete.get('complete') is not True:
                    raise ValueError('Training incomplete')
                best=min(state['trials'],key=lambda t:(t['objective'],t['evaluation']))
                for key in ['objective','vector','evaluation']:
                    if fit[key]!=best[key]:raise ValueError('Best fit differs from saved training optimum')
                for p in [job/'checkpoint.json',job/'best.json',job/'complete.json']:
                    hashes[str(p.relative_to(ROOT))]=sha(p)
                policies.append(dict(label=f'{method}_{seed}',policy='heuristic',vector=fit['vector']))
        assert len(policies)==19
        registry[layout]=policies
    return m,registry,hashes

def validate_row(row,case,max_time):
    if row['scenario_id']!=case['scenario_id'] or row['missions_sha256']!=case['missions_sha256']:
        raise ValueError('Episode identity differs')
    if row['layout']!=case['layout'] or row['robots']!=case['robots'] or row['cell']!=case['cell']:
        raise ValueError('Episode condition differs')
    if row['safety_checks_passed'] is not True:
        raise ValueError('Safety check failed')
    import math
    if not math.isfinite(row['makespan_s']) or not 0<=row['makespan_s']<=max_time+1e-8:
        raise ValueError('Invalid makespan')
    if row['status']=='completed' and row['completed']!=row['robots']:
        raise ValueError('Completion count differs')

def prepare(out=OUT):
    if out.exists():
        load_locked(out)
        print('Existing final protocol verified; no regeneration',flush=True)
        return
    m,registry,hashes=frozen_inputs()
    exclusions=json.loads(EXCLUSIONS.read_text())
    forbidden={(x['layout'],x['missions_sha256']) for x in exclusions['excluded_scenarios']}
    seeds=set(exclusions['excluded_seeds'])
    for s in m['scenarios']:
        if (s['layout'],s['missions_sha256']) not in forbidden or s['seed'] not in seeds:
            raise ValueError('Prior exclusions do not cover main train/development')
    cases=[];used=set()
    for li,(layout,n) in enumerate(COUNTS.items()):
        engine=build_layout(layout)
        for cell,(robots,window,profile) in enumerate(itertools.product(
                [12,24],[0,15],['uniform','approach_skew','exit_skew'])):
            for rep in range(n//12):
                seed=731000000+li*1000000+cell*10000+rep
                missions,start,goal=generate_missions(engine,robots,seed,window,profile)
                fingerprint=digest(missions);key=(layout,fingerprint)
                if key in forbidden or key in used or seed in seeds:
                    raise ValueError('Independent scenario collision; stop without reselection')
                used.add(key);seeds.add(seed)
                cases.append(dict(scenario_id=f'final_{layout}_c{cell:02d}_r{rep:03d}',
                    split='final_test',layout=layout,cell=cell,robots=robots,spawn_window_s=window,
                    demand_profile=profile,dominant_start=start,dominant_goal=goal,seed=seed,
                    missions_sha256=fingerprint,missions=missions))
    final=dict(protocol='agv_independent_final_v1_20261104',sample_counts=COUNTS,
        sample_planning='0.5 percentage points of development layout FCFS mean; normal approximation, not power guarantee',
        primary_family='15 = 3 layouts * (basic/bo/pso vs fcfs, bo/pso vs basic)',
        primary_estimand='Mean penalized makespan difference across scenarios, averaging the five fixed fitted optimizer seeds first',
        confirmatory_method='Two-sided stratified paired Welch-Satterthwaite t approximation; Holm FWER .05 over 15 tests',
        secondary_scope='Strict FCFS, fixed4 and warmstart comparisons descriptive; no uncorrected confirmatory claims',
        scenario_seed_recipe='731000000 + layout_index*1000000 + cell*10000 + replicate; collision stops without reselection',
        physical_config=m['physical_config'],dt_s=m['dt_s'],max_time_s=m['max_time_s'],
        source_sha256=runtime_hashes(),runner_sha256=sha(__file__),
        analyzer_sha256=sha(ROOT/'scripts/analyze_agv_final_test.py'),
        final_requirements_sha256=sha(ROOT/'requirements-final.txt'),
        exclusions_sha256=sha(EXCLUSIONS),frozen_input_hashes=hashes,policies=registry,
        expected_scenarios=2052,expected_episodes=38988,scenarios=cases,
        limitations=['Fixed fitted policies; does not establish performance across fresh training data/seeds.',
                    'Fixed 12-cell mixture and fixed routes; no global deadlock-freedom guarantee.'])
    out.mkdir(parents=True,exist_ok=False)
    atomic_durable(out/'manifest.json',final)
    atomic_durable(out/'lock.json',dict(manifest_sha256=sha(out/'manifest.json'),
        protocol_frozen_before_evaluation=True,python=sys.version,cpu=platform.machine()))
    print('FINAL_PROTOCOL_FROZEN',sha(out/'manifest.json'),'scenarios',len(cases),'episodes',38988,flush=True)

def load_locked(out=OUT):
    m=json.loads((out/'manifest.json').read_text());lock=json.loads((out/'lock.json').read_text())
    if lock['manifest_sha256']!=sha(out/'manifest.json') or m['runner_sha256']!=sha(__file__):
        raise ValueError('Frozen final protocol or runner changed')
    if m['source_sha256']!=runtime_hashes() or m['exclusions_sha256']!=sha(EXCLUSIONS):
        raise ValueError('Frozen source/exclusions changed')
    if m['analyzer_sha256']!=sha(ROOT/'scripts/analyze_agv_final_test.py') or m['final_requirements_sha256']!=sha(ROOT/'requirements-final.txt'):
        raise ValueError('Frozen analysis or final requirements changed')
    for p,h in m['frozen_input_hashes'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen fit/input changed: '+p)
    if m['sample_counts']!=COUNTS or len(m['scenarios'])!=2052:
        raise ValueError('Final sample count changed')
    return m,lock['manifest_sha256']

def episode_path(out,layout,label,case):
    return out/'episodes'/layout/label/(case['scenario_id']+'.json')

def collect(out,m,signature,require_complete=False):
    rows=[]
    for layout in COUNTS:
        cases=[s for s in m['scenarios'] if s['layout']==layout]
        for policy in m['policies'][layout]:
            for case in cases:
                path=episode_path(out,layout,policy['label'],case)
                if not path.exists():
                    if require_complete:raise ValueError('Missing final episode: '+str(path))
                    continue
                record=json.loads(path.read_text())
                if record['manifest_sha256']!=signature or record['policy']!=policy:
                    raise ValueError('Saved final episode belongs to another experiment')
                validate_row(record['row'],case,m['max_time_s'])
                rows.append(dict(label=policy['label'],**record['row']))
    return rows

def worker(task):
    out,layout,label=task;out=Path(out);m,signature=load_locked(out)
    policy=next(p for p in m['policies'][layout] if p['label']==label)
    cases=[s for s in m['scenarios'] if s['layout']==layout];engine=build_layout(layout)
    for i,case in enumerate(cases):
        path=episode_path(out,layout,label,case)
        if path.exists():
            d=json.loads(path.read_text())
            if d['manifest_sha256']!=signature or d['policy']!=policy:raise ValueError('Episode resume mismatch')
            validate_row(d['row'],case,m['max_time_s'])
        else:
            row=run_case(engine,case,m,policy['policy'],policy['vector'])
            validate_row(row,case,m['max_time_s'])
            atomic_durable(path,dict(manifest_sha256=signature,policy=policy,row=row))
        if (i+1)%10==0 or i+1==len(cases):
            print('FINAL_PROGRESS',layout,label,i+1,'/',len(cases),flush=True)
    return layout,label

def evaluate(out=OUT,workers=2):
    m,signature=load_locked(out)
    lock=(out/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(phase='final_test',complete=False,manifest_sha256=signature,
                expected_episodes=38988,completed_jobs=[])
    atomic_durable(out/'status.json',status)
    try:
        tasks=[(str(out),l,p['label']) for l in COUNTS for p in m['policies'][l]]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(worker,t) for t in tasks]
            for f in concurrent.futures.as_completed(futures):
                status['completed_jobs'].append(f.result())
                atomic_durable(out/'status.json',status)
                print('FINAL_JOB_COMPLETE',status['completed_jobs'][-1],len(status['completed_jobs']),'/ 57',flush=True)
        rows=collect(out,m,signature,True)
        if len(rows)!=38988:raise ValueError('Final episode total mismatch')
        status.update(phase='final_evaluation_complete',complete=True,completed_episodes=len(rows),
            incomplete_episodes=sum(r['status']!='completed' for r in rows),
            safety_failures=sum(not r['safety_checks_passed'] for r in rows))
        atomic_durable(out/'status.json',status)
        print('FINAL_EVALUATION_COMPLETE',len(rows),flush=True)
    except BaseException as error:
        status.update(phase='failed',complete=False,error=repr(error))
        atomic_durable(out/'status.json',status)
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['prepare','run','status'],required=True)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if sys.version_info[:2]!=(3,12):raise ValueError('Use the .agv312 Python environment')
    if not 1<=args.workers<=2:raise ValueError('Use 1..2 workers on the selected 2-core machine')
    if args.phase=='prepare':prepare(args.output)
    elif args.phase=='run':evaluate(args.output,args.workers)
    else:
        m,sig=load_locked(args.output);rows=collect(args.output,m,sig)
        print(json.dumps(dict(saved_episodes=len(rows),expected_episodes=38988,
            remaining_episodes=38988-len(rows),status=json.loads((args.output/'status.json').read_text())
            if (args.output/'status.json').exists() else 'not started'),indent=2))

if __name__=='__main__':main()
