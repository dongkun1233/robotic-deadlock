"""Teammate-style fair-priority pilot: multi layout, paired overlay and retraining."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import numpy as np
from scripts.agv_timed_main_common import ROOT, atomic_durable, read_manifest, require_rows, save_csv, weights, key, config_from_manifest
from scripts.run_multicross_search import search
from rware.agv_multicross_experiments import digest
from rware.agv_timed_fair_simulation import run, build_layout

_STATE = {}
RUNTIME_FILES = ['rware/agv_timed_fair_simulation.py', 'rware/agv_timed_simulation.py',
                 'rware/agv_multicross_policy_simulation.py', 'rware/agv_physical_time.py',
                 'scripts/run_multicross_search.py', 'scripts/agv_timed_main_common.py',
                 'scripts/run_agv_timed_fair_pilot.py']


def source_hashes():
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in RUNTIME_FILES}


def initialize(physical_config, dt, horizon):
    _STATE.update(layout=build_layout('multi'), config=config_from_manifest({'physical_config':physical_config}), dt=dt, horizon=horizon)


def percentile95(values):
    values=sorted(values);p=.95*(len(values)-1);i=int(p);j=min(i+1,len(values)-1)
    return values[i]+(p-i)*(values[j]-values[i])


def episode(case, vector, threshold, policy):
    start=time.process_time();wall=time.perf_counter()
    result=run(_STATE['layout'],case['missions'],config=_STATE['config'],dt_s=_STATE['dt'],
               policy=policy,weights=None if vector is None else weights(vector),
               wait_threshold_s=threshold,max_time_s=_STATE['horizon'])
    details=result['robots_detail']
    if digest([dict(id=r['id'],path=r['path'],planned_s=r['planned_s']) for r in details])!=case['missions_sha256']:
        raise AssertionError('Missions differ')
    entry=[r['entry_wait_s'] for r in details];squares=sum(v*v for v in entry)
    return dict(execution_kind='simulated',scenario_id=case['scenario_id'],missions_sha256=case['missions_sha256'],
        cell=case['cell'],robots=case['robots'],status=result['status'],completed=result['completed'],
        makespan_s=result['makespan_s'],entry_mean_s=statistics.mean(entry),entry_p95_s=percentile95(entry),
        entry_max_s=max(entry),entry_over30=sum(t>=30 for t in entry),entry_over50=sum(t>=50 for t in entry),
        jain_entry_wait=sum(entry)**2/(len(entry)*squares) if squares else 1.,
        stationary_mean_s=result['mean_wait_s'],stationary_p95_s=result['p95_wait_s'],stationary_max_s=result['max_wait_s'],
        mean_spawn_delay_s=result['mean_spawn_delay_s'],protected_admissions=result['fairness_protected_admissions'],
        protected_competitions=result['fairness_protected_competitions'],overrides=result['fairness_override_count'],
        safety_checks_passed=result['safety_checks_passed'],waits_censored=result['waits_censored'],
        trajectory_sha256=digest({'robots':details,'admission_log':result['admission_log']}),
        episode_cpu_s=time.process_time()-start,episode_wall_s=time.perf_counter()-wall)


def objective(rows,horizon):
    return sum(r['status']!='completed' for r in rows)*(horizon+1)+statistics.mean(r['makespan_s'] for r in rows)


def candidate_rows(pool, base, protocol_sha, cases, vector, rule, threshold, policy='heuristic', split='train'):
    identity=digest(dict(vector=vector,rule=rule,policy=policy))
    path=base/split/'cache'/(identity+'.json')
    spec=dict(protocol_sha256=protocol_sha,vector=vector,rule=rule,threshold_s=threshold,policy=policy,split=split)
    record=json.loads(path.read_text()) if path.exists() else dict(config=spec,rows=[],status='running')
    if record['config']!=spec:raise ValueError('Cache inputs differ')
    require_rows(record['rows'],cases,complete=False)
    hit=len(record['rows'])==len(cases)
    plain_path=base/split/'cache'/(digest(dict(vector=vector,rule='plain',policy=policy))+'.json')
    plain_rows={}
    if threshold is not None and policy=='heuristic' and plain_path.exists():
        plain_record=json.loads(plain_path.read_text())
        if plain_record['config']['protocol_sha256']!=protocol_sha:raise ValueError('Plain reuse protocol differs')
        plain_rows={r['scenario_id']:r for r in plain_record['rows']}
    pending=[]
    for case in cases[len(record['rows']):]:
        old=plain_rows.get(case['scenario_id'])
        # Every local queue wait is bounded by this robot's cumulative entry wait.
        # If all cumulative waits are below T, protection cannot alter this trace.
        if old and old['status']=='completed' and old['entry_max_s']<threshold-1e-8:
            reused=dict(old,execution_kind='threshold_inactive_reuse',episode_cpu_s=0.,episode_wall_s=0.,
                        source_plain_cache=plain_path.name)
            pending.append(reused)
        else:
            pending.append(pool.submit(episode,case,vector,threshold,policy))
    # Persist in scenario order; cancellation/retry never silently changes pairing.
    for item in pending:
        record['rows'].append(item if isinstance(item,dict) else item.result());atomic_durable(path,record)
    require_rows(record['rows'],cases)
    record['status']='complete';atomic_durable(path,record)
    return record['rows'],identity,hit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('outputs/agv_timed_fair_pilot_v1/protocol.json'))
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();protocol=json.loads(args.protocol.read_text());base=args.protocol.parent
    if protocol['source_sha256']!=source_hashes():raise ValueError('Fair pilot source changed')
    protocol_sha=hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    manifest_path=ROOT/protocol['main_manifest'];main_manifest=read_manifest(manifest_path)
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=protocol['main_manifest_sha256']:raise ValueError('Data changed')
    cases={split:[s for s in main_manifest['scenarios'] if s['layout']=='multi' and s['split']=='main_'+split]
           for split in ['train','development']}
    if any(len(v)!=48 for v in cases.values()):raise ValueError('Expected 48 paired scenarios in each split')
    initial=np.array(protocol['common_initial_vectors']);best_fits=[]
    rules=[('plain',None),('fair20',20.),('fair40',40.)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
            initargs=(main_manifest['physical_config'],main_manifest['dt_s'],main_manifest['max_time_s'])) as pool:
        for rule,threshold in rules:
            for method in protocol['methods']:
                checkpoint=base/'train'/f'{rule}_{method}_{protocol["optimizer_seed"]}.json'
                state=json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(protocol_sha256=protocol_sha,trials=[])
                if state['protocol_sha256']!=protocol_sha:raise ValueError('Checkpoint inputs changed')
                number=0
                def evaluate(vector,source):
                    nonlocal number
                    number+=1;values=list(map(float,vector))
                    if number<=len(state['trials']):
                        trial=state['trials'][number-1]
                        if trial['vector']!=values or trial['evaluation']!=number:raise ValueError('Replay differs')
                        return trial['objective']
                    rows,cache,hit=candidate_rows(pool,base,protocol_sha,cases['train'],values,rule,threshold)
                    if rule=='plain' and number==1:
                        original=json.loads((ROOT/'outputs/agv_timed_main_v1/calibration/multi/fcfs.json').read_text())['rows']
                        if any(a['trajectory_sha256']!=b['trajectory_sha256'] for a,b in zip(original,rows)) or len(original)!=len(rows):
                            raise AssertionError('FCFS physical-model parity failed on expanded training data')
                    trial=dict(evaluation=number,vector=values,source=source,cache_key=cache,cache_hit=hit,
                               objective=objective(rows,main_manifest['max_time_s']),
                               incomplete=sum(r['status']!='completed' for r in rows))
                    state['trials'].append(trial);atomic_durable(checkpoint,state)
                    print('TRAIN',rule,method,number,round(trial['objective'],4),flush=True)
                    return trial['objective']
                search(method,protocol['optimizer_seed'],protocol['budget'],initial,evaluate)
                if len(state['trials'])!=protocol['budget']:raise ValueError('Budget differs')
                best=min(state['trials'],key=lambda x:(x['objective'],x['evaluation']))
                best_fits.append(dict(rule=rule,threshold_s=threshold,method=method,**best))
        atomic_durable(base/'best_fits.json',best_fits)
        policies=[('fcfs','fcfs',None,None),('strict_fcfs','strict_fcfs',None,None),
                  ('fixed4','heuristic',protocol['fixed4_vector'],None),
                  ('fixed4_fair20','heuristic',protocol['fixed4_vector'],20.),
                  ('fixed4_fair40','heuristic',protocol['fixed4_vector'],40.)]
        for fit in best_fits:
            policies.append((f"{fit['method']}_{fit['rule']}_retrained",'heuristic',fit['vector'],fit['threshold_s']))
            if fit['rule']=='plain':
                for threshold in [20.,40.]:
                    policies.append((f"{fit['method']}_sameweights_fair{int(threshold)}",'heuristic',fit['vector'],threshold))
        registry=[]
        for label,policy,vector,threshold in policies:
            rule='plain' if threshold is None else f'fair{int(threshold)}'
            rows,cache,_=candidate_rows(pool,base,protocol_sha,cases['development'],vector,rule,threshold,
                                       policy=policy,split='development')
            registry.append(dict(label=label,rule=rule,policy=policy,vector=vector,cache_key=cache,n=len(rows)))
            print('DEVELOPMENT',label,len(rows),flush=True)
        atomic_durable(base/'development_registry.json',registry)
    atomic_durable(base/'complete.json',dict(complete=True,train_fits=len(best_fits),
        evaluations_per_fit=protocol['budget'],optimizer_seed=protocol['optimizer_seed'],
        development_policy_rows=len(registry)*48,independent_development_scenarios=48,
        final_test_generated=False,final_test_evaluated=False))


if __name__=='__main__':main()
