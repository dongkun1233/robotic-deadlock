"""Locked, episode-resumable Random/BO/PSO fitting. Training split only.

Smoke runs are isolated from main fitting and never reusable as main results.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

from scripts.prepare_agv_capstone_v2 import ROOT, fingerprint, build_layout
from rware.agv_capstone_v2_simulation import run
from rware.agv_multicross_policy_simulation import FEATURES
from scripts.run_multicross_search import search
import numpy as np


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    raw = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
    if path.suffix == '.gz':
        raw = gzip.compress(raw, mtime=0)
    with temporary.open('wb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


class Pause(Exception):
    pass


def fit(layout_name, method, seed, cases, bank, budget, base, spec, stop_after):
    layout = build_layout(layout_name)
    job = base/'jobs'/f'{layout_name}_{seed}_{method}'
    cache = base/'cache'/layout_name
    job.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    job_spec = dict(**spec,layout=layout_name,method=method,seed=seed,
                    scenario_ids=[s['scenario_id'] for s in cases])
    checkpoint = job/'checkpoint.json'
    state = read(checkpoint) if checkpoint.exists() else dict(spec=job_spec,trials=[])
    if state['spec'] != job_spec:
        raise ValueError('Job checkpoint identity changed')
    if len(state['trials']) > budget:
        raise ValueError('Checkpoint exceeds budget')
    number = 0
    def evaluate(vector, source):
        nonlocal number
        number += 1
        values = [float(x) for x in vector]
        if number <= len(state['trials']):
            previous = state['trials'][number-1]
            if previous['evaluation'] != number or previous['vector'] != values or previous['source'] != source:
                raise ValueError('Optimizer replay differs from checkpoint')
            return previous['objective']
        if stop_after is not None and number > stop_after:
            raise Pause()
        key = fingerprint([float(x).hex() for x in values])
        path = cache/(key+'.json.gz')
        record = read(path) if path.exists() else dict(spec=spec,vector=values,rows=[])
        if record['spec'] != spec or record['vector'] != values or len(record['rows']) > len(cases):
            raise ValueError('Cache identity changed')
        for row, case in zip(record['rows'],cases):
            if row['scenario_id'] != case['scenario_id'] or row['missions_sha256'] != case['missions_sha256']:
                raise ValueError('Cache scenario identity/order changed')
        reused = len(record['rows'])
        for case in cases[reused:]:
            begin = time.perf_counter()
            result = run(layout,case['missions'],policy='heuristic',weights=dict(zip(FEATURES,values)),
                         dt_s=spec['dt_s'],max_time_s=spec['max_time_s'])
            if not result['safety_checks_passed']:
                raise AssertionError('Grid reservation check failed')
            raw = {k:result[k] for k in ('status','completed','makespan_s','mean_wait_s',
                    'p95_wait_s','max_wait_s','mean_spawn_delay_s','robots_detail','admission_log')}
            raw.update(scenario_id=case['scenario_id'],missions_sha256=case['missions_sha256'],
                       episode_wall_s=time.perf_counter()-begin)
            record['rows'].append(raw)
            save(path,record)
        failed = sum(r['status']!='completed' for r in record['rows'])
        objective = failed*(spec['max_time_s']+1)+statistics.mean(
            min(spec['max_time_s'],r['makespan_s']) for r in record['rows'])
        trial = dict(evaluation=number,vector=values,source=source,objective=objective,
                     incomplete_scenarios=failed,episodes_reused=reused,
                     episodes_new=len(cases)-reused,cache_key=key)
        state['trials'].append(trial)
        save(checkpoint,state)
        print(f'{layout_name}/{seed}/{method} {number}/{budget} objective={objective:.6f} '
              f'incomplete={failed} new={len(cases)-reused} reused={reused}',flush=True)
        return objective
    try:
        search(method,seed,budget,np.array(bank,dtype=float),evaluate)
    except Pause:
        return False
    if len(state['trials']) != budget or number != budget:
        raise AssertionError('Wrong evaluation budget')
    best = min(state['trials'],key=lambda r:(r['objective'],r['evaluation']))
    save(job/'best.json',dict(spec=job_spec,best=best,complete=True,
                             development_evaluated=False,final_test_evaluated=False))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--mode',choices=['smoke','main'],required=True)
    parser.add_argument('--layouts',default='single,double,multi')
    parser.add_argument('--methods',default='basic,bo,pso')
    parser.add_argument('--stop-after-evaluations',type=int)
    parser.add_argument('--smoke-scenarios',type=int,choices=[2,12],default=12)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not manifest.get('direction_marginals_balanced') or manifest.get('final_test_generated'):
        raise ValueError('Balanced train/development manifest required')
    for name, expected in manifest['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source mismatch: '+name)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if args.mode=='main':
        if args.stop_after_evaluations is not None:
            raise ValueError('Main jobs cannot use a smoke stop limit')
        audit = args.manifest.parent/'fcfs_audit'
        summary, checkpoint = read(audit/'summary.json'),read(audit/'checkpoint.json')
        if not summary.get('complete') or checkpoint['spec']['manifest_sha256'] != digest:
            raise ValueError('Complete engineering audit of this exact manifest required')
        audit_ids = [r['scenario_id'] for r in checkpoint['rows']]
        if audit_ids != [s['scenario_id'] for s in manifest['scenarios']]:
            raise ValueError('Audit does not cover exact manifest order')
        if any(r['status']!='completed' for r in checkpoint['rows']):
            raise ValueError('Review noncompleted engineering cases before main fitting')
    layouts, methods = args.layouts.split(','), args.methods.split(',')
    if not layouts or len(set(layouts))!=len(layouts) or any(x not in ('single','double','multi') for x in layouts):
        raise ValueError('Invalid layouts')
    if not methods or len(set(methods))!=len(methods) or any(x not in manifest['methods'] for x in methods):
        raise ValueError('Invalid methods')
    bank = manifest['shared_initial_vectors']
    if len(bank)!=6 or any(len(v)!=4 or not all(np.isfinite(v)) or abs(np.linalg.norm(v)-1)>1e-10 for v in bank):
        raise ValueError('Six finite unit vectors required')
    if args.stop_after_evaluations is not None and not 1<=args.stop_after_evaluations<=8:
        raise ValueError('Smoke stop must be between 1 and 8')
    budget = 8 if args.mode=='smoke' else manifest['search_budget']
    seeds = [101] if args.mode=='smoke' else manifest['optimizer_seeds']
    base = args.manifest.parent/('training_smoke_'+str(args.smoke_scenarios) if args.mode=='smoke' else 'training_main')
    base.mkdir(exist_ok=True)
    spec = dict(manifest_sha256=digest,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                mode=args.mode,budget=budget,dt_s=manifest['dt_s'],max_time_s=manifest['max_time_s'],
                numpy=np.__version__,python=platform.python_version(),initial=bank)
    with (base/'runner.lock').open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another training runner owns this output directory')
        planned = [(l,s,m) for l in layouts for s in seeds for m in methods]
        for name,seed,method in planned:
            cases = [s for s in manifest['scenarios'] if s['layout']==name and s['split']=='train']
            if not cases or any(fingerprint(s['missions'])!=s['missions_sha256'] for s in cases):
                raise ValueError('Invalid training cases')
            if args.mode=='smoke':
                groups = {}
                for s in cases:
                    groups.setdefault((s['robots'],s['scenario_case']),s)
                cases = list(groups.values())[:args.smoke_scenarios]
            fit(name,method,seed,cases,bank,budget,base,spec,args.stop_after_evaluations)
        jobs = [f'{l}_{s}_{m}' for l,s,m in planned]
        complete = [j for j in jobs if (base/'jobs'/j/'best.json').exists()]
        save(base/'status.json',dict(mode=args.mode,requested_jobs=jobs,completed_jobs=complete,
             complete=len(complete)==len(jobs),full_main_complete=(args.mode=='main' and
             len(complete)==45),final_test_started=False,development_evaluated=False))
        print(f'{args.mode}: {len(complete)}/{len(jobs)} requested jobs complete',flush=True)


if __name__=='__main__':
    main()
