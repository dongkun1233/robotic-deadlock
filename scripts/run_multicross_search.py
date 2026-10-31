"""Equal-evaluation random/BO/PSO search on fixed training scenarios only.

Example: python scripts/run_multicross_search.py --manifest MANIFEST --output-dir DIR
Validation and test simulations are deliberately not exposed by this command.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rware import agv_multicross_policy_simulation as sim
from rware import agv_multicross_layout as layout_module
from rware.agv_multicross_experiments import digest, plan

FEATURES = sim.FEATURES


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError('Zero weight vector is not a valid search candidate')
    return vector / norm


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def kernel(a, b, length=0.45):
    delta = (a[:,None,:]-b[None,:,:])/length
    return np.exp(-0.5*np.sum(delta*delta, axis=2))


def gp_lcb_candidate(vectors, objectives, rng, count=256):
    dimension = len(vectors[0])
    global_count = int(count*.65)
    raw = rng.uniform(-1,1,(global_count,dimension))
    best = vectors[int(np.argmin(objectives))]
    local = np.clip(best+rng.normal(0,.18,(count-global_count,dimension)),-1,1)
    candidates = np.array([unit(v) for v in np.vstack([raw,local])])
    x, test = (np.array(vectors)+1)/2, (candidates+1)/2
    y = np.asarray(objectives, dtype=float)
    mean, scale = y.mean(), max(float(y.std()),1e-8)
    normalized = (y-mean)/scale
    covariance = kernel(x,x)
    chol = None
    for noise in (1e-6,1e-5,1e-4,1e-3):
        try:
            chol = np.linalg.cholesky(covariance+noise*np.eye(len(x)))
            break
        except np.linalg.LinAlgError:
            pass
    if chol is None:
        raise RuntimeError('GP covariance factorization failed')
    cross = kernel(test,x)
    alpha = np.linalg.solve(chol.T,np.linalg.solve(chol,normalized))
    mu = cross@alpha*scale+mean
    solved = np.linalg.solve(chol,cross.T)
    sigma = np.sqrt(np.maximum(1-np.sum(solved*solved,axis=0),1e-12))*scale
    return candidates[int(np.argmin(mu-1.96*sigma))]


def search(algorithm, seed, budget, initial, evaluate):
    dimension = initial.shape[1]
    rng = np.random.default_rng(seed+{'basic':10000,'bo':20000,'pso':30000}[algorithm])
    vectors, objectives = [], []
    for vector in initial:
        vectors.append(vector.copy())
        objectives.append(evaluate(vector,'shared_initial'))
    if algorithm in ('basic','bo'):
        while len(vectors)<budget:
            vector = unit(rng.uniform(-1,1,dimension)) if algorithm=='basic' else gp_lcb_candidate(vectors,objectives,rng)
            objectives.append(evaluate(vector,'random_search' if algorithm=='basic' else 'gp_lcb'))
            vectors.append(vector.copy())
        return
    positions = initial.copy()
    velocities = rng.uniform(-.2,.2,positions.shape)
    personal = positions.copy()
    personal_y = np.array(objectives,dtype=float)
    global_best = personal[int(np.argmin(personal_y))].copy()
    completed = len(initial)
    # Synchronous swarm update; every particle in a generation sees the same gbest.
    while completed<budget:
        inertia = .7-.3*(completed-len(initial))/max(1,budget-len(initial))
        velocities = np.clip(inertia*velocities+1.4*rng.random(positions.shape)*(personal-positions)
            +1.4*rng.random(positions.shape)*(global_best-positions),-.35,.35)
        proposed = np.array([unit(v) for v in np.clip(positions+velocities,-1,1)])
        for particle in range(len(initial)):
            if completed>=budget:
                break
            value = evaluate(proposed[particle],'pso_update')
            positions[particle] = proposed[particle]
            if value<personal_y[particle]:
                personal_y[particle] = value
                personal[particle] = proposed[particle].copy()
            completed += 1
        global_best = personal[int(np.argmin(personal_y))].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--budget',type=int,default=24)
    parser.add_argument('--initial-count',type=int,default=6)
    parser.add_argument('--seeds',default='101,202,303')
    parser.add_argument('--models',default='full')
    parser.add_argument('--simulator',choices=['v1','v2'],default='v1')
    args = parser.parse_args()
    global sim
    original_sim = sim
    if args.simulator=='v2':
        from rware import agv_multicross_policy_v2
        sim = agv_multicross_policy_v2
    seeds = [int(x) for x in args.seeds.split(',')]
    model_names = args.models.split(',')
    model_features = {'full':FEATURES}
    model_features.update({f'without_{f}':tuple(k for k in FEATURES if k!=f) for f in FEATURES})
    if not seeds or len(set(seeds))!=len(seeds) or any(m not in model_features for m in model_names) or len(set(model_names))!=len(model_names):
        raise ValueError('Invalid seeds/models')
    if args.initial_count<2 or args.budget<args.initial_count:
        raise ValueError('budget >= initial-count >= 2 required')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    for module in (layout_module,original_sim):
        name = Path(module.__file__).name
        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if manifest['source_sha256'].get(name)!=actual:
            raise ValueError(f'Source file differs from scenario preparation: {name}')
    scenarios = [s for s in manifest['scenarios'] if s['split']=='train']
    if len(scenarios)!=manifest['split_counts']['train']:
        raise ValueError('Wrong training scenario count')
    layout = layout_module.build_multicross_layout()
    for s in scenarios:
        if digest(plan(layout,s['robots'],s['seed'],s['spawn_window']))!=s['missions_sha256']:
            raise ValueError('Training replay definition mismatch')
    hashes = [s['missions_sha256'] for s in manifest['scenarios']]
    if len(hashes)!=len(set(hashes)):
        raise ValueError('Duplicate scenario definitions')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'raw').mkdir(exist_ok=True)
    config = {'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'source_sha256':manifest['source_sha256'], 'search_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'budget':args.budget,'initial_count':args.initial_count,'seeds':seeds,'models':model_names,
        'split':'train','training_scenarios':len(scenarios),'capacity':manifest['capacity'],
        'max_steps':manifest['max_steps'],'objective':'incomplete_scenarios*(max_steps+1)+mean_total_steps',
        'versions':{'python':sys.version.split()[0],'numpy':np.__version__}}
    config['simulator']=args.simulator
    config['simulator_source_sha256']=hashlib.sha256(Path(sim.__file__).read_bytes()).hexdigest()
    checkpoint = args.output_dir/'checkpoint.json'
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state['config']!=config:
            raise ValueError('Resume configuration does not match checkpoint')
    else:
        state={'config':config,'trials':[],'baseline':None}
        atomic_json(checkpoint,state)
    started = time.perf_counter()
    algorithms = ('basic','bo','pso')
    def simulate(weights, policy):
        rows=[]
        for s in scenarios:
            result=sim.run(robots=s['robots'],seed=s['seed'],spawn_window=s['spawn_window'],
                policy=policy,weights=weights,capacity=manifest['capacity'],max_steps=manifest['max_steps'])
            replay=[{k:r[k] for k in ('id','start','goal','planned','path')} for r in result['robot_results']]
            if digest(replay)!=s['missions_sha256']:
                raise ValueError('Actual simulator mission replay mismatch')
            rows.append({'scenario_id':s['scenario_id'],'case':s['case'],'robots':s['robots'],
                'completed':result['completed'],'timed_out':result['timed_out'],'total_steps':result['total_steps'],
                'mean_wait':result['mean_wait_steps_all_agents'],'max_wait':result['max_wait_steps_all_agents'],
                'mean_spawn_delay':result['mean_spawn_delay']})
        return rows
    if state['baseline'] is None:
        rows=simulate(None,'fcfs')
        write_csv(args.output_dir/'fcfs_training_raw.csv',rows)
        state['baseline']={'mean_total_steps':float(np.mean([r['total_steps'] for r in rows])),
            'incomplete':sum(r['timed_out'] for r in rows)}
        atomic_json(checkpoint,state)
    for model in model_names:
        active=model_features[model]
        for seed in seeds:
            rng=np.random.default_rng(seed)
            initial=np.array([unit(v) for v in rng.uniform(-1,1,(args.initial_count,len(active)))])
            for algorithm in algorithms:
                print(f'{model} seed={seed} algorithm={algorithm}',flush=True)
                history=[t for t in state['trials'] if (t['model'],t['seed'],t['algorithm'])==(model,seed,algorithm)]
                history.sort(key=lambda t:t['evaluation'])
                evaluation=0
                def evaluate(vector,source):
                    nonlocal evaluation
                    evaluation+=1
                    weights={f:0.0 for f in FEATURES}
                    weights.update({f:float(v) for f,v in zip(active,vector)})
                    if evaluation<=len(history):
                        old=history[evaluation-1]
                        if old['evaluation']!=evaluation or not all(abs(old['weights'][f]-weights[f])<1e-12 for f in FEATURES):
                            raise ValueError('Deterministic resume replay differs')
                        return old['objective']
                    t=time.perf_counter()
                    rows=simulate(weights,'heuristic')
                    incomplete=sum(r['timed_out'] for r in rows)
                    average=float(np.mean([r['total_steps'] for r in rows]))
                    value=incomplete*(manifest['max_steps']+1)+average
                    raw_name=f'{model}_{seed}_{algorithm}_{evaluation:03d}.csv'
                    write_csv(args.output_dir/'raw'/raw_name,rows)
                    record={'model':model,'seed':seed,'algorithm':algorithm,'evaluation':evaluation,
                        'source':source,'weights':weights,'objective':value,'incomplete':incomplete,
                        'mean_total_steps':average,'mean_wait':float(np.mean([r['mean_wait'] for r in rows])),
                        'worst_agent_wait':max(r['max_wait'] for r in rows),
                        'evaluation_seconds':time.perf_counter()-t,'raw_file':f'raw/{raw_name}'}
                    state['trials'].append(record)
                    atomic_json(checkpoint,state)
                    print(f'  {evaluation}/{args.budget} mean={average:.3f} incomplete={incomplete}',flush=True)
                    return value
                search(algorithm,seed,args.budget,initial,evaluate)
                if evaluation!=args.budget:
                    raise RuntimeError('Optimizer evaluation budget mismatch')
    best=[]
    for model in model_names:
        for seed in seeds:
            for algorithm in algorithms:
                history=[t for t in state['trials'] if (t['model'],t['seed'],t['algorithm'])==(model,seed,algorithm)]
                chosen=min(history,key=lambda t:(t['objective'],t['evaluation']))
                best.append({**{k:v for k,v in chosen.items() if k!='weights'},'weights_json':json.dumps(chosen['weights'],sort_keys=True)})
    trial_rows=[{**{k:v for k,v in t.items() if k!='weights'},'weights_json':json.dumps(t['weights'],sort_keys=True)} for t in state['trials']]
    write_csv(args.output_dir/'train_trials.csv',trial_rows)
    write_csv(args.output_dir/'best_training_weights.csv',best)
    atomic_json(args.output_dir/'run_complete.json',{'completed':True,'trials':len(state['trials']),
        'simulation_runs':len(state['trials'])*len(scenarios),'baseline':state['baseline'],
        'split':'train','validation_evaluated':False,'test_evaluated':False})
    print('\nTRAINING COMPLETE',state['baseline'],flush=True)
    for row in best:
        print(row['model'],row['seed'],row['algorithm'],round(row['mean_total_steps'],3),row['incomplete'])
    print('This invocation seconds:',round(time.perf_counter()-started,2))
    print('Validation and test were not evaluated. Results:',args.output_dir.resolve())


if __name__=='__main__':
    main()
