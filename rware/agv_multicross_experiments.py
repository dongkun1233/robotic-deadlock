"""Prepare fixed scenario splits and check replay without evaluating held-out splits."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from rware import agv_multicross_layout as layout_module
from rware import agv_multicross_policy_simulation as policy_module


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def plan(layout, robots, seed, spawn_window):
    rng = random.Random(seed)
    starts = sorted(n for n in layout.stations if n.endswith('_IN'))
    goals = sorted(n for n in layout.stations if n.endswith('_OUT'))
    missions = []
    for robot_id in range(robots):
        start = rng.choice(starts)
        goal = rng.choice([g for g in goals if g.rsplit('_',1)[0] != start.rsplit('_',1)[0]])
        missions.append({'id': robot_id, 'start': start, 'goal': goal,
            'planned': rng.randint(0, spawn_window),
            'path': policy_module.shortest_path(layout.graph, start, goal)})
    return missions


def prepare(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ('scenario_manifest.json','scenario_index.csv','replay_smoke.json'):
        if (output_dir / filename).exists():
            raise FileExistsError(f'Refusing to overwrite {output_dir / filename}')
    layout = layout_module.build_multicross_layout()
    split_sizes = {'train':20, 'validation':10, 'test':20}
    cases = [(24,20),(24,0),(48,20),(48,0)]
    scenarios = []
    seen_hashes, seen_seeds = set(), set()
    for split_index, (split, count) in enumerate(split_sizes.items()):
        for case_index, (robots, window) in enumerate(cases):
            for repeat in range(count):
                seed = 700000 + split_index*100000 + case_index*1000 + repeat
                missions = plan(layout, robots, seed, window)
                content_hash = digest(missions)
                if content_hash in seen_hashes or seed in seen_seeds:
                    raise ValueError('Duplicate scenario content or seed')
                seen_hashes.add(content_hash)
                seen_seeds.add(seed)
                scenarios.append({'scenario_id':f'{split}_c{case_index+1}_{repeat:03d}',
                    'split':split, 'case':case_index+1, 'robots':robots,
                    'spawn_window':window, 'seed':seed, 'missions_sha256':content_hash,
                    'missions':missions})
    source_hashes = {
        Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (layout_module,policy_module)
    }
    manifest = {'schema_version':1, 'layout':layout.name,
        'source_sha256':source_hashes, 'capacity':2,'max_steps':2000,
        'split_counts':{split:count*len(cases) for split,count in split_sizes.items()},
        'evaluation_rule':'Use training objectives to optimize weights, validation to select settings; keep test unevaluated until the protocol is fixed.',
        'normalization':'Per-decision max scaling for waiting/exit/path; maneuver left=0,straight=.5,right=1.',
        'scenario_replay':'Current simulator replays from seed. Stored mission paths and plans must match before any new runner is used.',
        'scenarios':scenarios}
    # Save the complete explicit scenario definitions before any smoke simulation.
    (output_dir/'scenario_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    fields = ['scenario_id','split','case','robots','spawn_window','seed','missions_sha256']
    with (output_dir/'scenario_index.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:s[k] for k in fields} for s in scenarios)
    smoke = []
    for case in range(1,5):
        # Only training scenarios are used here; validation/test are not evaluated.
        scenario = next(s for s in scenarios if s['split']=='train' and s['case']==case)
        for policy in ('fcfs','heuristic'):
            result=policy_module.run(robots=scenario['robots'],seed=scenario['seed'],
                spawn_window=scenario['spawn_window'],policy=policy,capacity=2,max_steps=2000)
            replay=[{k:r[k] for k in ('id','start','goal','planned','path')} for r in result['robot_results']]
            if digest(replay)!=scenario['missions_sha256']:
                raise ValueError('Scenario replay mismatch')
            smoke.append({'scenario_id':scenario['scenario_id'],'policy':policy,
                'completed':result['completed'],'robots':scenario['robots'],
                'total_steps':result['total_steps'],'timed_out':result['timed_out'],
                'competitive_decisions':result['competitive_decisions'],
                'mean_wait_steps_all_agents':result['mean_wait_steps_all_agents'],
                'max_wait_steps_all_agents':result['max_wait_steps_all_agents'],
                'scenario_replay_verified':True})
    (output_dir/'replay_smoke.json').write_text(json.dumps(smoke,indent=2),encoding='utf-8')
    print('[Scenario preparation]')
    print('Output:',output_dir.resolve())
    print('Split counts:',manifest['split_counts'])
    print('Unique scenario seeds:',len(seen_seeds))
    print('Unique mission-plan hashes:',len(seen_hashes))
    print('\n[Training-only replay smoke check]')
    for row in smoke:
        print(f"{row['scenario_id']} / {row['policy']}: completed={row['completed']}/{row['robots']}, steps={row['total_steps']}, timeout={row['timed_out']}, replay=OK")
    print('\nValidation and test splits have NOT been simulated.')
    print('Heuristic weights are provisional; these are not BO/PSO results.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    prepare(args.output_dir)
