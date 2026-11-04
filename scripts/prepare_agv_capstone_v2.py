"""Create train/development only, with exact OD quotas and legacy-shaped releases."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preflight_agv_capstone_v2 import inspect_layout
from rware.agv_capstone_v2_simulation import build_layout
from rware.agv_multicross_policy_simulation import shortest_path

COUNTS = {'single': [20, 24, 28], 'double': [24, 30, 36], 'multi': [20, 30, 40]}
FLAGS = {1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0), 4: (0, 0, 1),
         5: (1, 1, 1), 6: (1, 1, 0), 7: (1, 0, 1), 8: (0, 1, 1)}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def quota(nodes, total, dominant, skew, rng):
    if not skew:
        assert total % len(nodes) == 0
        return dict.fromkeys(nodes, total // len(nodes))
    upper = total//2+4 if len(nodes) == 4 else round(total*2/3)
    upper = min(upper, total-len(nodes)+1)
    count = rng.randint(total//2, upper)
    others = [n for n in nodes if n != dominant]
    # Balanced remainder avoids an unintended second dominant inlet/outlet.
    rng.shuffle(others)
    result = {dominant: count}
    remainder = total-count
    result.update({n: remainder//len(others)+(i < remainder % len(others))
                   for i, n in enumerate(others)})
    assert sum(result.values()) == total and min(result.values()) >= 1
    return result


def generate(layout, name, count, case, rep, seed):
    rng = random.Random(seed)
    starts = sorted(n for n in layout.stations if n.endswith(('_START', '_IN')))
    goals = sorted(n for n in layout.stations if n.endswith(('_EXIT', '_OUT')))
    burst, approach, exit_skew = FLAGS[case]
    dominant_start = starts[rep % len(starts)]
    dominant_goal = goals[(rep+1) % len(goals)]
    left = quota(starts, count, dominant_start, approach, rng)
    right = quota(goals, count, dominant_goal, exit_skew, rng)
    entries = [n for n in starts for _ in range(left[n])]
    exits = [n for n in goals for _ in range(right[n])]
    rng.shuffle(entries)
    rng.shuffle(exits)
    paths = {}
    for start in starts:
        for goal in goals:
            if start.rsplit('_', 1)[0] == goal.rsplit('_', 1)[0]:
                continue
            paths[start, goal] = shortest_path(layout.graph, start, goal)
    # Slot-level bipartite matching preserves BOTH exact marginals. Impossible
    # combinations fail explicitly; no retries that select favorable traffic.
    owner = {}
    def augment(i, seen):
        for j, goal in enumerate(exits):
            if j in seen or (entries[i], goal) not in paths:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False
    for i in range(count):
        if not augment(i, set()):
            raise ValueError(f'Infeasible exact OD quotas: {name}/{count}/S{case}/{rep}')
    assignment = {i: exits[j] for j, i in owner.items()}
    early = set(rng.sample(range(count), count//2)) if burst else set()
    releases = {}
    for start in starts:
        indices = [i for i, node in enumerate(entries) if node == start]
        values = sorted((rng.randint(0, 2) if i in early else rng.randint(3, 8))*1.875
                        for i in indices) if burst else [k*7.5 for k in range(len(indices))]
        releases.update(zip(indices, values))
    missions = [dict(id=i, path=paths[entries[i], assignment[i]], planned_s=releases[i])
                for i in range(count)]
    assert Counter(m['path'][0] for m in missions) == Counter(left)
    assert Counter(m['path'][-1] for m in missions) == Counter(right)
    return dict(layout=name, robots=count, scenario_case=case, replicate=rep, seed=seed,
                start_quota=left, exit_quota=right, dominant_start=dominant_start,
                dominant_goal=dominant_goal, missions=missions,
                missions_sha256=fingerprint(missions), last_release_s=max(releases.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('New directory required')
    scenarios, seen = [], set()
    for li, name in enumerate(COUNTS):
        layout = build_layout(name)
        audit = inspect_layout(name)
        assert not audit['invalid_pairs'] and audit['balanced_allocation_possible']
        for si, (split, cases, reps) in enumerate([('train', range(1, 5), 10),
                                                   ('development', range(1, 9), 5)]):
            for ni, count in enumerate(COUNTS[name]):
                for case in cases:
                    for rep in range(reps):
                        seed = 56000000+li*1000000+si*100000+ni*10000+case*100+rep
                        item = generate(layout, name, count, case, rep, seed)
                        identity = (name, item['missions_sha256'])
                        if identity in seen:
                            raise ValueError('Duplicate scenario content')
                        seen.add(identity)
                        item.update(split=split, scenario_id=f'{split}_{name}_{count}_S{case}_{rep:02}')
                        scenarios.append(item)
    sources = ['scripts/prepare_agv_capstone_v2.py', 'rware/agv_capstone_v2_simulation.py',
               'rware/agv_multicross_policy_simulation.py', 'rware/agv_physical_time.py',
               'rware/agv_layouts.py', 'rware/agv_multicross_layout.py']
    manifest = dict(protocol='agv_capstone_v2_scenario_draft', final_test_generated=False,
                    training_started=False, source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                                           for p in sources},
                    deviations=['Remainder of skew quotas is balanced, unlike legacy random compositions',
                                'Exact OD quota matching; no per-vehicle independent goal sampling'],
                    unbalanced_direction_repeats='10/5 replicates cannot evenly cover all dominant OD pairs; expand or block before confirmation',
                    counts=COUNTS, scenarios=scenarios)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'scenario_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for name in COUNTS:
        print(name, {split: sum(x['layout']==name and x['split']==split for x in scenarios)
                     for split in ('train', 'development')})
    print('DRAFT ONLY: no optimizer or final test was executed')


if __name__ == '__main__':
    main()
