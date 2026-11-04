"""Freeze marginally direction-balanced train/development and common initial bank.

No policy outcomes are used and no final-test scenarios are generated.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

from scripts.prepare_agv_capstone_v2 import ROOT, COUNTS, FLAGS, build_layout, generate, fingerprint


def unit(values):
    length = math.sqrt(sum(v*v for v in values))
    return [v/length for v in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Preserve prior manifests: choose a new output directory')
    cases, seen, balance = [], set(), []
    for li, name in enumerate(COUNTS):
        layout = build_layout(name)
        gates = len([n for n in layout.stations if n.endswith(('_START', '_IN'))])
        for si, (split, types, target) in enumerate([('train', range(1,5), 10),
                                                   ('development', range(1,9), 5)]):
            repetitions = math.ceil(target/gates)*gates
            for ni, count in enumerate(COUNTS[name]):
                for case in types:
                    group = []
                    for rep in range(repetitions):
                        seed = 66000000+li*1000000+si*100000+ni*10000+case*100+rep
                        item = generate(layout, name, count, case, rep, seed)
                        identity = (name, item['missions_sha256'])
                        if identity in seen:
                            raise ValueError('Duplicate scenario content')
                        seen.add(identity)
                        item.update(split=split, scenario_id=f'{split}_{name}_{count}_S{case}_{rep:02}')
                        cases.append(item)
                        group.append(item)
                    burst, approach, exit_skew = FLAGS[case]
                    row = dict(layout=name, split=split, robots=count, scenario_case=case,
                               repetitions=repetitions)
                    for field, active in [('dominant_start',approach),('dominant_goal',exit_skew)]:
                        counts = Counter(s[field] for s in group) if active else Counter()
                        if counts and (len(counts)!=gates or len(set(counts.values()))!=1):
                            raise ValueError('Direction marginal imbalance')
                        row[field] = dict(counts)
                    balance.append(row)
    # Fixed anchors + seeded random unit vectors, without consulting any outcomes.
    rng = random.Random(20261105)
    bank = [[1.,0.,0.,0.],unit([1.,.8,1.2,1.5])]
    bank.extend(unit([rng.uniform(-1,1) for _ in range(4)]) for _ in range(4))
    assert len({tuple(v) for v in bank}) == 6
    sources = ['scripts/prepare_agv_capstone_v2_balanced.py', 'scripts/prepare_agv_capstone_v2.py',
               'rware/agv_capstone_v2_simulation.py', 'rware/agv_multicross_policy_simulation.py',
               'rware/agv_physical_time.py', 'rware/agv_layouts.py', 'rware/agv_multicross_layout.py',
               'scripts/run_multicross_search.py']
    manifest = dict(protocol='agv_capstone_v2_scenario_draft', direction_marginals_balanced=True,
                    pair_coverage='Inherited offset pairing; NOT exhaustive dominant inlet/outlet combinations',
                    final_test_generated=False, training_started=False, counts=COUNTS,
                    dt_s=.01,max_time_s=1200., methods=['basic','bo','pso'],
                    optimizer_seeds=[101,202,303,404,505],search_budget=72,
                    shared_initial_vectors=bank,
                    initialization='six predeclared vectors, no outcome-based calibration or legacy learned weights',
                    objective='incomplete scenarios*1201 + mean makespan capped at 1200; each scenario counts once',
                    source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                    balance=balance, scenarios=cases,
                    deviations=['Balanced non-dominant remainder; exact OD quota matching',
                                'Rounded per-cell repetition counts upward for equal directional marginals'],
                    required_before_training='complete FCFS engineering audit of this exact manifest; source identity match',
                    final_analysis='Equal type/count weights; optimizer-seed repeats are not independent scenarios')
    args.output_dir.mkdir(parents=True)
    output = args.output_dir/'scenario_manifest.json'
    output.write_text(json.dumps(manifest,indent=2)+'\n')
    for name in COUNTS:
        print(name, {s:sum(c['layout']==name and c['split']==s for c in cases)
                     for s in ('train','development')})
    print('Direction marginals balanced; same six initial vectors; 72 evaluations; five seeds.')
    print('No training or final test executed. Manifest:', output)


if __name__=='__main__':
    main()
