"""Paired reference/idle-skip equivalence and timing; no optimizer/final test."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

from scripts.prepare_agv_capstone_v2 import ROOT, build_layout
from rware.agv_capstone_v2_simulation import run as reference
from rware.agv_capstone_v2_fast_simulation import run as fast
from rware.agv_multicross_policy_simulation import FEATURES


def compare(a,b,where='result'):
    if isinstance(a,dict):
        assert set(a)==set(b),where
        for key in a:
            if key!='simulation_ticks':compare(a[key],b[key],where+'.'+str(key))
    elif isinstance(a,list):
        assert len(a)==len(b),(where,len(a),len(b))
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{where}[{i}]')
    elif isinstance(a,float):
        assert isinstance(b,(float,int)) and math.isclose(a,b,rel_tol=0,abs_tol=1e-7),(where,a,b)
    else:
        assert a==b,(where,a,b)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--per-cell',type=int,default=1)
    parser.add_argument('--profile',choices=['fcfs','mixed'],default='mixed')
    args=parser.parse_args()
    if args.per_cell<1 or args.output_dir.exists():
        raise ValueError('Positive repetitions and a new output directory required')
    data=json.loads(args.manifest.read_text())
    assert data.get('direction_marginals_balanced') and not data.get('final_test_generated')
    for name,expected in data['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    args.output_dir.mkdir(parents=True)
    groups=defaultdict(list)
    for case in data['scenarios']:
        if case['split']=='development':
            groups[case['layout'],case['robots'],case['scenario_case']].append(case)
    rng=random.Random(8252026)
    rows=[]
    layouts={name:build_layout(name) for name in ('single','double','multi')}
    for key,cases in sorted(groups.items()):
        for case in cases[:args.per_cell]:
            vector={f:rng.uniform(-1,1) for f in FEATURES}
            configs=[dict(policy='fcfs')]
            if args.profile=='mixed':
                configs += [dict(policy='strict_fcfs'),dict(policy='heuristic',weights=vector),
                            dict(policy='heuristic',weights=vector,wait_threshold_s=0.),
                            dict(policy='heuristic',weights=vector,wait_threshold_s=20.),
                            dict(policy='heuristic',weights=vector,wait_threshold_s=40.)]
            for config in configs:
                start=time.perf_counter()
                a=reference(layouts[key[0]],case['missions'],**config)
                ref_s=time.perf_counter()-start
                start=time.perf_counter()
                b=fast(layouts[key[0]],case['missions'],**config)
                fast_s=time.perf_counter()-start
                try:compare(a,b)
                except AssertionError as exc:
                    (args.output_dir/'failure.json').write_text(json.dumps(dict(
                        scenario_id=case['scenario_id'],config=config,error=str(exc)),indent=2))
                    raise
                rows.append(dict(scenario_id=case['scenario_id'],layout=key[0],config=config,
                                 status=a['status'],reference_s=ref_s,fast_s=fast_s,
                                 reference_ticks=a['simulation_ticks'],fast_ticks=b['simulation_ticks'],
                                 protected_admissions=a['fairness_protected_admissions'],
                                 overrides=a['fairness_override_count']))
            print('matched',case['scenario_id'],len(rows),'paired runs',flush=True)
    sources=['rware/agv_capstone_v2_simulation.py','rware/agv_capstone_v2_fast_simulation.py',
             'scripts/verify_agv_capstone_v2_fast.py']
    report=dict(scope='engineering equivalence; finite sampled cases, not a universal proof',
                manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                profile=args.profile,per_cell=args.per_cell,matched_pairs=len(rows),
                float_absolute_tolerance_s=1e-7,exact_fields='events, admissions, protection choices, status and counts',
                speedup=sum(r['reference_s'] for r in rows)/sum(r['fast_s'] for r in rows),rows=rows)
    (args.output_dir/'equivalence_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ALL MATCHED',len(rows),'pairs; measured speedup',round(report['speedup'],2),flush=True)


if __name__=='__main__':
    main()
