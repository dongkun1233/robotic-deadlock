"""Engineering checks only; not a policy performance experiment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from scripts.prepare_agv_capstone_v2 import build_layout
from rware.agv_capstone_v2_simulation import run
from rware.agv_multicross_policy_simulation import FEATURES, shortest_path
from rware.agv_physical_time import AGVPhysicalTimeConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report_path = args.report or args.manifest.parent/'engineering_checks.json'
    if report_path.exists():
        raise FileExistsError('Preserve previous engineering report')
    manifest = json.loads(args.manifest.read_text())
    root = Path(__file__).resolve().parents[1]
    for name, expected_hash in manifest['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f'Frozen scenario source changed: {name}')
    rows = []
    for name in ('single', 'double', 'multi'):
        layout = build_layout(name)
        case = next(s for s in manifest['scenarios'] if s['layout']==name and s['scenario_case']==1)
        mission = dict(case['missions'][0], id=0, planned_s=0.)
        c = AGVPhysicalTimeConfig()
        path = mission['path']
        directions = [tuple(layout.graph.nodes[b].position[i]-layout.graph.nodes[a].position[i]
                            for i in range(2)) for a,b in zip(path,path[1:])]
        expected = len(directions)*math.ceil(c.translation_time_s(c.cell_size_m)/.01-1e-10)*.01
        for a,b in zip(directions,directions[1:]):
            angle=abs(math.atan2(a[0]*b[1]-a[1]*b[0],sum(a[i]*b[i] for i in range(2))))
            expected += math.ceil(c.rotation_time_s(angle)/.01-1e-10)*.01
        result = run(layout,[mission])
        assert result['status']=='completed' and abs(result['makespan_s']-expected)<1e-7
        finer = run(layout,[mission],dt_s=.005)
        phases = len(directions)+sum(a!=b for a,b in zip(directions,directions[1:]))
        assert finer['status']=='completed'
        assert abs(result['makespan_s']-finer['makespan_s']) <= phases*.01+1e-7
        w = dict(zip(FEATURES,[.4,-.5,-.1,-.7]))
        plain = run(layout,case['missions'],policy='heuristic',weights=w)
        high = run(layout,case['missions'],policy='heuristic',weights=w,wait_threshold_s=1e9)
        assert plain['robots_detail']==high['robots_detail'] and plain['admission_log']==high['admission_log']
        fcfs = run(layout,case['missions'],policy='fcfs')
        zero = run(layout,case['missions'],policy='heuristic',weights=w,wait_threshold_s=0.)
        assert fcfs['robots_detail']==zero['robots_detail'] and fcfs['admission_log']==zero['admission_log']
        assert all(r['safety_checks_passed'] for r in (plain,high,fcfs,zero))
        rows.append(dict(layout=name,single_vehicle_theory_passed=True,
                         single_vehicle_finer_clock_bound_passed=True,
                         inactive_protection_equivalence=True,zero_protection_fcfs_equivalence=True,
                         traffic_status=fcfs['status'],completed=fcfs['completed'],robots=len(case['missions'])))
        print(name, 'engineering checks passed', flush=True)
    report_path.write_text(json.dumps(dict(scope='engineering only; no final test',checks=rows),indent=2)+'\n')


if __name__=='__main__':
    main()
