"""Read-only topology audit; write an isolated v2 preparation report, never train."""
import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Topology modules use only the standard library. Avoid importing the unrelated
# Gym warehouse registration in rware/__init__.py for this standalone audit.
if 'rware' not in sys.modules:
    package = types.ModuleType('rware')
    package.__path__ = [str(ROOT / 'rware')]
    sys.modules['rware'] = package
from rware.agv_timed_simulation import build_layout
from rware.agv_multicross_policy_simulation import shortest_path


def inspect_layout(name):
    layout = build_layout(name)
    starts = sorted(n for n in layout.stations if n.endswith(('_START', '_IN')))
    goals = sorted(n for n in layout.stations if n.endswith(('_EXIT', '_OUT')))
    pairs, failures = [], []
    for start in starts:
        for goal in goals:
            if start.rsplit('_', 1)[0] == goal.rsplit('_', 1)[0]:
                continue
            try:
                path = shortest_path(layout.graph, start, goal)
                if not path or path[0] != start or path[-1] != goal:
                    raise ValueError('Invalid route endpoints')
                zones = {a.area_id for a in layout.interaction_areas
                         if set(path) & a.conflict_zone_nodes}
                if not zones:
                    raise ValueError('Route bypasses all shared zones')
                pairs.append(dict(start=start, goal=goal, edges=len(path)-1,
                                  shared_zones=len(zones)))
            except (ValueError, KeyError, RuntimeError) as exc:
                failures.append(dict(start=start, goal=goal, error=str(exc)))
    counts = {'single': [20, 24, 28], 'double': [24, 30, 36],
              'multi': [2*len(starts), 3*len(starts), 4*len(starts)]}[name]
    return dict(layout=name, nodes=len(layout.graph.nodes), starts=starts, goals=goals,
                shared_zones=len(layout.interaction_areas), proposed_counts=counts,
                count_basis=('legacy generator defaults' if name != 'multi' else
                             '2/3/4 vehicles per inlet; input-defined demand, not congestion labels'),
                balanced_allocation_possible=all(n % len(starts) == 0 and n % len(goals) == 0
                                                 for n in counts),
                reachable_pairs=pairs, invalid_pairs=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a new report directory; previous records are immutable')
    layouts = [inspect_layout(n) for n in ('single', 'double', 'multi')]
    sources = ['rware/agv_timed_simulation.py', 'rware/agv_layouts.py',
               'rware/agv_multicross_layout.py', 'scripts/preflight_agv_capstone_v2.py']
    report = dict(protocol='agv_capstone_v2_preparation', training_started=False,
                  final_test_generated=False, layouts=layouts,
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                 for p in sources},
                  release_gates=['Implement and verify separate grid-only engine',
                                 'Verify exact start/exit quotas with reachable OD matching',
                                 'Freeze multi skew allocation and timeout before training',
                                 'Freeze common initialization and optimizer budget rules'],
                  note='Topology audit only; no collision, liveness, or policy performance claim.')
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'topology_preflight.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    for item in layouts:
        print(item['layout'], 'inlets', len(item['starts']), 'outlets', len(item['goals']),
              'counts', item['proposed_counts'], 'valid OD', len(item['reachable_pairs']),
              'invalid OD', len(item['invalid_pairs']),
              'balanced', item['balanced_allocation_possible'])


if __name__ == '__main__':
    main()
