"""Freeze expanded train/development data and training-only initial candidates."""
import argparse
import itertools
import json
from pathlib import Path
from scripts.agv_timed_main_common import *


def candidate_bank(layout, pilot_dir):
    candidates = []
    def add(label, vector, source):
        vector = np.array(vector, dtype=float)
        if any(np.allclose(vector, c['vector'], atol=1e-12, rtol=0) for c in candidates):
            return
        candidates.append(dict(candidate_id=label, vector=list(map(float, vector)), source=source))
    add('fcfs', [1, 0, 0, 0], 'pure waiting; trajectory equivalence checked on new train')
    add('fixed4_reference', FIXED4, 'legacy fixed four-feature projection; not legacy full Basic')
    for seed in [101, 202, 303]:
        fits = json.loads((pilot_dir / 'jobs' / f'{layout}_{seed}' / 'best.json').read_text())
        for method in ['basic', 'bo', 'pso']:
            fit = next(f for f in fits if f['algorithm'] == method)
            add(f'pilot_{method}_{seed}', fit['vector'],
                f'all pilot training-best fits, without consulting development results: {method}/{seed}')
    rng = np.random.default_rng(271828 + ['single', 'double', 'multi'].index(layout))
    index = 0
    while len(candidates) < 24:
        add(f'random_{index:02d}', unit(rng.uniform(-1, 1, 4)), 'seeded cube-uniform vector, L2-normalized')
        index += 1
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    pilot_dir = ROOT / 'outputs/agv_timed_pilot_v1'
    pilot = json.loads((pilot_dir / 'manifest.json').read_text())
    engineering = json.loads((ROOT / 'outputs/agv_timed_validation_v1/protocol_before_run.json').read_text())
    excluded = {(s['layout'], s['missions_sha256']) for s in pilot['scenarios'] + engineering['scenarios']}
    old_seeds = {s['seed'] for s in pilot['scenarios'] + engineering['scenarios'] if 'seed' in s}
    seen = set()
    seeds = set()
    cases = []
    for split_index, split in enumerate(['main_train', 'main_development']):
        for layout_index, name in enumerate(['single', 'double', 'multi']):
            layout = build_layout(name)
            for cell, (robots, window, profile) in enumerate(itertools.product(
                    [12, 24], [0, 15], ['uniform', 'approach_skew', 'exit_skew'])):
                for rep in range(4):
                    seed = 25100000 + split_index * 1000000 + layout_index * 100000 + cell * 100 + rep
                    missions, start, goal = generate_missions(layout, robots, seed, window, profile)
                    fingerprint = digest(missions)
                    identity = (name, fingerprint)
                    if identity in excluded or identity in seen or seed in old_seeds or seed in seeds:
                        raise ValueError('Scenario/seed duplicates prior physical checks or new split')
                    seen.add(identity)
                    seeds.add(seed)
                    cases.append(dict(scenario_id=f'{split}_{name}_c{cell:02d}_r{rep:02d}',
                                      split=split, layout=name, cell=cell, robots=robots,
                                      spawn_window_s=window, demand_profile=profile,
                                      dominant_start=start, dominant_goal=goal, seed=seed,
                                      missions_sha256=fingerprint, missions=missions))
    manifest = dict(
        protocol='agv_timed_main_preparation_v1', parent_results_commit='acba966726a26bdc5f8dc39543a2632736b9495d',
        physical_config=json.loads((ROOT / 'configs/agv_timed_v1.json').read_text()),
        dt_s=.01, max_time_s=1200., optimizer_seeds=[101, 202, 303, 404, 505],
        methods=['basic', 'bo', 'pso'], search_budget=72, initial_count=6,
        calibration_candidates_per_layout=24,
        charged_objectives_per_fit_including_calibration=96,
        initial_selection=dict(extra_candidates=4, minimum_L2_distance=.35,
                               fallback='fill from training-objective rank if diversity cannot yield four'),
        candidate_banks={l: candidate_bank(l, pilot_dir) for l in ['single', 'double', 'multi']},
        training_counts=dict(main_train=144, main_development=144, final_test=0),
        excluded_prior_physical_scenarios=len(excluded), source_sha256=runtime_hashes(),
        selection_access='training only; development missions generated but not evaluated during calibration',
        cost_accounting='24 shared calibration calls plus 72 search calls for every method/seed; six search initialization calls count inside 72 and reuse calibration records',
        split_note='Development is for tuning/planning, never final confirmation; final test not generated.',
        scenarios=cases,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_durable(args.output_dir / 'manifest.json', manifest)
    save_csv(args.output_dir / 'scenario_index.csv', [{k: v for k, v in s.items() if k != 'missions'} for s in cases])
    print(manifest['training_counts'])


if __name__ == '__main__':
    main()
