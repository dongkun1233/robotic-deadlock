"""Shared, immutable inputs and durable output for physical-time main preparation."""
import hashlib
import gzip
import json
import os
from pathlib import Path
import numpy as np
from scripts.agv_timed_pilot_common import (
    ROOT, FEATURES, FIXED4, SOURCE_FILES, build_layout, config_from_manifest,
    digest, generate_missions, key, objective, save_csv, simulate, unit, weights,
)
EXTRA_SOURCES = [
    'scripts/agv_timed_main_common.py', 'scripts/prepare_agv_timed_main.py',
    'scripts/calibrate_agv_timed_initial.py', 'scripts/train_agv_timed_main.py',
]


def atomic_durable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def runtime_hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in dict.fromkeys(SOURCE_FILES + EXTRA_SOURCES)}


def read_manifest(path):
    path = Path(path)
    if not path.exists() and path.with_suffix(path.suffix + '.gz').exists():
        data = gzip.decompress(path.with_suffix(path.suffix + '.gz').read_bytes())
        integrity = json.loads((path.parent / 'manifest_integrity.json').read_text())
        if hashlib.sha256(data).hexdigest() != integrity['manifest_sha256']:
            raise ValueError('Compressed manifest checksum mismatch')
        path.write_bytes(data)
    manifest = json.loads(path.read_text())
    if manifest['source_sha256'] != runtime_hashes():
        raise ValueError('Runtime source differs from frozen main manifest')
    return manifest


def training_cases(manifest, layout):
    cases = [s for s in manifest['scenarios']
             if s['split'] == 'main_train' and s['layout'] == layout]
    if len(cases) != 48 or len({s['scenario_id'] for s in cases}) != 48:
        raise ValueError('Expected 48 unique training scenarios/layout')
    return cases


def require_rows(rows, cases, complete=True):
    if len(rows) > len(cases) or (complete and len(rows) != len(cases)):
        raise ValueError('Wrong scenario count')
    for row, case in zip(rows, cases):
        if row['scenario_id'] != case['scenario_id'] or row['missions_sha256'] != case['missions_sha256']:
            raise ValueError('Saved scenario identity mismatch')


def run_case(layout, case, manifest, policy, vector):
    row, result = simulate(layout, case, config_from_manifest(manifest),
                           manifest['dt_s'], policy, vector, manifest['max_time_s'])
    if not result['safety_checks_passed']:
        raise AssertionError('Engine safety checks did not pass')
    row['missions_sha256'] = case['missions_sha256']
    row['trajectory_sha256'] = digest({
        'robots': result['robots_detail'], 'admission_log': result['admission_log']})
    row['safety_checks_passed'] = True
    return row


def choose_initial(candidate_results, count=4, min_distance=.35):
    """Rank only training objective; retain diversity with explicit fallback."""
    ranked = sorted((r for r in candidate_results
                     if r['candidate_id'] not in ('fcfs', 'fixed4_reference')),
                    key=lambda r: (r['objective'], r['candidate_id']))
    selected = []
    anchors = [np.array([1., 0., 0., 0.]), FIXED4]
    for item in ranked:
        vector = np.array(item['vector'])
        if all(np.linalg.norm(vector - old) >= min_distance
               for old in anchors + [np.array(r['vector']) for r in selected]):
            selected.append(item)
        if len(selected) == count:
            return selected, False
    for item in ranked:
        if item['candidate_id'] not in {r['candidate_id'] for r in selected}:
            selected.append(item)
        if len(selected) == count:
            return selected, True
    raise ValueError('Not enough distinct initial candidates')
