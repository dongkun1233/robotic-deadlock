import copy
import json
import sys
import unittest
import tempfile
import hashlib
import io
import contextlib
from unittest.mock import patch
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.agv_timed_main_common import ROOT, FIXED4, choose_initial, require_rows, read_manifest, training_cases
from scripts.run_multicross_search import search

class MainPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = read_manifest(ROOT / 'outputs/agv_timed_main_v1/manifest.json')

    def test_split_identity_and_all_strata(self):
        scenarios = self.manifest['scenarios']
        self.assertEqual(len(scenarios), 288)
        self.assertEqual(len({s['seed'] for s in scenarios}), 288)
        self.assertEqual(len({(s['layout'], s['missions_sha256']) for s in scenarios}), 288)
        for layout in ['single', 'double', 'multi']:
            for split in ['main_train', 'main_development']:
                cases = [s for s in scenarios if s['layout'] == layout and s['split'] == split]
                self.assertEqual(len(cases), 48)
                self.assertTrue(all(sum(s['cell'] == c for s in cases) == 4 for c in range(12)))
        self.assertFalse(any(s['split'] == 'final_test' for s in scenarios))

    def test_training_filter_excludes_development(self):
        for layout in ['single', 'double', 'multi']:
            self.assertTrue(all(s['split'] == 'main_train' for s in training_cases(self.manifest, layout)))

    def test_prior_physical_cases_are_excluded(self):
        previous = []
        for path in ['outputs/agv_timed_pilot_v1/manifest.json', 'outputs/agv_timed_validation_v1/protocol_before_run.json']:
            previous.extend(json.loads((ROOT / path).read_text())['scenarios'])
        old = {(s['layout'], s['missions_sha256']) for s in previous}
        self.assertFalse(old.intersection((s['layout'], s['missions_sha256']) for s in self.manifest['scenarios']))

    def test_bank_is_finite_unique_and_seeded(self):
        for bank in self.manifest['candidate_banks'].values():
            self.assertEqual(len(bank), 24)
            self.assertEqual(len({c['candidate_id'] for c in bank}), 24)
            vectors = np.array([c['vector'] for c in bank])
            self.assertTrue(np.all(np.isfinite(vectors)))
            self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=1), 1))
            for i, vector in enumerate(vectors):
                self.assertFalse(any(np.allclose(vector, other, atol=1e-12, rtol=0) for other in vectors[:i]))

    def test_selection_rank_diversity_and_fallback(self):
        anchors = [dict(candidate_id='fcfs', vector=[1.,0,0,0], objective=10),
                   dict(candidate_id='fixed4_reference', vector=list(FIXED4), objective=20)]
        rest = [dict(candidate_id=str(i), vector=v, objective=i) for i,v in enumerate(
            [[0.,1,0,0], [0.,0,1,0], [0.,0,0,1], [-1.,0,0,0], [0.,-1,0,0]])]
        chosen, fallback = choose_initial(anchors + rest)
        self.assertEqual([c['candidate_id'] for c in chosen], ['0','1','2','3'])
        self.assertFalse(fallback)
        chosen, fallback = choose_initial(anchors + rest, min_distance=3)
        self.assertEqual([c['candidate_id'] for c in chosen], ['0','1','2','3'])
        self.assertTrue(fallback)

    def test_partial_cache_cannot_impersonate_complete_or_other_scenario(self):
        cases = training_cases(self.manifest, 'single')
        rows = [dict(scenario_id=c['scenario_id'], missions_sha256=c['missions_sha256']) for c in cases]
        require_rows(rows, cases)
        require_rows(rows[:10], cases, complete=False)
        with self.assertRaises(ValueError): require_rows(rows[:10], cases)
        corrupted = copy.deepcopy(rows);corrupted[0]['missions_sha256'] = 'wrong'
        with self.assertRaises(ValueError): require_rows(corrupted, cases)

    def test_main_budget_counts_shared_initial_for_all_methods(self):
        initial = np.array([[1.,0,0,0], FIXED4, [0.,1,0,0], [0.,0,1,0], [0.,0,0,1], [-1.,0,0,0]])
        histories = []
        for method in self.manifest['methods']:
            history = []
            def evaluate(v, source):
                history.append(list(v));return float(np.dot(v, [1,2,3,4]))
            search(method, 101, self.manifest['search_budget'], initial, evaluate)
            self.assertEqual(len(history), 72)
            self.assertEqual(len(history) + self.manifest['calibration_candidates_per_layout'], 96)
            histories.append(history[:6])
        self.assertEqual(histories[0], histories[1]);self.assertEqual(histories[1], histories[2])

    def test_real_search_driver_resumes_partial_objective_without_recounting(self):
        from scripts import train_agv_timed_main as driver
        base = ROOT / 'outputs/agv_timed_main_v1'
        if not (base / 'initial_weights.json').exists():
            self.skipTest('Requires archived calibration output restored')
        selection = json.loads((base / 'initial_weights.json').read_text())
        for method in self.manifest['methods']:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                (out / 'manifest.json').write_text('{}')
                selected = copy.deepcopy(selection)
                selected['manifest_sha256'] = hashlib.sha256(b'{}').hexdigest()
                (out / 'initial_weights.json').write_text(json.dumps(selected))
                for item in selected['selected_by_layout']['single']['candidates']:
                    path = out / 'calibration/single' / (item['candidate_id'] + '.json')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((base / 'calibration/single' / path.name).read_bytes())
                completed_calls = []
                interrupt = [True]
                def simulate(layout, case, manifest, policy, vector):
                    if interrupt[0] and len(completed_calls) == 51:
                        raise RuntimeError('deliberate interruption within objective')
                    completed_calls.append(case['scenario_id'])
                    return dict(scenario_id=case['scenario_id'], missions_sha256=case['missions_sha256'],
                                status='completed', makespan_s=100+sum(abs(float(v)) for v in vector))
                argv = ['train', '--manifest', str(out / 'manifest.json'), '--layout', 'single',
                        '--seed', '101', '--method', method]
                with patch.object(sys, 'argv', argv), patch.object(driver, 'read_manifest', return_value=self.manifest), \
                     patch.object(driver, 'run_case', side_effect=simulate), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, 'deliberate interruption'):
                        driver.main()
                    interrupt[0] = False
                    driver.main()
                    self.assertEqual(len(completed_calls), 66*48)
                    before = len(completed_calls)
                    driver.main()
                    self.assertEqual(len(completed_calls), before)
                done = json.loads((out / 'search' / f'single_101_{method}' / 'complete.json').read_text())
                self.assertEqual(done['trials'], 72)
                self.assertFalse(done['development_evaluated'])

if __name__ == '__main__': unittest.main()
