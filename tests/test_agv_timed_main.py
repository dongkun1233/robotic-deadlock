import copy
import json
import sys
import unittest
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

if __name__ == '__main__': unittest.main()
