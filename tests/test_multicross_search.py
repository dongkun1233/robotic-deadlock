import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('search_runner',Path(__file__).resolve().parents[1]/'scripts/run_multicross_search.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SearchTests(unittest.TestCase):
    def exercise(self, algorithm, limit=None, previous=None):
        rng=np.random.default_rng(101)
        initial=np.array([module.unit(v) for v in rng.uniform(-1,1,(4,4))])
        calls=[]
        def evaluate(vector,source):
            value=float(np.sum((vector-np.array([.5,-.5,-.5,-.5]))**2))
            row=(vector.copy(),value,source)
            if previous is not None and len(calls)<len(previous):
                old=previous[len(calls)]
                np.testing.assert_allclose(vector,old[0],atol=1e-14)
                self.assertEqual(source,old[2])
            if limit is not None and len(calls)==limit:
                raise InterruptedError
            calls.append(row)
            return value
        try:
            module.search(algorithm,101,12,initial,evaluate)
        except InterruptedError:
            pass
        return calls

    def test_equal_initial_candidates_and_budgets(self):
        runs=[self.exercise(a) for a in ('basic','bo','pso')]
        for run in runs:
            self.assertEqual(len(run),12)
            for vector,_,_ in run:
                self.assertAlmostEqual(float(np.linalg.norm(vector)),1.0)
        for i in range(4):
            np.testing.assert_array_equal(runs[0][i][0],runs[1][i][0])
            np.testing.assert_array_equal(runs[0][i][0],runs[2][i][0])

    def test_deterministic_resume(self):
        for algorithm in ('basic','bo','pso'):
            with self.subTest(algorithm=algorithm):
                full=self.exercise(algorithm)
                partial=self.exercise(algorithm,limit=7)
                resumed=self.exercise(algorithm,previous=partial)
                for a,b in zip(full,resumed):
                    np.testing.assert_array_equal(a[0],b[0])
                    self.assertEqual(a[1:],b[1:])

    def test_zero_vector_rejected(self):
        with self.assertRaises(ValueError):
            module.unit(np.zeros(4))


if __name__=='__main__':
    unittest.main()
