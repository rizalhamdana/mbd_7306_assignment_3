
import unittest
import pandas as pd
from pattern_mining_engine import PatternMiningEngine

class TestPatternMiningEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sample dataset: binary encoded transaction data
        data = {
            'milk': [1, 0, 1, 1],
            'bread': [1, 1, 1, 0],
            'butter': [0, 1, 1, 1],
            'apples': [0, 0, 1, 1]
        }
        cls.df_encoded = pd.DataFrame(data)
        cls.engine = PatternMiningEngine(cls.df_encoded)

    def test_default_parameters(self):
        self.assertEqual(self.engine.min_support, 0.003)
        self.assertEqual(self.engine.min_confidence, 0.001)
        self.assertEqual(self.engine.top_n, 100)
        self.assertEqual(self.engine.algorithm, 'both')

    def test_setters(self):
        self.engine.set_min_support(0.01)
        self.assertEqual(self.engine.min_support, 0.01)

        self.engine.set_min_confidence(0.05)
        self.assertEqual(self.engine.min_confidence, 0.05)

        self.engine.set_top_n(10)
        self.assertEqual(self.engine.top_n, 10)

        self.engine.set_weights((0.2, 0.3, 0.5))
        self.assertEqual(self.engine.weights, (0.2, 0.3, 0.5))

        self.engine.set_algorithm('apriori')
        self.assertEqual(self.engine.algorithm, 'apriori')

    def test_run(self):
        result_ap, result_fp, result_combined = self.engine.run()
        # Ensure DataFrames or None are returned
        self.assertTrue(result_ap is not None or result_fp is not None)

if __name__ == '__main__':
    unittest.main()
