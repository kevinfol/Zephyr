import sys, os

sys.path.append(os.getcwd())

import unittest
import numpy as np
from scoring import calc_adj_rsqrd


class TestAdjRSqrd(unittest.TestCase):

    print("Scoring: Testing Adjusted R-Squared")

    def test_1(self):

        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        obsvt = np.array([1.1, 1.9, 3.2, 4.0, 4.9, 6.1, 7.0, 8.3, 9.1])
        n_feats = 4

        arsq = calc_adj_rsqrd(predicted=preds, observed=obsvt, n_feats=n_feats)
        self.assertEqual(arsq, np.float64(0.9941879237972231))


if __name__ == "__main__":
    unittest.main()
