import sys, os

sys.path.append(os.getcwd())

import unittest
import numpy as np
from scoring import calc_rmse


class TestRMSE(unittest.TestCase):

    print("Scoring: Testing RMSE")

    def test_1(self):

        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        obsvt = np.array([1.1, 1.9, 3.2, 4.0, 4.9, 6.1, 7.0, 8.3, 9.1])

        rmse = calc_rmse(predicted=preds, observed=obsvt)
        self.assertEqual(rmse, np.float64(0.14142135623730964))


if __name__ == "__main__":
    unittest.main()
