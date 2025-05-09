import sys, os

sys.path.append(os.getcwd())

import unittest
import numpy as np
from scoring import calc_rmse_dsqrd


class TestDSqrdRMSE(unittest.TestCase):

    print("Scoring: Testing D2-RMSE")

    def test_1(self):

        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        obsvt = np.array([1.1, 1.9, 3.2, 4.0, 4.9, 6.1, 7.0, 8.3, 9.1])

        dsq = calc_rmse_dsqrd(predicted=preds, observed=obsvt)
        self.assertAlmostEqual(dsq, np.float64(0.9462), 2)


if __name__ == "__main__":
    unittest.main()
