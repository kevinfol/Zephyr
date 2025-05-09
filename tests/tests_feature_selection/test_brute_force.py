import sys, os

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/..")

import unittest
from feature_selection import BruteForceFeatureSelection


class TestBruteForce(unittest.TestCase):

    def test_1(self):
        print("Feature Selection: Testing Brute Force")

        num_features = 5
        forced_features = [1]
        bf = BruteForceFeatureSelection(num_features, forced_features=forced_features)
        # check that selector is running
        self.assertEqual(bf.is_still_running(), True)
        # Check that the first batch is sequential
        batch = bf.next()
        self.assertEqual(batch, [2, 3, 6, 7, 10, 11, 14, 15, 18, 19])


if __name__ == "__main__":
    unittest.main()
