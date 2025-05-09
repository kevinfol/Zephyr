import sys, os

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/..")

import unittest
from feature_selection import *


class TestSequential(unittest.TestCase):

    def test_1(self):

        print("Feature Selection: Testing Sequential Floating")

        num_features = 5
        forced_features = []
        bf = SequentialFloatingSelection(num_features, forced_features=forced_features)
        # check that selector is running
        self.assertEqual(bf.is_still_running(), True)

        # Get the starting chromosome
        start = bf.starting_chromosome

        # Create the expected batch
        expected_batch = [start] + [start ^ (2**n) for n in range(5)]

        # Check the first batch
        batch = bf.next()

        # filter out the zeros
        while 0 in expected_batch:
            idx = expected_batch.index(0)
            batch.pop(idx)
            expected_batch.pop(idx)

        print()
        self.assertEqual(batch, expected_batch)

        bf.set_scores([[0], [0], [0.5], [1], [0], [0]])
        self.assertEqual(bf.chromosome_scoring_table[start], [0])
        # new starting chromosome should be (start ^ (2**2))
        self.assertEqual(bf.starting_chromosome, start ^ (2**2))


if __name__ == "__main__":
    unittest.main()
