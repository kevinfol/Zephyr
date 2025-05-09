import sys, os

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/..")

import unittest
from feature_selection import *


class TestGenetic(unittest.TestCase):

    def test_1(self):

        print("Feature Selection: Testing Genetic Algorithm")

        num_features = 5
        gen = GeneticSelection(num_features, population=6, generations=10)

        # check that selector is running
        self.assertEqual(gen.is_still_running(), True)

        # Check the first batch
        batch1 = gen.next()

        # set some scores
        gen.set_scores([[0.1], [0.2], [0.4], [2], [1], [1.2]])

        # generate next batch
        batch2 = gen.next()
        self.assertEqual(len(batch2), 6)


if __name__ == "__main__":
    unittest.main()
