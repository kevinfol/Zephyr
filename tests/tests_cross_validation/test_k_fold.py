import sys, os

sys.path.append(os.getcwd())

import unittest

from cross_validation import k_fold


class TestKFold(unittest.TestCase):

    def test_1(self):
        print("Cross-Validation: Testing K-Fold")
        output = k_fold(4, 4)
        self.assertEqual(len(output), 4)
        for i, (test, train) in enumerate(output):
            self.assertEqual(i, test[0])
            self.assertTrue(test[0] not in train)


if __name__ == "__main__":
    unittest.main()
