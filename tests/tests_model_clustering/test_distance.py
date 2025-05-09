import sys, os

sys.path.append(os.getcwd())

import unittest

from model_clustering import distance_analysis, weighted_hamming_distance, to_bitstring


class TestDistance(unittest.TestCase):

    chromosome_list = list(range(1, 36))
    score_list = [
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
    ]
    score_map = {c: s for (c, s) in zip(chromosome_list, score_list)}

    def test_distance(self):

        print("Clustering: Testing Hamming Distance")

        distance = weighted_hamming_distance(
            bitstring_1=to_bitstring(self.chromosome_list[5], 6),
            bitstring_2=to_bitstring(self.chromosome_list[10], 6),
            score_2=[self.score_list[10]],
        )
        self.assertAlmostEqual(distance, 0.6, 2)

        best_three_chromosomes = distance_analysis(self.score_map, 3, 6)
        self.assertListEqual(best_three_chromosomes, [6, 24, 35])


if __name__ == "__main__":
    unittest.main()
