"""Unit tests for Phase 4 parameter sweep and retrieval evaluation."""
import unittest
import numpy as np

from run_phase3_corrected import fuse_ranking
from run_phase4 import adaptive_ranking, fixed_ranking, recall, choose_parameters, classify_top1


class Phase4Tests(unittest.TestCase):
    def setUp(self):
        self.candidates = np.array([[3, 2, 4, 1, 5, 0, 6, 7, 8, 9],
                                    [7, 1, 6, 8, 3, 5, 9, 0, 4, 2]])
        self.cosine = np.zeros((2, 10), dtype=float)
        self.cosine[0, 3] = 0.78
        self.cosine[0, 2] = 0.75
        self.cosine[1, 7] = 0.8
        self.cosine[1, 1] = 0.77
        self.deep = (self.cosine[np.arange(2)[:, None], self.candidates] + 1) / 2
        self.stats = {"inliers": np.zeros((2, 10), dtype=int),
                      "inlier_ratio": np.zeros((2, 10), dtype=float)}
        self.stats["inliers"][0, 1] = 30
        self.stats["inlier_ratio"][0, 1] = 0.9

    def test_zero_bonus_reproduces_deep_ranking(self):
        ranked = adaptive_ranking(self.candidates, self.deep, self.stats, 0, 8, 0.2)
        self.assertTrue(np.array_equal(ranked[:, 0], np.array([3, 7])))

    def test_geometry_bonus_promotes_reliable_candidate(self):
        ranked = adaptive_ranking(self.candidates, self.deep, self.stats, 0.05, 8, 0.2)
        self.assertEqual(int(ranked[0, 0]), 2)
        self.assertEqual(int(ranked[1, 0]), 7)

    def test_vectorized_matches_phase3_fusion(self):
        rank = adaptive_ranking(self.candidates, self.deep, self.stats, .05, 8, .2)
        for i in range(2):
            stats = [{"inliers": int(self.stats["inliers"][i, j]),
                      "inlier_ratio": float(self.stats["inlier_ratio"][i, j])}
                     for j in range(10)]
            expected = fuse_ranking(self.candidates[i], self.cosine[i], stats,
                                    .7, 8, .2, True, geom_bonus=.05, inlier_scale=30)
            self.assertTrue(np.array_equal(rank[i], expected))

    def test_fixed_fusion_keeps_same_candidates(self):
        r = fixed_ranking(self.candidates, self.deep, self.stats)
        for i in range(2):
            self.assertEqual(set(r[i]), set(self.candidates[i]))

    def test_recall_uses_global_indices_after_split(self):
        rank = np.array([[1, 5, 3, 7, 2, 4, 6, 0, 8, 9],
                         [9, 1, 2, 3, 0, 4, 5, 6, 7, 8]])
        self.assertEqual(recall(rank, np.array([1]), tolerance=2)["recall_at_1"], 0)
        self.assertEqual(recall(rank, np.array([0]), tolerance=2)["recall_at_1"], 1)

    def test_choose_parameters_uses_development_only(self):
        rows = [
            dict(subset="development", sequence="day_left", geom_bonus=.01, min_inliers=8,
                 min_ratio=.2, recall_at_1=.8, recall_at_5=.9),
            dict(subset="development", sequence="night_right", geom_bonus=.01, min_inliers=8,
                 min_ratio=.2, recall_at_1=.6, recall_at_5=.9),
            dict(subset="development", sequence="day_left", geom_bonus=.1, min_inliers=8,
                 min_ratio=.2, recall_at_1=.7, recall_at_5=.9),
            dict(subset="development", sequence="night_right", geom_bonus=.1, min_inliers=8,
                 min_ratio=.2, recall_at_1=.5, recall_at_5=.9),
            dict(subset="retrospective_evaluation", sequence="day_left", geom_bonus=.1,
                 min_inliers=8, min_ratio=.2, recall_at_1=1.0, recall_at_5=1.0),
        ]
        self.assertEqual(choose_parameters(rows)["geom_bonus"], .01)

    def test_case_labels(self):
        self.assertEqual(classify_top1(10, 30, 30, 2), "improved")
        self.assertEqual(classify_top1(30, 10, 30, 2), "worsened")
        self.assertEqual(classify_top1(30, 31, 30, 2), "both_correct")
        self.assertEqual(classify_top1(10, 11, 30, 2), "both_wrong")


if __name__ == "__main__":
    unittest.main()
