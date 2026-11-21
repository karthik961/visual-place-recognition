import unittest
import numpy as np
from run_phase3_corrected import fuse_ranking

class TestScoring(unittest.TestCase):
    def test_reliable_candidate_is_not_penalized(self):
        candidates = np.array([0, 1])
        cosine = np.array([0.7, 0.6]) # normalized scores 0.85, 0.80
        stats = [{'inliers': 10, 'inlier_ratio': 0.5},
                 {'inliers': 0, 'inlier_ratio': 0.0}]
        ranks = fuse_ranking(candidates, cosine, stats, .7, 8, .2, True)
        self.assertEqual(int(ranks[0]), 0)

    def test_bonus_can_promote_reliable_candidate(self):
        candidates = np.array([0, 1])
        cosine = np.array([0.6, 0.58]) # normalized scores 0.80, 0.79
        stats = [{'inliers': 0, 'inlier_ratio': 0.0},
                 {'inliers': 30, 'inlier_ratio': 0.8}]
        ranks = fuse_ranking(candidates, cosine, stats, .7, 8, .2, True)
        self.assertEqual(int(ranks[0]), 1)

    def test_never_introduces_candidates(self):
        candidates = np.array([4, 0, 9, 2, 7])
        cosine = np.linspace(0.0, .9, 10)
        stats = [{'inliers': 0, 'inlier_ratio': 0.0} for _ in candidates]
        ranks = fuse_ranking(candidates, cosine, stats, .7, 8, .2, True)
        self.assertEqual(set(ranks.tolist()), set(candidates.tolist()))

if __name__ == '__main__':
    unittest.main()
