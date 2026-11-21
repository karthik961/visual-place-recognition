"""ORB and RANSAC verification of candidate place matches."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


@dataclass
class LocalFeatures:
    points: np.ndarray  # N x 2, pixel coordinates on resized image
    descriptors: np.ndarray | None


def extract_local_features(path: Path, orb, max_width: int = 640) -> LocalFeatures:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError(f"Unable to read image: {path}")
    if im.shape[1] > max_width:
        new_h = round(im.shape[0] * max_width / im.shape[1])
        im = cv2.resize(im, (max_width, new_h), interpolation=cv2.INTER_AREA)
    keypoints, descriptors = orb.detectAndCompute(im, None)
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2)
    return LocalFeatures(points, descriptors)


def verify_pair(query: LocalFeatures, ref: LocalFeatures, ratio: float = 0.75,
                ransac_threshold: float = 3.0) -> dict:
    """Return good matches, homography inliers, and inlier fraction.

    A homography is a *verification heuristic* for nonplanar scenes, not proof of place identity.
    """
    empty = {"good_matches": 0, "inliers": 0, "inlier_ratio": 0.0}
    if query.descriptors is None or ref.descriptors is None:
        return empty
    if len(query.descriptors) < 2 or len(ref.descriptors) < 2:
        return empty

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    neighbors = matcher.knnMatch(query.descriptors, ref.descriptors, k=2)
    good = [m for pair in neighbors if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]
    if len(good) < 4:
        return {"good_matches": len(good), "inliers": 0, "inlier_ratio": 0.0}

    src = np.float32([query.points[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref.points[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    try:
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)
    except cv2.error:
        mask = None
    count = int(mask.ravel().sum()) if mask is not None else 0
    return {"good_matches": len(good), "inliers": count,
            "inlier_ratio": count / len(good)}
