"""Pith detection algorithms for CT log slices.

Each detector exposes a callable `detect(img: np.ndarray) -> (x, y)` interface
returning a single (x, y) point in image coordinates.

Candidates implemented:
  - image_centre:        constant baseline, returns the geometric centre
  - wood_centroid:       centroid of the wood mask (threshold + largest CC)
  - brightest_near_centre: brightest small blob inside the wood mask
  - gradient_radial_vote:  gradient-direction Hough voting (Longuetaud-style)
  - ring_hough:          OpenCV HoughCircles on a smoothed edge map
"""

from typing import Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

XY = Tuple[int, int]


def image_centre(img: np.ndarray) -> XY:
    """Constant baseline: image geometric centre."""
    h, w = img.shape[:2]
    return w // 2, h // 2


def _wood_mask(img: np.ndarray, thresh: int = 20) -> np.ndarray:
    """Binary mask of the wood region (largest bright connected component)."""
    binary = (img > thresh).astype(np.uint8)
    labeled, n = ndi.label(binary)
    if n == 0:
        return binary
    sizes = ndi.sum(binary, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return (labeled == largest).astype(np.uint8)


def wood_centroid(img: np.ndarray) -> XY:
    """Centroid of the wood mask. Pith is roughly central within the cross-section."""
    mask = _wood_mask(img)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return image_centre(img)
    return int(xs.mean()), int(ys.mean())


def brightest_near_centre(img: np.ndarray, blur_sigma: float = 3.0, search_radius_frac: float = 0.3) -> XY:
    """Brightest point of a heavily-blurred image, restricted to a disk
    around the wood-mask centroid. Pith is locally the densest/brightest
    structure once you blur away the growth rings."""
    mask = _wood_mask(img).astype(bool)
    if not mask.any():
        return image_centre(img)
    cx, cy = wood_centroid(img)
    # Gaussian-blur the image to suppress ring detail
    blurred = cv2.GaussianBlur(img.astype(np.float32), (0, 0), blur_sigma)
    # Restrict search to a disk around the wood centroid
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    r = search_radius_frac * min(h, w) / 2.0
    search_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < r**2
    search_mask &= mask
    blurred[~search_mask] = -np.inf
    y, x = np.unravel_index(np.argmax(blurred), blurred.shape)
    return int(x), int(y)


def gradient_radial_vote(
    img: np.ndarray,
    canny_lo: int = 30,
    canny_hi: int = 90,
    blur_sigma: float = 2.0,
    vote_blur: float = 6.0,
) -> XY:
    """Longuetaud-style voting: edges from Canny vote along their gradient
    direction (perpendicular to the local ring tangent → toward the pith).
    Peak of the accumulator is the pith."""
    mask = _wood_mask(img).astype(bool)
    h, w = img.shape[:2]
    # Smooth then Canny
    smooth = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    edges = cv2.Canny(smooth, canny_lo, canny_hi)
    edges &= mask.astype(np.uint8) * 255
    # Gradient direction (Sobel) — points orthogonal to ring tangent
    gx = cv2.Sobel(smooth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mag[mag == 0] = 1.0
    # Unit-direction vectors at edge pixels
    accum = np.zeros((h, w), dtype=np.float32)
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return image_centre(img)
    dx = gx[ys, xs] / mag[ys, xs]
    dy = gy[ys, xs] / mag[ys, xs]
    # Cast a short line in BOTH directions along the gradient (we don't know
    # which side the pith is on a priori; the true side accumulates more votes
    # because all rings agree)
    max_len = int(0.5 * min(h, w))
    steps = np.arange(1, max_len)
    for sign in (-1, 1):
        line_x = xs[:, None] + sign * steps[None, :] * dx[:, None]
        line_y = ys[:, None] + sign * steps[None, :] * dy[:, None]
        line_x = np.round(line_x).astype(np.int32)
        line_y = np.round(line_y).astype(np.int32)
        valid = (line_x >= 0) & (line_x < w) & (line_y >= 0) & (line_y < h)
        flat_x = line_x[valid]
        flat_y = line_y[valid]
        np.add.at(accum, (flat_y, flat_x), 1.0)
    # Restrict accumulator to wood mask, smooth, take peak
    accum[~mask] = 0
    accum = cv2.GaussianBlur(accum, (0, 0), vote_blur)
    y, x = np.unravel_index(np.argmax(accum), accum.shape)
    return int(x), int(y)


def ring_hough(
    img: np.ndarray,
    dp: float = 1.0,
    min_dist_frac: float = 0.5,
    canny_thresh: int = 90,
    accum_thresh: int = 30,
) -> XY:
    """HoughCircles over the wood region, then average the centres of detected
    circles (each ring votes for its own centre; the pith is the mean)."""
    mask = _wood_mask(img).astype(np.uint8)
    masked = img.copy()
    masked[mask == 0] = 0
    smooth = cv2.GaussianBlur(masked, (0, 0), 2.0)
    h, w = img.shape[:2]
    min_dist = int(min_dist_frac * min(h, w))
    min_r = int(0.05 * min(h, w))
    max_r = int(0.5 * min(h, w))
    circles = cv2.HoughCircles(
        smooth,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=canny_thresh,
        param2=accum_thresh,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None or len(circles) == 0:
        return wood_centroid(img)
    centres = circles[0, :, :2]
    return int(centres[:, 0].mean()), int(centres[:, 1].mean())


DETECTORS = {
    "image_centre": image_centre,
    "wood_centroid": wood_centroid,
    "brightest_near_centre": brightest_near_centre,
    "gradient_radial_vote": gradient_radial_vote,
    "ring_hough": ring_hough,
}
