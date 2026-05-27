"""Wood-mask detectors.

Each detector takes a grayscale slice + optional pith point and returns a
binary mask of the wood cross-section.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

XY = Tuple[float, float]


def threshold_largest_cc(img: np.ndarray, thresh: int = 30) -> np.ndarray:
    """Binary threshold + keep largest connected component. Classical baseline."""
    binary = (img > thresh).astype(np.uint8)
    labeled, n = ndi.label(binary)
    if n == 0:
        return binary
    sizes = ndi.sum(binary, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    mask = (labeled == largest).astype(np.uint8)
    return ndi.binary_fill_holes(mask).astype(np.uint8)


def threshold_peel(
    img: np.ndarray,
    thresh: int = 30,
    intensity_thresh: int = 60,
    iterations: int = 5,
) -> np.ndarray:
    """Threshold + iterative boundary peeling of dark (bark) pixels.

    Starts with threshold_largest_cc, then repeatedly removes boundary pixels
    whose intensity is below intensity_thresh. Peels bark ring without
    eroding bright wood interior.
    """
    mask = threshold_largest_cc(img, thresh=thresh).astype(bool)
    for _ in range(iterations):
        boundary = ndi.binary_dilation(~mask, iterations=1) & mask
        dark_boundary = boundary & (img < intensity_thresh)
        if dark_boundary.sum() == 0:
            break
        mask = mask & ~dark_boundary
    labeled, n = ndi.label(mask.astype(np.uint8))
    if n == 0:
        return mask.astype(np.uint8)
    sizes = ndi.sum(mask, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return ndi.binary_fill_holes(labeled == largest).astype(np.uint8)


def threshold_morphology(img: np.ndarray, thresh: int = 30, close_radius: int = 5) -> np.ndarray:
    """Threshold + largest CC + morphological closing to fill rim gaps + fill holes."""
    mask = threshold_largest_cc(img, thresh=thresh)
    if close_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius * 2 + 1,) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return ndi.binary_fill_holes(mask).astype(np.uint8)


def sam_centre_plus_corners(
    img_rgb: np.ndarray,
    predictor,
    centre_point: Optional[XY] = None,
    corner_margin: int = 10,
) -> np.ndarray:
    """SAM2 with one positive point (image centre or supplied pith point) and
    four corner negative points to bound the wood region."""
    h, w = img_rgb.shape[:2]
    if centre_point is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = centre_point
    points = np.array(
        [
            [cx, cy],
            [corner_margin, corner_margin],
            [w - corner_margin, corner_margin],
            [corner_margin, h - corner_margin],
            [w - corner_margin, h - corner_margin],
        ],
        dtype=np.float32,
    )
    labels = np.array([1, 0, 0, 0, 0], dtype=np.int32)
    predictor.set_image(img_rgb)
    masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=False)
    return masks[0].astype(np.uint8)


def sam_pith_plus_corners(
    img_rgb: np.ndarray,
    predictor,
    pith_xy: Optional[XY] = None,
    corner_margin: int = 10,
) -> np.ndarray:
    """SAM2 with the pith point as the positive seed (rather than image centre)."""
    if pith_xy is None:
        # fall back to image centre
        return sam_centre_plus_corners(img_rgb, predictor, corner_margin=corner_margin)
    return sam_centre_plus_corners(img_rgb, predictor, centre_point=pith_xy, corner_margin=corner_margin)
