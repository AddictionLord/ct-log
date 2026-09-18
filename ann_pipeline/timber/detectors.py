"""Timber instance detectors.

Each detector takes a grayscale CT slice and returns a list of binary masks,
one per detected timber board. Unlike the CT-log wood detectors, which keep a
single largest connected component, these keep every component that looks like
a board.
"""

from typing import List, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi


def component_shape(mask: np.ndarray) -> Tuple[float, float]:
    """Return (long_side, aspect_ratio) of the component's min-area rectangle."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0
    width, height = cv2.minAreaRect(max(contours, key=cv2.contourArea))[1]
    long_side, short_side = max(width, height), min(width, height)
    if short_side <= 0:
        return float(long_side), 0.0
    return float(long_side), float(long_side / short_side)


def threshold_components(
    img: np.ndarray,
    thresh: int = 30,
    min_px: int = 3000,
    max_px: int = 12000,
    max_long_side: float = 220.0,
    max_aspect: float = 3.5,
    open_radius: int = 0,
) -> List[np.ndarray]:
    """Threshold, label, and keep components shaped like an annotated timber board.

    Area alone cannot reject the unannotated wide top board (its area overlaps
    the true boards), so shape gates on the min-area rectangle do it: the top
    board is markedly longer and more elongated.
    """
    binary = (img > thresh).astype(np.uint8)
    if open_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_radius * 2 + 1,) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = ndi.binary_fill_holes(binary)
    labeled, count = ndi.label(binary)
    if count == 0:
        return []
    out = []
    for label_id in range(1, count + 1):
        mask = labeled == label_id
        area = int(mask.sum())
        if not min_px <= area <= max_px:
            continue
        long_side, aspect = component_shape(mask)
        if long_side > max_long_side or aspect > max_aspect:
            continue
        out.append(mask)
    out.sort(key=lambda m: int(m.sum()), reverse=True)
    return out


def threshold_components_split(
    img: np.ndarray,
    thresh: int = 30,
    min_px: int = 3000,
    max_px: int = 12000,
    max_long_side: float = 220.0,
    max_aspect: float = 3.5,
    open_radius: int = 3,
) -> List[np.ndarray]:
    """Same as threshold_components, but a morphological opening first severs
    thin bridges between boards that touch after thresholding."""
    return threshold_components(
        img,
        thresh=thresh,
        min_px=min_px,
        max_px=max_px,
        max_long_side=max_long_side,
        max_aspect=max_aspect,
        open_radius=open_radius,
    )
