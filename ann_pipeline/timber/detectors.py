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
from skimage.segmentation import watershed


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


def split_blob_watershed(mask: np.ndarray, seed_frac: float = 0.5) -> List[np.ndarray]:
    """Split a blob of touching boards via watershed on its distance transform.

    Boards are convex and meet at a thin neck, so the distance transform has one
    peak per board. Seeds are the connected components of dist > seed_frac*max.
    Returns [mask] unchanged when only one seed is found.
    """
    dist = ndi.distance_transform_edt(mask)
    if dist.max() <= 0:
        return [mask]
    seeds, n_seeds = ndi.label(dist > seed_frac * dist.max())
    if n_seeds <= 1:
        return [mask]
    labels = watershed(-dist, seeds, mask=mask)
    return [labels == i for i in range(1, n_seeds + 1) if (labels == i).any()]


def threshold_components_watershed(
    img: np.ndarray,
    thresh: int = 30,
    min_px: int = 3000,
    max_px: int = 12000,
    max_long_side: float = 220.0,
    max_aspect: float = 3.5,
    seed_frac: float = 0.5,
) -> List[np.ndarray]:
    """Threshold + components, watershed-splitting blobs too large to be one board.

    Recovers boards that thresholding fused to a neighbour (or to the top plank),
    which the plain shape gate would discard wholesale.
    """
    binary = ndi.binary_fill_holes((img > thresh).astype(np.uint8))
    labeled, count = ndi.label(binary)
    if count == 0:
        return []
    candidates = []
    for label_id in range(1, count + 1):
        mask = labeled == label_id
        area = int(mask.sum())
        if area < min_px:
            continue
        if area > max_px:
            candidates.extend(split_blob_watershed(mask, seed_frac=seed_frac))
        else:
            candidates.append(mask)
    out = []
    for mask in candidates:
        area = int(mask.sum())
        if not min_px <= area <= max_px:
            continue
        long_side, aspect = component_shape(mask)
        if long_side > max_long_side or aspect > max_aspect:
            continue
        out.append(mask)
    out.sort(key=lambda m: int(m.sum()), reverse=True)
    return out


def threshold_components_full(
    img: np.ndarray,
    thresh: int = 30,
    min_px: int = 3000,
    max_px: int = 12000,
    max_long_side: float = 220.0,
    max_aspect: float = 3.5,
    open_radius: int = 3,
    seed_frac: float = 0.5,
) -> List[np.ndarray]:
    """Opening to sever thin bridges, then watershed on any blob still oversized.

    The opening handles boards joined by a narrow neck; the watershed handles
    the wider fusions it cannot cut. Shape gates are applied last, so a board
    rescued from a merge is judged on its own dimensions.
    """
    binary = (img > thresh).astype(np.uint8)
    if open_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_radius * 2 + 1,) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = ndi.binary_fill_holes(binary)
    labeled, count = ndi.label(binary)
    if count == 0:
        return []
    candidates = []
    for label_id in range(1, count + 1):
        mask = labeled == label_id
        area = int(mask.sum())
        if area < min_px:
            continue
        if area > max_px:
            candidates.extend(split_blob_watershed(mask, seed_frac=seed_frac))
        else:
            candidates.append(mask)
    out = []
    for mask in candidates:
        area = int(mask.sum())
        if not min_px <= area <= max_px:
            continue
        long_side, aspect = component_shape(mask)
        if long_side > max_long_side or aspect > max_aspect:
            continue
        out.append(mask)
    out.sort(key=lambda m: int(m.sum()), reverse=True)
    return out


def threshold_components_v2(
    img: np.ndarray,
    thresh: int = 30,
    min_px: int = 3000,
    max_px: int = 12000,
    max_long_side: float = 228.0,
    open_radius: int = 3,
    seed_frac: float = 0.5,
) -> List[np.ndarray]:
    """Recommended detector: opening + watershed, gated on long side only.

    After the opening, aspect ratio no longer separates boards from the top
    plank (true boards reach 3.62, plank fragments start at 3.29), but the
    min-area-rect long side does, with a clean margin: true boards top out at
    211.8px, plank fragments start at 245.8px. Gating on aspect as well costs a
    real board on frame 033, whose retained sliver of plank inflates its aspect.
    """
    binary = (img > thresh).astype(np.uint8)
    if open_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_radius * 2 + 1,) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = ndi.binary_fill_holes(binary)
    labeled, count = ndi.label(binary)
    if count == 0:
        return []
    candidates = []
    for label_id in range(1, count + 1):
        mask = labeled == label_id
        area = int(mask.sum())
        if area < min_px:
            continue
        if area > max_px:
            candidates.extend(split_blob_watershed(mask, seed_frac=seed_frac))
        else:
            candidates.append(mask)
    out = []
    for mask in candidates:
        area = int(mask.sum())
        if not min_px <= area <= max_px:
            continue
        if component_shape(mask)[0] > max_long_side:
            continue
        out.append(mask)
    out.sort(key=lambda m: int(m.sum()), reverse=True)
    return out
