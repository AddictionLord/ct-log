from typing import Optional, Tuple

import numpy as np
import torch

CLASS_COLORS = np.array([[0, 0, 0], [140, 100, 60], [230, 40, 40], [0, 200, 255]], dtype=np.uint8)
PITH_TRUE_COLOR = (0, 255, 0)
PITH_PRED_COLOR = (255, 0, 255)
SEPARATOR_WIDTH = 4


def render_panel(
    image: torch.Tensor,
    target: np.ndarray,
    prediction: np.ndarray,
    pith_true: Optional[Tuple[float, float]] = None,
    pith_pred: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Render one slice as input | ground truth | prediction | error, side by side.

    Args:
        image: [3, H, W] stacked slices in [0, 1]; the centre channel is shown.
        target: [H, W] ground-truth class ids.
        prediction: [H, W] predicted class ids.
        pith_true: Normalized (x, y) ground-truth pith, drawn as a green cross.
        pith_pred: Normalized (x, y) predicted pith, drawn as a magenta cross.

    Returns:
        np.ndarray: [H, 4 * W + 3 * SEPARATOR_WIDTH, 3] uint8 RGB image.
    """
    centre = (image[image.shape[0] // 2].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    slice_rgb = np.stack([centre] * 3, axis=-1)
    for point, color in ((pith_true, PITH_TRUE_COLOR), (pith_pred, PITH_PRED_COLOR)):
        if point is not None:
            draw_cross(slice_rgb, point, color)

    separator = np.full((centre.shape[0], SEPARATOR_WIDTH, 3), 255, dtype=np.uint8)
    panels = [slice_rgb, colorize(target), colorize(prediction), error_overlay(prediction, target)]
    return np.concatenate([part for panel in panels for part in (panel, separator)][:-1], axis=1)


def colorize(mask: np.ndarray) -> np.ndarray:
    """Map class ids to RGB.

    Args:
        mask: [H, W] class-id array.

    Returns:
        np.ndarray: [H, W, 3] uint8 image.
    """
    return CLASS_COLORS[mask]


def error_overlay(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Colour agreement and error types: correct green, wrong class yellow, false positive red, missed blue.

    Args:
        pred: [H, W] predicted class ids.
        target: [H, W] ground-truth class ids.

    Returns:
        np.ndarray: [H, W, 3] uint8 image.
    """
    out = np.zeros((*pred.shape, 3), dtype=np.uint8)
    pf, tf = pred > 0, target > 0
    out[pf & tf & (pred == target)] = (0, 190, 0)
    out[pf & tf & (pred != target)] = (255, 210, 0)
    out[pf & ~tf] = (255, 0, 0)
    out[~pf & tf] = (0, 90, 255)
    return out


def draw_cross(image: np.ndarray, point: Tuple[float, float], color: Tuple[int, int, int], size: int = 6) -> None:
    """Draw a 3 px thick cross in place.

    Args:
        image: [H, W, 3] uint8 image, modified in place.
        point: Normalized (x, y) in [0, 1].
        color: RGB colour.
        size: Half-length of each arm in pixels.
    """
    height, width = image.shape[:2]
    x = int(round(point[0] * (width - 1)))
    y = int(round(point[1] * (height - 1)))
    image[max(y - 1, 0) : y + 2, max(x - size, 0) : x + size + 1] = color
    image[max(y - size, 0) : y + size + 1, max(x - 1, 0) : x + 2] = color
