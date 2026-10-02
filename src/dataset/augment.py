import math
from typing import Tuple

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F

FLIP_PROB = 0.5
INTENSITY_PROB = 0.8
CONTRAST_RANGE = (0.8, 1.2)
BRIGHTNESS_RANGE = (-0.08, 0.08)
GAMMA_RANGE = (0.8, 1.25)
NOISE_STD_MAX = 0.02


def augment_sample(
    image: torch.Tensor, mask: torch.Tensor, pith_xy: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random rotation, horizontal flip and intensity jitter for one CT slice stack.

    A log cross-section has no preferred orientation, so the rotation angle is uniform over the
    full circle. The pith point is moved with the image; corners exposed by the rotation are
    filled with 0, which is background both in the image and in the mask.

    Args:
        image: [3, H, W] float image in [0, 1] (stacked neighbor slices).
        mask: [H, W] int64 class-id mask.
        pith_xy: [3] normalized (x, y, valid) pith target.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: augmented image, mask and pith target.
    """
    angle = float(torch.empty(1).uniform_(0.0, 360.0))
    image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
    mask = F.rotate(mask.unsqueeze(0), angle, interpolation=InterpolationMode.NEAREST, fill=0).squeeze(0)
    pith_xy = rotate_point(pith_xy, angle, image.shape[-1], image.shape[-2])

    if torch.rand(1).item() < FLIP_PROB:
        image = image.flip(-1)
        mask = mask.flip(-1)
        if pith_xy[2] > 0:
            width = image.shape[-1]
            pith_xy = torch.stack([(width - 1) / width - pith_xy[0], pith_xy[1], pith_xy[2]])

    if torch.rand(1).item() < INTENSITY_PROB:
        image = jitter_intensity(image)
    return image, mask, pith_xy


def rotate_point(pith_xy: torch.Tensor, angle: float, width: int, height: int) -> torch.Tensor:
    """Rotate a normalized point about the image centre, counter-clockwise as F.rotate does.

    F.rotate turns about pixel ((W - 1) / 2, (H - 1) / 2); using 0.5 in normalized coordinates
    instead would shift the point by up to ~1.4 px at 784 px.

    Args:
        pith_xy: [3] normalized (x, y, valid).
        angle: Rotation in degrees, counter-clockwise on screen (y axis pointing down).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        torch.Tensor: [3] rotated (x, y, valid); returned unchanged when valid is 0.
    """
    if pith_xy[2] <= 0:
        return pith_xy
    theta = math.radians(angle)
    cx, cy = (width - 1) / (2 * width), (height - 1) / (2 * height)
    dx, dy = (float(pith_xy[0]) - cx) * width, (float(pith_xy[1]) - cy) * height
    x = cx + (dx * math.cos(theta) + dy * math.sin(theta)) / width
    y = cy + (-dx * math.sin(theta) + dy * math.cos(theta)) / height
    return torch.tensor([x, y, float(pith_xy[2])], dtype=pith_xy.dtype)


def jitter_intensity(image: torch.Tensor) -> torch.Tensor:
    """Gamma, contrast, brightness and Gaussian noise, shared across the stacked slices.

    Args:
        image: [3, H, W] float image in [0, 1].

    Returns:
        torch.Tensor: [3, H, W] jittered image clamped to [0, 1].
    """
    gamma = float(torch.empty(1).uniform_(*GAMMA_RANGE))
    contrast = float(torch.empty(1).uniform_(*CONTRAST_RANGE))
    brightness = float(torch.empty(1).uniform_(*BRIGHTNESS_RANGE))
    noise_std = float(torch.empty(1).uniform_(0.0, NOISE_STD_MAX))
    image = image.clamp(0.0, 1.0).pow(gamma)
    mean = image.mean()
    image = (image - mean) * contrast + mean + brightness
    image = image + torch.randn_like(image) * noise_std
    return image.clamp(0.0, 1.0)
