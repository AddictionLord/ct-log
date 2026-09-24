from typing import List, Sequence

import torch


def effective_number_weights(class_counts: Sequence[int], beta: float) -> torch.Tensor:
    """Class-Balanced weights from the effective number of samples (Cui et al. 2019).

    Effective number E_n = (1 - beta^n) / (1 - beta) models the diminishing marginal value of
    more samples of an already-common class (near-duplicate pixels/instances overlap in the
    space they cover). Weight is proportional to 1 / E_n, renormalized to sum to num_classes so
    the weighted loss stays on the same scale as the unweighted one.

    beta=0 gives uniform weights (1 / E_n = 1 for every class, since E_n = 1).
    beta->1 approaches inverse-frequency weighting (1 / n).
    Reference: https://arxiv.org/abs/1901.05555

    Beta must be picked relative to the SCALE of class_counts, not copied from the paper as-is.
    The paper's typical 0.9-0.9999 assumes instance/image counts in the hundreds to low
    thousands; beta^n underflows to 0 once n is large (pixel counts run into the millions to
    billions here), and once it underflows for every class E_n is identical for all of them and
    the weights silently collapse to uniform, no error raised. For per-pixel counts on this
    dataset (background/wood/knot/pith around 1e9/1e8/1e6/2e5), differentiation only appears
    around beta in [1 - 1e-6, 1 - 1e-8]; see scripts/compute_class_pixel_counts.py for the
    per-run numbers. Prefer inverse_frequency_weights when counts are at pixel scale unless this
    beta range has been checked to still separate the classes.

    Args:
        class_counts: Per-class sample count (pixels or instances), index = class id.
        beta: Effective-number hyperparameter in [0, 1). Needs beta close enough to 1 that
            beta ** max(class_counts) has not underflowed to 0 (see note above), or every
            class gets the same weight.

    Returns:
        torch.Tensor: [C] float32 weights, sum(weights) == num_classes.

    Raises:
        ValueError: If beta is not in [0, 1), any class_counts entry is <= 0, or beta is so far
            from 1 that beta ** count underflows to 0 for the most frequent class (the weights
            would silently collapse to uniform).
    """
    if not 0 <= beta < 1:
        msg = f"beta must be in [0, 1), got {beta}"
        raise ValueError(msg)
    counts = torch.tensor(class_counts, dtype=torch.float64)
    if (counts <= 0).any():
        msg = f"class_counts must all be positive, got {class_counts}"
        raise ValueError(msg)

    if beta == 0.0:
        effective_num = torch.ones_like(counts)
    else:
        beta_tensor = torch.tensor(beta, dtype=torch.float64)
        if torch.pow(beta_tensor, counts.max()) == 0.0:
            msg = (
                f"beta={beta} is too far from 1 for class_counts up to {int(counts.max())}: "
                "beta ** count underflows to 0, so every class gets the same effective number "
                "and the weights silently collapse to uniform. Use a beta closer to 1 (see the "
                "docstring), or inverse_frequency_weights."
            )
            raise ValueError(msg)
        effective_num = (1.0 - torch.pow(beta_tensor, counts)) / (1.0 - beta)
    weights = 1.0 / effective_num
    weights = weights / weights.sum() * len(counts)
    return weights.float()


def inverse_frequency_weights(class_counts: Sequence[int], power: float = 1.0) -> torch.Tensor:
    """Inverse-frequency class weights: weight_c proportional to (1 / n_c) ** power.

    power=1.0 is standard inverse frequency; power=2.0 is inverse-squared, a common (more
    aggressive) variant seen in imbalanced segmentation work. Renormalized to sum to num_classes.

    Args:
        class_counts: Per-class sample count (pixels or instances), index = class id.
        power: Exponent applied to the inverse frequency.

    Returns:
        torch.Tensor: [C] float32 weights, sum(weights) == num_classes.

    Raises:
        ValueError: If any class_counts entry is <= 0.
    """
    counts = torch.tensor(class_counts, dtype=torch.float64)
    if (counts <= 0).any():
        msg = f"class_counts must all be positive, got {class_counts}"
        raise ValueError(msg)
    weights = counts.pow(-power)
    weights = weights / weights.sum() * len(counts)
    return weights.float()


def class_counts_from_masks(masks: List[torch.Tensor], num_classes: int) -> List[int]:
    """Sum per-class pixel counts over a list of integer class-id masks.

    Args:
        masks: Each [H, W] int64 tensor of class ids.
        num_classes: Total number of classes including background.

    Returns:
        List[int]: Pixel count per class, index = class id.
    """
    counts = torch.zeros(num_classes, dtype=torch.int64)
    for mask in masks:
        counts += torch.bincount(mask.flatten(), minlength=num_classes)
    return counts.tolist()
