import pytest
from src.utils.class_balance import (
    capped_inverse_frequency_weights,
    class_counts_from_masks,
    effective_number_weights,
    inverse_frequency_weights,
)
import torch


def test_uniform_when_beta_zero() -> None:
    """beta=0 disables reweighting: every class gets weight 1."""
    weights = effective_number_weights([1000, 100, 10, 1], beta=0.0)

    assert torch.allclose(weights, torch.ones(4))


def test_upweights_rare_classes() -> None:
    """A rarer class gets a strictly larger weight."""
    weights = effective_number_weights([1000, 100, 10, 1], beta=0.99)

    assert weights[0] < weights[1] < weights[2] < weights[3]


def test_sums_to_num_classes() -> None:
    """Weights are renormalized to keep the loss on the same scale."""
    weights = effective_number_weights([500, 50, 5], beta=0.999)

    assert weights.sum().item() == pytest.approx(3.0, abs=1e-4)


def test_higher_beta_is_more_aggressive() -> None:
    """A higher beta pushes the rare-class weight further from uniform."""
    low = effective_number_weights([1000, 10], beta=0.9)
    high = effective_number_weights([1000, 10], beta=0.9999)

    assert high[1] > low[1]


def test_rejects_beta_out_of_range() -> None:
    """beta must be in [0, 1)."""
    with pytest.raises(ValueError, match="beta must be in"):
        effective_number_weights([10, 5], beta=1.0)


def test_rejects_zero_count() -> None:
    """A class with zero samples would divide by zero in the effective number."""
    with pytest.raises(ValueError, match="must all be positive"):
        effective_number_weights([10, 0], beta=0.99)


def test_inverse_frequency_matches_power_one() -> None:
    """power=1.0 is plain inverse frequency, renormalized."""
    weights = inverse_frequency_weights([100, 10, 1], power=1.0)

    assert weights[0] < weights[1] < weights[2]
    assert weights.sum().item() == pytest.approx(3.0, abs=1e-4)


def test_inverse_squared_is_more_aggressive_than_linear() -> None:
    """power=2.0 (inverse squared) separates classes further than power=1.0."""
    linear = inverse_frequency_weights([1000, 10], power=1.0)
    squared = inverse_frequency_weights([1000, 10], power=2.0)

    assert squared[1] / squared[0] > linear[1] / linear[0]


def test_class_counts_from_masks_sums_pixels() -> None:
    """Counts accumulate across multiple masks, indexed by class id."""
    masks = [torch.tensor([[0, 1, 2]]), torch.tensor([[0, 0, 1]])]

    assert class_counts_from_masks(masks, num_classes=3) == [3, 2, 1]


def test_rejects_beta_that_underflows_for_large_counts() -> None:
    """A beta too far from 1 for pixel-scale counts would silently collapse to uniform weights."""
    with pytest.raises(ValueError, match="underflows to 0"):
        effective_number_weights([3_226_729_598, 388_031_918, 5_227_266, 214_822], beta=0.9999)


def test_pixel_scale_counts_differentiate_at_high_beta() -> None:
    """The rarer classes get more weight once beta is close enough to 1 for these magnitudes.

    background and wood (1e9 and 4e8 px) still tie at this beta: only knot and pith (1e6 and
    2e5 px) are low enough to differentiate. A still higher beta is needed to also separate
    background from wood; see the module docstring.
    """
    weights = effective_number_weights([3_226_729_598, 388_031_918, 5_227_266, 214_822], beta=1 - 1e-7)

    assert weights[3] > weights[2] > weights[1] == weights[0]


def test_capped_inverse_frequency_floor_one_is_uniform() -> None:
    """floor_share=1.0 clamps every class to the total count."""
    weights = capped_inverse_frequency_weights([1000, 100, 10], floor_share=1.0)

    assert torch.allclose(weights, torch.ones(3))


def test_capped_inverse_frequency_limits_tiny_class_ratio() -> None:
    """The floor limits a tiny class to the configured maximum weight ratio."""
    weights = capped_inverse_frequency_weights([1000, 10], floor_share=0.1)

    assert (weights[1] / weights[0]).item() <= 10.0 + 1e-6


@pytest.mark.parametrize("floor_share", [0.0, -0.1, 1.01])
def test_capped_inverse_frequency_rejects_invalid_floor_share(floor_share: float) -> None:
    """floor_share must be greater than zero and no greater than one."""
    with pytest.raises(ValueError, match="floor_share must be in"):
        capped_inverse_frequency_weights([100, 10], floor_share=floor_share)


def test_capped_inverse_frequency_sums_to_num_classes() -> None:
    """Capped weights retain the normalization invariant."""
    weights = capped_inverse_frequency_weights([500, 50, 5], floor_share=0.01)

    assert weights.sum().item() == pytest.approx(3.0, abs=1e-4)


def test_capped_inverse_frequency_real_counts_are_bounded_and_ordered() -> None:
    """Measured KWP counts stay ordered while respecting the configured ratio cap."""
    weights = capped_inverse_frequency_weights(
        [3_226_729_598, 388_031_918, 5_227_266, 214_822],
        floor_share=0.01,
    )

    assert (weights.max() / weights.min()).item() <= 100.0 + 1e-4
    assert weights[3] >= weights[2] >= weights[1] >= weights[0]
