from ann_pipeline.pith.detectors import snap_to_darkest
import numpy as np
import pytest


@pytest.fixture
def slice_with_dark_spot() -> np.ndarray:
    """Uniform 20x20 slice with one dark pixel at column 11, row 7."""
    img = np.full((20, 20), 200, dtype=np.uint8)
    img[7, 11] = 10
    return img


def test_snaps_to_darkest_pixel_in_window(slice_with_dark_spot: np.ndarray) -> None:
    """A dark pixel within the window wins."""
    assert snap_to_darkest(slice_with_dark_spot, 10.4, 8.6, window=5) == (11, 7)


def test_dark_pixel_outside_window_is_ignored(slice_with_dark_spot: np.ndarray) -> None:
    """The search never leaves the window, so a far dark pixel cannot pull the estimate."""
    sx, sy = snap_to_darkest(slice_with_dark_spot, 3.5, 3.5, window=5)
    assert abs(sx - 3) <= 2 and abs(sy - 3) <= 2


def test_floor_convention_centres_window_on_containing_pixel() -> None:
    """x=10.9 lies in pixel 10 (detector pixel i spans [i, i+1)), so a 1x1 window returns 10, not 11."""
    img = np.zeros((20, 20), dtype=np.uint8)
    assert snap_to_darkest(img, 10.9, 4.9, window=1) == (10, 4)


def test_window_is_clipped_at_border() -> None:
    """Estimates at the image edge stay inside the image."""
    img = np.full((10, 10), 100, dtype=np.uint8)
    img[0, 0] = 0
    assert snap_to_darkest(img, 0.2, 0.2, window=5) == (0, 0)


def test_even_window_is_rejected() -> None:
    """An even window has no centre pixel."""
    with pytest.raises(ValueError):
        snap_to_darkest(np.zeros((5, 5)), 2.0, 2.0, window=4)
