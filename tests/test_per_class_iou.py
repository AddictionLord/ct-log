import pytest
from src.dataset.kwp_mask import KwpMaskBuilder
from src.utils.per_class_iou import PerClassIoU
import torch


def test_uses_class_names_as_keys() -> None:
    """Metric keys carry the class names instead of ids."""
    iou = PerClassIoU(num_classes=3, class_names=KwpMaskBuilder.class_names())
    iou.update(torch.tensor([[0, 1, 2, 0]]), torch.tensor([[0, 1, 2, 2]]))

    assert iou.compute() == {
        "iou_background": 0.5,
        "iou_wood": 1.0,
        "iou_knot": 0.5,
        "mean_iou_fg": 0.75,
    }


def test_falls_back_to_ids() -> None:
    """Without names the keys stay iou_<id>."""
    assert set(PerClassIoU(num_classes=2).compute()) == {"iou_0", "iou_1", "mean_iou_fg"}


def test_rejects_wrong_number_of_names() -> None:
    """A name list that does not match num_classes fails loudly."""
    with pytest.raises(ValueError, match="Expected 4 class names"):
        PerClassIoU(num_classes=4, class_names=["background", "wood"])
