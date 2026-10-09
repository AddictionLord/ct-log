from src.dataset.kwp_mask import KwpMaskBuilder
import torch


def test_pith_is_regression_only() -> None:
    """Pith stays available as coordinates without entering the segmentation mask."""
    annotation = {
        "size": {"height": 10, "width": 20},
        "objects": [
            {
                "classTitle": "pith",
                "geometryType": "point",
                "points": {"exterior": [[5, 4]]},
            }
        ],
    }
    builder = KwpMaskBuilder()

    mask = builder.build(annotation)
    pith_xy = builder.pith_xy_normalized(annotation)

    assert torch.equal(mask, torch.zeros((10, 20), dtype=torch.int64))
    assert torch.allclose(pith_xy, torch.tensor([0.25, 0.4, 1.0]))
