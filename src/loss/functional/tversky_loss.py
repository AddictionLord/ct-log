import torch
import torch.nn.functional as F


def multiclass_tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: int = 1,
    ignore_background: bool = False,
) -> torch.Tensor:
    """Computes Tversky Loss for multi-class segmentation.

    Tversky loss is a generalization of Dice loss that allows for controlling
    false positives and false negatives separately using alpha and beta parameters.
    When alpha=beta=0.5, it reduces to Dice loss.

    Args:
        pred: Tensor of predictions (batch_size, C, H, W).
        target: One-hot encoded ground truth (batch_size, C, H, W).
        alpha: Weight for false positives (0 to 1).
        beta: Weight for false negatives (0 to 1).
        smooth: Smoothing factor.
        ignore_background: If True, class index 0 is excluded from the average.

    Returns:
        torch.Tensor: Scalar Tversky Loss.
    """
    tversky = torch.tensor(0.0, device=pred.device)
    num_classes: int = pred.shape[1]
    start_class = 1 if ignore_background else 0
    counted_classes = num_classes - start_class

    pred = F.softmax(pred, dim=1)

    for c in range(start_class, num_classes):
        pred_c = pred[:, c]
        target_c = target[:, c]

        true_positives = (pred_c * target_c).sum(dim=(1, 2))
        false_positives = (pred_c * (1 - target_c)).sum(dim=(1, 2))
        false_negatives = ((1 - pred_c) * target_c).sum(dim=(1, 2))

        tversky_index = (true_positives + smooth) / (
            true_positives + alpha * false_positives + beta * false_negatives + smooth
        )
        tversky += tversky_index.mean()

    return 1 - tversky.mean() / counted_classes
