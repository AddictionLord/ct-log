import torch
import torch.nn.functional as F


def multiclass_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Computes Focal Loss for multi-class segmentation.

    Focal loss addresses class imbalance by down-weighting easy examples and
    focusing on hard examples. It modifies cross-entropy loss with a modulating
    factor (1-p)^gamma where p is the predicted probability for the true class.

    Args:
        pred: Tensor of predictions (batch_size, C, H, W).
        target: Ground truth class indices (batch_size, H, W).
        alpha: Weighting factor for rare class (default: 1.0).
        gamma: Focusing parameter, higher values focus more on hard examples (default: 2.0).

    Returns:
        torch.Tensor: Scalar Focal Loss.
    """
    # Apply softmax to get probabilities
    pred_probs = F.softmax(pred, dim=1)

    # Compute cross entropy loss without reduction
    ce_loss = F.cross_entropy(pred, target, reduction="none")

    # Get the probability of the true class for each pixel
    # target is (batch_size, H, W), we need to gather the predicted probabilities
    # for the true class at each pixel
    target_expanded = target.unsqueeze(1)  # (batch_size, 1, H, W)
    p_t = pred_probs.gather(1, target_expanded).squeeze(1)  # (batch_size, H, W)

    # Compute focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma

    # Apply focal weight and alpha
    focal_loss = alpha * focal_weight * ce_loss

    return focal_loss.mean()