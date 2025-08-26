import torch
import torch.nn.functional as F


def multiclass_dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: int = 1) -> torch.Tensor:
    """Computes Dice Loss for multi-class segmentation.
    Source: https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68

    Args:
        pred: Tensor of predictions (batch_size, C, H, W).
        target: One-hot encoded ground truth (batch_size, C, H, W).
        smooth: Smoothing factor.

    Returns:
        torch.Tensor: Scalar Dice Loss.
    """
    dice = torch.tensor(0.0)  # Initialize Dice loss accumulator
    num_classes: torch.Tensor = pred.shape[1]  # Number of classes (C)

    pred = F.softmax(pred, dim=1)  # Convert logits to probabilities
    for c in range(num_classes):  # Loop through each class
        pred_c = pred[:, c]  # Predictions for class c
        target_c = target[:, c]  # Ground truth for class c

        intersection = (pred_c * target_c).sum(dim=(1, 2))  # Element-wise multiplication
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))  # Sum of all pixels

        dice += (2. * intersection + smooth) / (union + smooth)  # Per-class Dice score

    return 1 - dice.mean() / num_classes  # Average Dice Loss across classes
