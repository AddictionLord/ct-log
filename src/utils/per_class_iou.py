from typing import Dict

import torch


class PerClassIoU:
    """Accumulates per-class intersection/union over batches for IoU.

    Unlike a macro mean that includes background, this exposes each class's IoU
    separately so rare classes (knot, pith) stay visible.
    """

    def __init__(self, num_classes: int) -> None:
        """Initialize the accumulator.

        Args:
            num_classes: Total number of classes including background.
        """
        self.num_classes = num_classes
        self.intersection = torch.zeros(num_classes, dtype=torch.float64)
        self.union = torch.zeros(num_classes, dtype=torch.float64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate counts from a batch of predictions and targets.

        Args:
            preds: [B, H, W] int64 predicted class ids.
            targets: [B, H, W] int64 ground-truth class ids.
        """
        preds = preds.flatten()
        targets = targets.flatten()
        for class_id in range(self.num_classes):
            pred_c = preds == class_id
            target_c = targets == class_id
            self.intersection[class_id] += (pred_c & target_c).sum().item()
            self.union[class_id] += (pred_c | target_c).sum().item()

    def compute(self) -> Dict[str, float]:
        """Compute per-class IoU and the foreground mean (excluding background).

        Returns:
            Dict[str, float]: iou_<class_id> for each class plus mean_iou_fg.
        """
        iou = self.intersection / self.union.clamp(min=1.0)
        result = {f"iou_{class_id}": float(iou[class_id].item()) for class_id in range(self.num_classes)}
        if self.num_classes > 1:
            result["mean_iou_fg"] = float(iou[1:].mean().item())
        return result
