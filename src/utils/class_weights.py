import torch


class ClassWeightTracker:
    """Tracks class frequencies and computes weights for loss functions."""

    def __init__(
        self,
        num_classes: int,
        mode: str = "online",
        ema_decay: float = 0.9,
        min_weight: float = 0.5,
        max_weight: float = 50.0,
    ) -> None:
        """Initialize the class weight tracker.

        Args:
            num_classes: Number of classes.
            mode: "online" for per-batch EMA, "offline" for fixed weights.
            ema_decay: Decay factor for EMA smoothing (online mode).
            min_weight: Minimum weight clamp value.
            max_weight: Maximum weight clamp value.
        """
        self.num_classes = num_classes
        self.mode = mode
        self.ema_decay = ema_decay
        self.min_weight = min_weight
        self.max_weight = max_weight
        self._running_counts: torch.Tensor | None = None
        self._fixed_weights: torch.Tensor | None = None

    def set_fixed_weights(self, weights: list[float]) -> None:
        """Set fixed weights for offline mode.

        Args:
            weights: List of class weights.
        """
        self._fixed_weights = torch.tensor(weights, dtype=torch.float32)

    def update(self, masks: torch.Tensor) -> None:
        """Update running counts from batch masks (online mode only).

        Args:
            masks: Ground truth masks (batch_size, H, W).
        """
        if self.mode != "online":
            return

        batch_counts = torch.bincount(masks.flatten(), minlength=self.num_classes).float()

        if self._running_counts is None:
            self._running_counts = batch_counts.to(masks.device)
        else:
            self._running_counts = self.ema_decay * self._running_counts + (1 - self.ema_decay) * batch_counts.to(
                masks.device
            )

    def get_weights(self, device: torch.device) -> torch.Tensor | None:
        """Get current class weights.

        Args:
            device: Device to place weights on.

        Returns:
            Class weights tensor or None if not ready.
        """
        if self.mode == "offline":
            if self._fixed_weights is None:
                return None
            return self._fixed_weights.to(device)

        if self._running_counts is None:
            return None

        weights = 1.0 / (self._running_counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes
        weights = weights.clamp(min=self.min_weight, max=self.max_weight)
        return weights.to(device)
