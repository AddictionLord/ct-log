from typing import Any, Dict, List, Optional

import torch


class WeightEma:
    """Exponential moving average of the trainable weights, swapped in for evaluation.

    Tracks every floating-point parameter and buffer of the heads (BatchNorm running statistics
    included, as in timm's ModelEmaV2) plus the fine-tuned backbone parameters. ``swap`` exchanges
    live and averaged tensors in place, so calling it twice restores the training weights.
    """

    def __init__(
        self,
        modules: List[torch.nn.Module],
        backbone_params: Optional[Dict[str, torch.nn.Parameter]],
        decay: float,
    ) -> None:
        """Initialize the shadows from the current weights.

        Args:
            modules: Fully trained modules (segmentation head, optional pith head).
            backbone_params: Fine-tuned backbone parameters, or None/empty for a frozen backbone.
            decay: Target EMA decay; the effective decay ramps up as min(decay, (1 + t) / (10 + t)).

        Raises:
            ValueError: If decay is not in (0, 1).
        """
        if not 0.0 < decay < 1.0:
            msg = f"EMA decay must be in (0, 1), got {decay}"
            raise ValueError(msg)
        self.decay = decay
        self.step = 0
        self.live: List[torch.Tensor] = []
        for module in modules:
            self.live.extend(t for t in module.state_dict(keep_vars=True).values() if t.is_floating_point())
        self.live.extend((backbone_params or {}).values())
        self.shadow = [t.detach().clone() for t in self.live]

    @torch.no_grad()
    def update(self) -> None:
        """Blend the current weights into the shadows after an optimizer step."""
        self.step += 1
        decay = min(self.decay, (1 + self.step) / (10 + self.step))
        for live, shadow in zip(self.live, self.shadow):
            shadow.mul_(decay).add_(live.detach(), alpha=1.0 - decay)

    @torch.no_grad()
    def swap(self) -> None:
        """Exchange live and averaged weights in place."""
        for live, shadow in zip(self.live, self.shadow):
            buffer = live.detach().clone()
            live.data.copy_(shadow)
            shadow.copy_(buffer)

    def state_dict(self) -> Dict[str, Any]:
        """Serializable state for resume.

        Returns:
            Dict[str, Any]: step counter and shadow tensors in tracking order.
        """
        return {"step": self.step, "shadow": [t.cpu() for t in self.shadow]}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the step counter and shadows saved by ``state_dict``.

        Args:
            state: Output of ``state_dict``.

        Raises:
            ValueError: If the number of tracked tensors differs.
        """
        if len(state["shadow"]) != len(self.shadow):
            msg = f"EMA state has {len(state['shadow'])} tensors, expected {len(self.shadow)}"
            raise ValueError(msg)
        self.step = state["step"]
        for shadow, saved in zip(self.shadow, state["shadow"]):
            shadow.copy_(saved.to(shadow.device))
