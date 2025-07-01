import json
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image
import torch

from src.dataset.ct_log_dataset_base import CTLogDatasetBase
from src.utils.mask import base64_to_mask


class CTLogMaskPreprocessor(CTLogDatasetBase):

    class_priority: ClassVar[list[str]] = [
        "knot_sound",
        "crack",
        "insects",
        "rot",
        "resign_pocket",
        "pith",
        "moisture",
        "moisture_real",
        "compression_wood",
        "wood",
        "background",
    ]

    def __getitem__(self, idx: int) -> dict[str, Path | torch.Tensor]:
        """Returns a dictionary containing the image and its corresponding annotation.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            dict[str, Path | torch.Tensor]:
        """
        image_path = self.image_paths[idx]
        image = self.to_tensor(Image.open(image_path).convert("RGB"))

        with open(annotation_path := self.annotation_paths[idx], "r") as f:
            annotation = json.load(f)

        # TODO: Sorting is not enough, some priority mapper might be needed
        mask = torch.zeros(len(self.class_to_id), *image.shape[1:], dtype=torch.int64)
        for obj in annotation["objects"]:
            if obj["geometryType"] == "point":
                mask = self.draw_point_into_mask(mask, obj)
                continue

            if obj["geometryType"] == "polygon":
                mask = self.draw_polygon_into_mask(mask, obj)
                continue

            if obj["geometryType"] == "bitmap":
                mask = self.draw_bitmap_into_mask(mask, obj)

            else:
                message = f"Unsupported geometry type: {obj['geometryType']}"
                raise ValueError(message)

        mask = self.merge_overlapping_masks(mask)

        return {"image": image, "mask": mask, "path": image_path}

    def draw_point_into_mask(self, mask: torch.Tensor, obj: dict[str, Any]) -> torch.Tensor:
        return mask

    def draw_polygon_into_mask(self, mask: torch.Tensor, obj: dict[str, Any]) -> torch.Tensor:
        return mask

    def draw_bitmap_into_mask(self, mask: torch.Tensor, obj: dict[str, Any]) -> torch.Tensor:
        """Draws a bitmap mask into the provided multi-class mask tensor

        Args:
            mask: [C, H, W] Multi-class mask tensor.
            obj: Object containing bitmap data and class information.

        Returns:
            torch.Tensor: [C, H, W] Multi-class mask tensor with the bitmap drawn in.
        """
        class_id = self.class_to_id[obj["classTitle"].lower().replace(" ", "_")]
        x, y = obj["bitmap"]["origin"]

        bitmap_mask = base64_to_mask(obj["bitmap"]["data"]) * class_id
        mask_slice = mask[class_id, y : y + bitmap_mask.shape[0], x : x + bitmap_mask.shape[1]]

        mask[class_id, y : y + bitmap_mask.shape[0], x : x + bitmap_mask.shape[1]] = torch.where(
            bitmap_mask != 0, bitmap_mask, mask_slice,
        )

        return mask

    def merge_overlapping_masks(self, mask: torch.Tensor) -> torch.Tensor:
        """Creates a composite mask for visualization where higher priority classes override lower ones.

        Args:
            mask: Multi-class mask tensor of shape [C, H, W]

        Returns:
            torch.Tensor: Single channel mask [H, W] with class IDs, prioritized by importance
        """
        priority_map = {self.class_to_id[cls]: idx for idx, cls in enumerate(self.class_priority)}
        composite_mask = torch.zeros(mask.shape[1:], dtype=torch.int64)

        for class_id in range(mask.shape[0]):
            class_mask = mask[class_id] > 0
            if class_mask.any():
                current_priority = priority_map.get(class_id, len(self.class_priority))

                update_mask = class_mask & (composite_mask == 0)
                for existing_val in composite_mask[class_mask].unique():
                    if existing_val == 0:
                        continue
                    existing_class_priority = priority_map.get(int(existing_val), len(self.class_priority))
                    if current_priority < existing_class_priority:
                        update_mask |= (composite_mask == existing_val) & class_mask

                composite_mask[update_mask] = class_id

        return composite_mask
