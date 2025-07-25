from pathlib import Path
from typing import Any, ClassVar

from PIL import Image, ImageDraw
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

    def __getitem__(self, idx: int) -> dict[str, Path | torch.Tensor | None]:
        """Returns a dictionary containing the image and its corresponding annotation.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            dict[str, Path | torch.Tensor]:
        """
        data = super().__getitem__(idx)

        pith = None
        mask = torch.zeros(len(self.class_to_id), *data["image"].shape[1:], dtype=torch.int64)
        for obj in data["annotation"]["objects"]:
            if obj["geometryType"] == "point":
                mask = self.draw_point_into_mask(mask, obj)
                pith = torch.tensor(obj["points"]["exterior"][0])

            elif obj["geometryType"] == "polygon":
                mask = self.draw_polygon_into_mask(mask, obj)

            elif obj["geometryType"] == "bitmap":
                mask = self.draw_bitmap_into_mask(mask, obj)

            else:
                message = f"Unsupported geometry type: {obj['geometryType']}"
                raise ValueError(message)

        data.update({"mask": self.merge_overlapping_masks(mask), "pith": pith})

        return data

    def draw_point_into_mask(self, mask: torch.Tensor, obj: dict[str, Any], blob_radius: int = 3) -> torch.Tensor:
        """Draws a point into the provided multi-class mask tensor.

        Args:
            mask: [C, H, W] Multi-class mask tensor.
            obj: Object containing point data and class information.
            blob_radius: Radius of the point to be drawn.

        Returns:
            torch.Tensor: [C, H, W] Multi-class mask tensor with the point drawn in.
        """
        points = obj["points"]["exterior"]
        class_id: int = self.class_to_id[obj["classTitle"].lower().replace(" ", "_")]

        height, width = mask.shape[1], mask.shape[2]
        pil_mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(pil_mask)

        for x, y in points:
            draw.ellipse([x - blob_radius, y - blob_radius, x + blob_radius, y + blob_radius], fill=1)

            point_tensor = torch.tensor(list(pil_mask.getdata())).reshape(height, width)
            mask[class_id] = torch.where(point_tensor > 0, class_id, mask[class_id])

        return mask

    def draw_polygon_into_mask(self, mask: torch.Tensor, obj: dict[str, Any]) -> torch.Tensor:
        """Draws a polygon into the provided multi-class mask tensor.

        Args:
            mask: [C, H, W] Multi-class mask tensor.
            obj: Object containing polygon data and class information.

        Returns:
            torch.Tensor: [C, H, W] Multi-class mask tensor with the polygon drawn in.
        """
        polygon: list[list[int]] = obj["points"]["exterior"]
        class_id: int = self.class_to_id[obj["classTitle"].lower().replace(" ", "_")]

        height, width = mask.shape[1], mask.shape[2]
        pil_mask = Image.new("L", (width, height), 0)

        draw = ImageDraw.Draw(pil_mask)
        flat_points = [coord for point in polygon for coord in point]
        draw.polygon(flat_points, fill=1)

        polygon_tensor = torch.tensor(list(pil_mask.getdata())).reshape(height, width)
        mask[class_id] = torch.where(polygon_tensor > 0, class_id, mask[class_id])

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
        composite_mask: torch.Tensor = torch.zeros(mask.shape[1:], dtype=torch.int64)

        for class_id in range(mask.shape[0]):
            class_mask = mask[class_id] > 0
            if class_mask.any():
                current_priority = priority_map.get(class_id, len(self.class_priority))

                update_mask: torch.Tensor = class_mask & (composite_mask == 0)
                for existing_val in composite_mask[class_mask].unique(): # type: ignore[reportUnknownVariableType]
                    if existing_val == 0 or not isinstance(existing_val, torch.Tensor):
                        continue

                    existing_class_priority = priority_map.get(int(existing_val), len(self.class_priority))
                    if current_priority < existing_class_priority:
                        update_mask |= (composite_mask == existing_val) & class_mask

                composite_mask[update_mask] = class_id

        return composite_mask
