from typing import Any, ClassVar, Dict, List

from PIL import Image, ImageDraw
from src.utils.mask import base64_to_mask
import torch


class KwpMaskBuilder:
    """Builds single-channel 3-class masks (knot / wood / pith) from Supervisely annotations.

    Class ids:
        0 = background
        1 = wood
        2 = knot
        3 = pith

    Higher-priority classes overwrite lower ones where they overlap. Pith (a point)
    sits on top of knot, which sits on top of wood.
    """

    class_to_id: ClassVar[Dict[str, int]] = {
        "background": 0,
        "wood": 1,
        "knot": 2,
        "pith": 3,
    }
    draw_order: ClassVar[List[str]] = ["wood", "knot", "pith"]

    @classmethod
    def class_names(cls) -> List[str]:
        """Class names ordered by class id.

        Returns:
            List[str]: Name of each class, index = class id.
        """
        return sorted(cls.class_to_id, key=cls.class_to_id.get)

    def __init__(self, pith_radius: int = 3) -> None:
        self.pith_radius = pith_radius

    @staticmethod
    def pith_xy_normalized(annotation: Dict[str, Any]) -> torch.Tensor:
        """Extract the pith point as normalized (x, y), resolution-independent.

        Args:
            annotation: Supervisely image annotation dict.

        Returns:
            torch.Tensor: [3] tensor (x, y, valid); valid is 0.0 when the slice
                has no pith annotation, in which case x and y are 0.
        """
        height = annotation["size"]["height"]
        width = annotation["size"]["width"]
        for obj in annotation["objects"]:
            if obj["classTitle"].lower().replace(" ", "_") == "pith":
                x, y = obj["points"]["exterior"][0]
                return torch.tensor([x / width, y / height, 1.0], dtype=torch.float32)
        return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    def build(self, annotation: Dict[str, Any]) -> torch.Tensor:
        """Rasterize a Supervisely annotation into a [H, W] int64 class-id mask.

        Args:
            annotation: Supervisely image annotation dict with "size" and "objects".

        Returns:
            torch.Tensor: [H, W] int64 mask of class ids.
        """
        height = annotation["size"]["height"]
        width = annotation["size"]["width"]
        mask = torch.zeros((height, width), dtype=torch.int64)

        by_class: Dict[str, List[Dict[str, Any]]] = {name: [] for name in self.draw_order}
        for obj in annotation["objects"]:
            name = obj["classTitle"].lower().replace(" ", "_")
            if name in by_class:
                by_class[name].append(obj)

        for name in self.draw_order:
            class_id = self.class_to_id[name]
            for obj in by_class[name]:
                mask = self._draw_object(mask, obj, class_id)

        return mask

    def _draw_object(self, mask: torch.Tensor, obj: Dict[str, Any], class_id: int) -> torch.Tensor:
        geometry_type = obj["geometryType"]
        if geometry_type == "bitmap":
            return self._draw_bitmap(mask, obj, class_id)
        if geometry_type == "point":
            return self._draw_point(mask, obj, class_id)
        if geometry_type == "polygon":
            return self._draw_polygon(mask, obj, class_id)

        message = f"Unsupported geometry type: {geometry_type}"
        raise ValueError(message)

    def _draw_bitmap(self, mask: torch.Tensor, obj: Dict[str, Any], class_id: int) -> torch.Tensor:
        x, y = obj["bitmap"]["origin"]
        bitmap = base64_to_mask(obj["bitmap"]["data"])
        height, width = bitmap.shape
        region = mask[y : y + height, x : x + width]
        mask[y : y + height, x : x + width] = torch.where(bitmap, class_id, region)
        return mask

    def _draw_point(self, mask: torch.Tensor, obj: Dict[str, Any], class_id: int) -> torch.Tensor:
        height, width = mask.shape
        pil_mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(pil_mask)
        radius = self.pith_radius
        for x, y in obj["points"]["exterior"]:
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=1)

        point_tensor = torch.tensor(list(pil_mask.getdata()), dtype=torch.int64).reshape(height, width)
        return torch.where(point_tensor > 0, class_id, mask)

    def _draw_polygon(self, mask: torch.Tensor, obj: Dict[str, Any], class_id: int) -> torch.Tensor:
        height, width = mask.shape
        pil_mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(pil_mask)
        flat_points = [coord for point in obj["points"]["exterior"] for coord in point]
        draw.polygon(flat_points, fill=1)

        polygon_tensor = torch.tensor(list(pil_mask.getdata()), dtype=torch.int64).reshape(height, width)
        return torch.where(polygon_tensor > 0, class_id, mask)
