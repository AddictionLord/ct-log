import base64
import json
from pathlib import Path
from typing import ClassVar
import zlib

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision


class CTLogDataset(Dataset):
    """Serves to load CT log dataset in the Supervisely format.

    Attributes:
        annotations_dir: Directory containing annotation files.
        image_dir: Directory containing image files.
        image_info_dir: Directory containing image info files.

    Args:
        data_dir: _description_

    Raises:
        FileNotFoundError: _description_
        ValueError: If the number of annotations, images, and image info files do not match.
    """

    annotations_dir: str = "ann"
    image_dir: str = "img"
    image_info_dir: str = "img_info"
    class_to_id: ClassVar[dict[str, int]] = {
        "background": 0,
        "compression_wood": 1,
        "crack": 2,
        "insects": 3,
        "knot_sound": 4,
        "moisture": 5,
        "moisture_real": 6,
        "pith": 7,
        "resign_pocket": 8,
        "rot": 9,
        "wood": 10,
    }

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            message = f"Data directory {data_dir} does not exist."
            raise FileNotFoundError(message)

        self.annotation_paths = sorted(self.data_dir.glob(f"{self.annotations_dir}/*.json"))
        self.image_paths = sorted(self.data_dir.glob(f"{self.image_dir}/*.png"))
        self.image_info_paths = sorted(self.data_dir.glob(f"{self.image_info_dir}/*.json"))

        if not len(self.annotation_paths) == len(self.image_paths) == len(self.image_info_paths):
            message = (
                f"Mismatch in number of annotations, images, and image info files: "
                f"{len(self.annotation_paths)}, {len(self.image_paths)}, {len(self.image_info_paths)}"
            )
            raise ValueError(message)

        self.to_tensor = torchvision.transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.annotation_paths)

    def __getitem__(self, idx: int) -> dict[str, Path | torch.Tensor]:
        """Returns a dictionary containing the image and its corresponding annotation.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            A dictionary with keys 'image' and 'annotation'.
        """
        image_path = self.image_paths[idx]
        image = self.to_tensor(Image.open(image_path).convert("RGB"))

        with open(annotation_path := self.annotation_paths[idx], "r") as f:
            annotation = json.load(f)

        mask = torch.zeros(image.shape[1:], dtype=torch.int64)
        for obj in annotation["objects"]:
            if obj["geometryType"] == "point":
                # TODO: Handle point geometry type
                continue

            class_id = self.class_to_id[obj["classTitle"].lower().replace(" ", "_")]
            if class_id == 10:
                continue

            x, y = obj["bitmap"]["origin"]
            bitmap_mask = base64_2_mask(obj["bitmap"]["data"]) * class_id

            # TODO: This overwrites the mask with zeros from bitmap, use torch.where
            mask[y:y+bitmap_mask.shape[0], x:x+bitmap_mask.shape[1]] = bitmap_mask

        return {
            "image": image,
            "path": image_path,
        }


def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return torch.from_numpy(mask)
