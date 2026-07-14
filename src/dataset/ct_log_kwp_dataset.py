import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from src.dataset.kwp_mask import KwpMaskBuilder
import torch
from torchvision import transforms

PAGE_RE = re.compile(r"page_(\d+)")


class CTLogKwpDataset(torch.utils.data.Dataset):
    """2.5D knot/wood/pith segmentation dataset over full contiguous CT logs.

    Each item is a center slice plus its axial neighbors stacked into the channel
    dimension. With ``window=1`` the channels are ``[z-1, z, z+1]`` (Option A).
    With ``window=0`` the center slice is replicated across all three channels
    (the single-slice DINOv3 baseline). Neighbors are resolved against the slices
    actually present on disk and clamped at volume boundaries / index gaps.

    Expected layout (one directory per log)::

        <log_dir>/
            img/page_<idx>.tiff
            ann/page_<idx>.tiff.json   # Supervisely annotation, knot/wood/pith
    """

    img_dir: str = "img"
    ann_dir: str = "ann"

    def __init__(
        self,
        log_dirs: List[str | Path],
        resolution: Optional[Tuple[int, int]] = None,
        window: int = 1,
        pith_radius: int = 3,
    ) -> None:
        """Initialize the dataset.

        Args:
            log_dirs: One directory per log, each with img/ and ann/ subdirectories.
            resolution: Target (height, width) for images and masks, or None for native.
            window: Number of neighbor slices on each side. 1 => [z-1, z, z+1] 2.5D,
                0 => center slice replicated 3x (single-slice baseline).
            pith_radius: Radius of the rasterized pith point blob.
        """
        if window not in (0, 1):
            message = f"window must be 0 or 1 to fit three channels, got {window}"
            raise ValueError(message)

        self.resolution = resolution
        self.window = window
        self.mask_builder = KwpMaskBuilder(pith_radius=pith_radius)
        self.to_tensor = transforms.ToTensor()

        self.resize_image = self._make_resize(resolution, transforms.InterpolationMode.BILINEAR)
        self.resize_mask = self._make_resize(resolution, transforms.InterpolationMode.NEAREST)

        self.samples: List[Dict[str, Any]] = []
        for log_dir in log_dirs:
            self._index_log(Path(log_dir))

        if not self.samples:
            message = f"No annotated slices found under {log_dirs}"
            raise ValueError(message)

    def _index_log(self, log_dir: Path) -> None:
        img_dir = log_dir / self.img_dir
        ann_dir = log_dir / self.ann_dir
        if not img_dir.exists():
            message = f"Missing image directory {img_dir}"
            raise FileNotFoundError(message)

        index_to_path: Dict[int, Path] = {}
        for path in img_dir.glob("*.tiff"):
            match = PAGE_RE.search(path.stem)
            if match:
                index_to_path[int(match.group(1))] = path

        sorted_indices = sorted(index_to_path)
        for position, index in enumerate(sorted_indices):
            ann_path = ann_dir / f"{index_to_path[index].name}.json"
            if not ann_path.exists():
                continue
            self.samples.append(
                {
                    "center": index_to_path[index],
                    "ann": ann_path,
                    "neighbors": self._resolve_neighbors(sorted_indices, index_to_path, position),
                }
            )

    def _resolve_neighbors(
        self,
        sorted_indices: List[int],
        index_to_path: Dict[int, Path],
        position: int,
    ) -> List[Path]:
        if self.window == 0:
            return []

        prev_position = max(position - 1, 0)
        next_position = min(position + 1, len(sorted_indices) - 1)
        return [
            index_to_path[sorted_indices[prev_position]],
            index_to_path[sorted_indices[next_position]],
        ]

    @staticmethod
    def _make_resize(
        resolution: Optional[Tuple[int, int]],
        interpolation: transforms.InterpolationMode,
    ) -> torch.nn.Module:
        if resolution is None:
            return torch.nn.Identity()
        return transforms.Resize(resolution, interpolation=interpolation)

    def _load_slice(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("L")
        return self.to_tensor(image)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a stacked-neighbor image and its center-slice 3-class mask.

        Args:
            idx: Sample index.

        Returns:
            dict: keys ``image`` [3, H, W] float32, ``mask`` [H, W] int64,
                ``path`` center-slice path string.
        """
        sample = self.samples[idx]
        center = self._load_slice(sample["center"])

        if self.window == 0:
            channels = center.repeat(3, 1, 1)
        else:
            prev_slice = self._load_slice(sample["neighbors"][0])
            next_slice = self._load_slice(sample["neighbors"][1])
            channels = torch.cat([prev_slice, center, next_slice], dim=0)

        image = self.resize_image(channels)

        with sample["ann"].open("r") as f:
            annotation = json.load(f)
        mask = self.mask_builder.build(annotation).unsqueeze(0)
        mask = self.resize_mask(mask).squeeze(0)

        return {"image": image, "mask": mask, "path": str(sample["center"])}
