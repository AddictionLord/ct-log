from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING
import warnings

import torch
import torchvision
from tqdm import tqdm

from src.dataset.ct_log_mask_preprocessor import CTLogMaskPreprocessor

if TYPE_CHECKING:
    from PIL.Image import Image


def preprocess_dataset(src_dir: Path, out_dir: Path) -> None:
    """_summary_

    Args:
        src_dir: _description_
        out_dir: _description_
        workers: _description_. Defaults to 2.
    """
    to_pil_image = torchvision.transforms.ToPILImage()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = CTLogMaskPreprocessor(data_dir=src_dir)

    for _, batch in enumerate(tqdm(dataset, desc="Processing dataset", unit="item")):
        mask, path = batch["mask"], Path(batch["path"])

        if batch["pith"] is None:
            message = f"Pith is None for image {path.name}. The annotation may be missing or incomplete."
            warnings.warn(message)

        mask_image: Image = to_pil_image(mask.to(torch.float32))
        mask_image.save(out_dir / path.name)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--source_data_dir",
        type=Path,
        default="data/raw/set_24",
        help="Directory containing the dataset.",
    )
    parser.add_argument(
        "--output_data_dir",
        type=Path,
        default="data/processed/set_24",
        help="Directory to save the processed dataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of worker threads for data loading.",
    )
    args = parser.parse_args()

    preprocess_dataset(args.source_data_dir, args.output_data_dir / "mask")


if __name__ == "__main__":
    main()
