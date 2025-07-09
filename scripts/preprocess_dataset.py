from argparse import ArgumentParser
from collections import Counter
import logging
from pathlib import Path
import warnings

import torch
import torchvision
from tqdm import tqdm

from src.dataset.ct_log_mask_preprocessor import CTLogMaskPreprocessor
from src.utils.metadata import save_resolutions


def preprocess_dataset(src_dir: Path, out_dir: Path) -> Counter[tuple[int, int]]:
    """Preprocess the dataset by constructing and converting masks to PIL images and saving them.

    Args:
        src_dir: Path to the source dataset directory containing CT images and masks.
        out_dir: Path to the output directory where processed masks will be saved.
    """
    to_pil_image = torchvision.transforms.ToPILImage()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = CTLogMaskPreprocessor(data_dir=src_dir)

    resolutions: Counter[tuple[int, int]] = Counter()
    for _, batch in enumerate(tqdm(dataset, desc="Processing dataset", unit="item")):
        mask, path = batch["mask"], Path(batch["path"])

        height, width = mask.shape
        resolutions[(height, width)] += 1

        if batch["pith"] is None:
            message = f"Pith is None for image {path.name}. The annotation may be missing or incomplete."
            warnings.warn(message)

        mask_image = to_pil_image(mask.to(torch.uint8))
        mask_image.save(out_dir / path.name)

    return resolutions


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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Source data directory: %s", args.source_data_dir)
    logger.info("Output data directory: %s", args.output_data_dir)

    preprocess_dataset(args.source_data_dir, out_path := (args.output_data_dir / "mask"))

    logger.info("Dataset preprocessing completed. Output saved to %s", out_path)


if __name__ == "__main__":
    main()
