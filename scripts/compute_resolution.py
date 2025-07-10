import argparse
import json
import logging


def find_square_resolution_near_mean(resolutions_file: str, logger: logging.Logger) -> tuple[int, int]:
    """Find a square resolution close to the mean of existing resolutions.

    Args:
        resolutions_file: Path to the JSON file containing resolutions
        logger: Logger instance for logging information

    Returns:
        tuple[int, int]: Tuple of (width, height) for the square resolution
    """
    with open(resolutions_file, "r") as f:
        resolutions = json.load(f)

    total_width = 0
    total_height = 0
    total_count = 0

    for resolution_str, count in resolutions.items():
        width, height = map(int, resolution_str.split("x"))
        total_width += width * count
        total_height += height * count
        total_count += count

    mean_width = total_width / total_count
    mean_height = total_height / total_count

    mean_dimension = (mean_width + mean_height) / 2
    square_size = round(mean_dimension)

    logger.info("Mean width: %.1f", mean_width)
    logger.info("Mean height: %.1f", mean_height)
    logger.info("Mean dimension: %.1f", mean_dimension)
    logger.info("Recommended non-square resolution: %dx%d", round(mean_width), round(mean_height))
    logger.info("Recommended square resolution: %dx%d", square_size, square_size)

    return (square_size, square_size)


def main() -> None:
    """Main function to execute the resolution computation."""
    parser = argparse.ArgumentParser("Find square resolution near mean of existing resolutions")
    parser.add_argument(
        "resolutions_file",
        nargs="?",
        default="data/processed/set_24/resolutions.json",
        help="Path to the JSON file containing resolutions",
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    find_square_resolution_near_mean(args.resolutions_file, logger)


if __name__ == "__main__":
    main()
