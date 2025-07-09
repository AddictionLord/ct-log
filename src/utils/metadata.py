from collections import Counter
import json
from pathlib import Path
from typing import Dict


def save_resolutions(resolutions: Counter[tuple[int, int]], output_path: Path) -> None:
    """Save resolution metadata to a JSON file.

    Args:
        resolutions: Counter containing resolution tuples (height, width) and their counts.
        output_path: Path where the JSON file will be saved.
    """
    with output_path.open("w") as f:
        json.dump(
            {f"{height}x{width}": count for (height, width), count in resolutions.items()},
            f,
            indent=4,
        )


def load_resolutions(input_path: Path) -> Counter[tuple[int, int]]:
    """Load resolution metadata from a JSON file.

    Args:
        input_path: Path to the JSON file containing resolution metadata.

    Returns:
        Counter containing resolution tuples (height, width) and their counts.
    """
    with input_path.open("r") as f:
        resolutions_data: Dict[str, int] = json.load(f)

    loaded_resolutions: Counter[tuple[int, int]] = Counter()
    for resolution_str, count in resolutions_data.items():
        height, width = map(int, resolution_str.split("x"))
        loaded_resolutions[(height, width)] = count

    return loaded_resolutions
