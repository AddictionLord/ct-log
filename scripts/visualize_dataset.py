from pathlib import Path

import numpy as np
import plotly.express as px
import torch

from src.dataset.ct_log_dataset import CTLogDataset


def render_mask_on_image(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.5) -> np.ndarray:
    """Render mask overlay on image with transparency.

    Args:
        image: RGB image tensor of shape (3, H, W)
        mask: Mask tensor of shape (H, W) with class indices
        alpha: Transparency level for mask overlay (0.0 to 1.0)

    Returns:
        Combined image as numpy array of shape (H, W, 3)
    """
    img_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.numpy()

    mask_colored = np.zeros((*mask_np.shape, 3))
    unique_classes = np.unique(mask_np)

    colors = [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 0.5, 0.0],
        [0.5, 0.0, 1.0], [0.0, 0.5, 0.5]
    ]

    for i, class_id in enumerate(unique_classes):
        if class_id == 0:
            continue
        color_idx = (i - 1) % len(colors)
        mask_colored[mask_np == class_id] = colors[color_idx]

    combined = img_np * (1 - alpha) + mask_colored * alpha
    return np.clip(combined, 0, 1)


def main() -> None:
    """Visualize dataset images and masks, displaying and saving them as PNG files."""
    dataset = CTLogDataset(data_dir="data/processed/set_24", num_classes=10, resolution=(458, 530))

    output_dir = Path("data/processed/visualizations")
    output_dir.mkdir(exist_ok=True)

    print(f"Dataset length: {len(dataset)}")

    for i in range(len(dataset)):
        item = dataset[i]

        base_name = Path(item["path"]).stem

        img_fig = px.imshow(item["image"].permute(1, 2, 0), title=str(item["path"]))
        img_fig.show()
        img_fig.write_image(output_dir / f"{base_name}_image.png")

        mask_fig = px.imshow(item["mask"], title=str(item["path"]))
        mask_fig.show()
        mask_fig.write_image(output_dir / f"{base_name}_mask.png")

        combined_image = render_mask_on_image(item["image"], item["mask"])
        combined_fig = px.imshow(combined_image, title=f"{item['path']} - Overlay")
        combined_fig.show()
        combined_fig.write_image(output_dir / f"{base_name}_overlay.png")

        break


if __name__ == "__main__":
    main()
