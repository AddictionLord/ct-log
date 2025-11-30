from pathlib import Path

import numpy as np
import plotly.express as px
import torch

from src.dataset.ct_log_dataset import CTLogDataset

try:
    import matplotlib.pyplot as plt

    matplotlib_available = True
except ImportError:
    matplotlib_available = False

# Color thresholds for fallback viridis approximation
QUARTER_THRESHOLD = 0.25
HALF_THRESHOLD = 0.5
THREE_QUARTER_THRESHOLD = 0.75


def render_mask_on_image(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.5) -> np.ndarray:
    """Render mask overlay on image with transparency using plotly's exact colors.

    Args:
        image: RGB image tensor of shape (3, H, W)
        mask: Mask tensor of shape (H, W) with class indices
        alpha: Transparency level for mask overlay (0.0 to 1.0)

    Returns:
        Combined image as numpy array of shape (H, W, 3)
    """
    img_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.numpy()

    # Create a colored mask using matplotlib's viridis (same as plotly)
    mask_rgb = np.zeros((*mask_np.shape, 3))

    # Get unique mask values (excluding background)
    unique_vals = np.unique(mask_np)
    non_zero_vals = unique_vals[unique_vals != 0]

    if len(non_zero_vals) > 0:
        # Plotly treats mask values as discrete categories, not continuous
        # We need to map each unique value to its position in the sorted unique values
        unique_vals_sorted = np.sort(np.unique(mask_np))

        # Create a mapping from value to normalized position
        mask_normalized = np.zeros_like(mask_np, dtype=float)
        for i, val in enumerate(unique_vals_sorted):
            normalized_val = i / (len(unique_vals_sorted) - 1) if len(unique_vals_sorted) > 1 else 0.0
            mask_normalized[mask_np == val] = normalized_val

        # Convert to RGB using matplotlib's viridis if available
        if matplotlib_available:
            try:
                # Use the modern matplotlib API
                viridis_cmap = plt.get_cmap("viridis")
            except AttributeError:
                # Fallback for older matplotlib versions
                from matplotlib import cm

                viridis_cmap = cm.get_cmap("viridis")
            mask_rgba = viridis_cmap(mask_normalized)
            mask_rgb = mask_rgba[:, :, :3]  # Remove alpha channel
        else:
            # Simple fallback - use the same discrete mapping
            for i, val in enumerate(unique_vals_sorted):
                norm_val = i / (len(unique_vals_sorted) - 1) if len(unique_vals_sorted) > 1 else 0.0
                mask_indices = mask_np == val

                if norm_val < QUARTER_THRESHOLD:
                    mask_rgb[mask_indices] = [0.267004, 0.004874, 0.329415]
                elif norm_val < HALF_THRESHOLD:
                    mask_rgb[mask_indices] = [0.127568, 0.566949, 0.550556]
                elif norm_val < THREE_QUARTER_THRESHOLD:
                    mask_rgb[mask_indices] = [0.369214, 0.788888, 0.382914]
                else:
                    mask_rgb[mask_indices] = [0.993248, 0.906157, 0.143936]

    # Combine image and mask
    combined = img_np * (1 - alpha) + mask_rgb * alpha
    return np.clip(combined, 0, 1)


def main() -> None:
    """Visualize dataset images and masks, displaying them in the browser."""
    dataset = CTLogDataset(data_dir="data/processed/set_24", resolution=(458, 530))

    output_dir = Path("data/processed/visualizations")
    output_dir.mkdir(exist_ok=True)
    output_path = Path("/home/mary/Downloads/ct_log/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    for item in dataset:
        img_fig = px.imshow(item["image"].permute(1, 2, 0), title=str(item["path"]))
        img_fig.show()

        mask_fig = px.imshow(item["mask"], title=str(item["path"]))
        # mask_fig.show()

        mask_fig.write_image(output_path / f"{Path(item['path']).stem}_mask.png")

        combined_image = render_mask_on_image(item["image"], item["mask"])
        combined_fig = px.imshow(combined_image, title=f"{item['path']} - Overlay")
        # combined_fig.show()

        combined_fig.write_image(output_path / f"{Path(item['path']).stem}_overlay.png")


if __name__ == "__main__":
    main()
