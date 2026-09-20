"""Render model predictions against ground truth for visual inspection.

Writes an HTML page per split with side-by-side image / GT / prediction /
error overlay, so failure modes (missed knots, boundary slop, false positives)
can be read directly rather than inferred from IoU.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.configs.kwp_training_config import KwpTrainingConfig
from src.dataset.ct_log_kwp_dataset import CTLogKwpDataset
from src.segmentation_head import create_dinov3_segmentor
from src.train_kwp import extract_features, make_transform
import torch

CLASS_NAMES = {0: "background", 1: "wood", 2: "knot", 3: "pith"}
# background, wood, knot, pith
CLASS_COLORS = np.array([[0, 0, 0], [120, 90, 60], [220, 50, 50], [0, 200, 255]], dtype=np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    """Map a class-id mask to RGB.

    Args:
        mask: [H, W] int array of class ids.

    Returns:
        np.ndarray: [H, W, 3] uint8 RGB image.
    """
    return CLASS_COLORS[mask]


def error_overlay(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Build an RGB overlay of agreement and error.

    Green = correct foreground, red = false positive, blue = false negative.

    Args:
        pred: [H, W] predicted class ids.
        target: [H, W] ground-truth class ids.

    Returns:
        np.ndarray: [H, W, 3] uint8 RGB image.
    """
    out = np.zeros((*pred.shape, 3), dtype=np.uint8)
    pred_fg, target_fg = pred > 0, target > 0
    out[pred_fg & target_fg & (pred == target)] = (0, 200, 0)
    out[pred_fg & target_fg & (pred != target)] = (255, 200, 0)
    out[pred_fg & ~target_fg] = (255, 0, 0)
    out[~pred_fg & target_fg] = (0, 80, 255)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/train_kwp_long.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Seg head weights; default config path.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--out", type=str, default="logs/kwp_long/predictions.html")
    args = parser.parse_args()

    config = KwpTrainingConfig.from_yaml(args.config)
    checkpoint = Path(args.checkpoint) if args.checkpoint else config.checkpoint_path
    log_dirs = {"train": config.train_logs, "val": config.val_logs, "test": config.test_logs}[args.split]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CTLogKwpDataset(
        log_dirs, resolution=config.resolution, window=config.window, pith_radius=config.pith_radius
    )

    model, seg_head = create_dinov3_segmentor(
        backbone_weights=config.backbone_weights,
        num_classes=config.num_classes + 1,
        input_size=config.resolution[0],
        n_layers=config.n_layers,
    )
    seg_head.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device).eval()
    seg_head = seg_head.to(device).eval()
    transform = make_transform()

    indices = np.linspace(0, len(dataset) - 1, args.num_samples).astype(int)
    rows: List[int] = list(range(len(indices)))
    figure = make_subplots(
        rows=len(rows),
        cols=4,
        subplot_titles=[t for _ in rows for t in ("slice", "ground truth", "prediction", "error")],
        vertical_spacing=0.01,
        horizontal_spacing=0.01,
    )

    for row, idx in enumerate(indices, start=1):
        sample = dataset[int(idx)]
        image = transform(sample["image"].unsqueeze(0).to(device))
        with torch.no_grad():
            features = extract_features(model, image, config.n_layers)
            prediction = seg_head(features).argmax(1).squeeze(0).cpu().numpy()
        target = sample["mask"].numpy()
        # Centre slice of the 2.5D stack, scaled to 8-bit for display.
        centre = (sample["image"][1].numpy() * 255).astype(np.uint8)

        panels = [
            np.stack([centre] * 3, axis=-1),
            colorize(target),
            colorize(prediction),
            error_overlay(prediction, target),
        ]
        for col, panel in enumerate(panels, start=1):
            figure.add_trace(go.Image(z=panel), row=row, col=col)

    figure.update_xaxes(showticklabels=False)
    figure.update_yaxes(showticklabels=False)
    figure.update_layout(
        height=260 * len(rows),
        width=1200,
        title=f"{args.split} predictions - {checkpoint.name} (green=correct, red=false pos, blue=missed)",
        showlegend=False,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(out_path))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
