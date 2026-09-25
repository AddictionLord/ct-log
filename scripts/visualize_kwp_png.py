"""Render KWP predictions as PNG grids for viewing on mobile.

Same content as the HTML visualizer but written straight to PNG, so no browser
is needed. Each row is one slice: image / ground truth / prediction / error.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from src.configs.kwp_training_config import KwpTrainingConfig
from src.dataset.ct_log_kwp_dataset import CTLogKwpDataset
from src.dataset.kwp_mask import KwpMaskBuilder
from src.segmentation_head import create_dinov3_segmentor
from src.train_kwp import extract_features, make_transform
from src.utils.per_class_iou import PerClassIoU
from src.utils.prediction_panels import colorize, error_overlay
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/train_kwp_long.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--knot_frames_only", action="store_true", help="Prefer slices containing knots.")
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
        num_classes=config.num_classes,
        input_size=config.resolution[0],
        n_layers=config.n_layers,
    )
    seg_head.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device).eval()
    seg_head = seg_head.to(device).eval()
    transform = make_transform()

    candidates = np.linspace(0, len(dataset) - 1, args.num_samples * 3).astype(int)
    chosen = []
    for idx in candidates:
        if len(chosen) >= args.num_samples:
            break
        sample = dataset[int(idx)]
        if args.knot_frames_only and (sample["mask"] == 2).sum() == 0:
            continue
        chosen.append((int(idx), sample))
    if not chosen:
        chosen = [(int(i), dataset[int(i)]) for i in candidates[: args.num_samples]]

    rows = len(chosen)
    figure, axes = plt.subplots(rows, 4, figsize=(13, 3.3 * rows))
    axes = np.atleast_2d(axes)

    for row, (idx, sample) in enumerate(chosen):
        image = transform(sample["image"].unsqueeze(0).to(device))
        with torch.no_grad():
            features = extract_features(model, image, config.n_layers)
            prediction = seg_head(features).argmax(1).squeeze(0).cpu().numpy()
        target = sample["mask"].numpy()

        iou = PerClassIoU(num_classes=config.num_classes, class_names=KwpMaskBuilder.class_names())
        iou.update(torch.from_numpy(prediction).unsqueeze(0), torch.from_numpy(target).unsqueeze(0))
        scores = iou.compute()

        centre = (sample["image"][1].numpy() * 255).astype(np.uint8)
        panels = [
            (np.stack([centre] * 3, -1), Path(sample["path"]).name),
            (colorize(target), "ground truth"),
            (colorize(prediction), f"pred knot IoU {scores['iou_knot']:.2f}"),
            (error_overlay(prediction, target), f"error  fg {scores['mean_iou_fg']:.2f}"),
        ]
        for col, (panel, title) in enumerate(panels):
            axes[row, col].imshow(panel)
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")

    handles = [
        mpatches.Patch(color="#00be00", label="correct"),
        mpatches.Patch(color="#ffd200", label="wrong class"),
        mpatches.Patch(color="#ff0000", label="false positive"),
        mpatches.Patch(color="#005aff", label="missed"),
        mpatches.Patch(color="#e62828", label="knot (GT/pred)"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False)
    figure.suptitle(f"{args.split} - {checkpoint.name}", fontsize=12)
    figure.tight_layout(rect=(0, 0.03, 1, 0.98))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
