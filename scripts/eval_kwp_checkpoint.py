"""Evaluate a trained KWP checkpoint on a given split.

Reports per-class IoU plus pith pixel error, so a finished model can be scored
on the held-out test log without rerunning training.
"""

import argparse
from pathlib import Path

from src.configs.kwp_training_config import KwpTrainingConfig
from src.dataset.ct_log_kwp_dataset import CTLogKwpDataset
from src.segmentation_head import PithRegressionHead, create_dinov3_segmentor
from src.train_kwp import evaluate, make_transform
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/train_kwp_long.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config = KwpTrainingConfig.from_yaml(args.config)
    checkpoint = Path(args.checkpoint) if args.checkpoint else config.checkpoint_path
    log_dirs = {"train": config.train_logs, "val": config.val_logs, "test": config.test_logs}[args.split]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CTLogKwpDataset(
        log_dirs, resolution=config.resolution, window=config.window, pith_radius=config.pith_radius
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
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

    # The saved checkpoint holds only the seg head; the pith head lives in the
    # resume state, so load it from there when present.
    pith_head = None
    resume_path = checkpoint.with_suffix(".resume.pth")
    if config.pith_regression and resume_path.exists():
        state = torch.load(resume_path, map_location="cpu")
        if state.get("pith_head") is not None:
            pith_head = PithRegressionHead(feature_dim=1024 * config.n_layers)
            pith_head.load_state_dict(state["pith_head"])
            pith_head = pith_head.to(device).eval()

    loss, metrics = evaluate(model, seg_head, loader, make_transform(), device, config, pith_head)
    print(f"split={args.split}  n={len(dataset)}  checkpoint={checkpoint.name}")
    print(f"loss {loss:.4f}")
    for key, value in metrics.items():
        print(f"  {key:>20} {value:.4f}")


if __name__ == "__main__":
    main()
