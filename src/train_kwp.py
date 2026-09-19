import argparse
from pathlib import Path

import torch
import torchvision

from src.configs.kwp_training_config import KwpTrainingConfig
from src.dataset.ct_log_kwp_dataset import CTLogKwpDataset
from src.loggers import CombinedLogger, LocalLogger, MlflowLogger
from src.loss.functional.focal_loss import multiclass_focal_loss
from src.loss.functional.tversky_loss import multiclass_tversky_loss
from src.segmentation_head import create_dinov3_segmentor
from src.utils.metrics import MetricsTracker
from src.utils.per_class_iou import PerClassIoU


def extract_features(model: torch.nn.Module, images: torch.Tensor, n_layers: int) -> torch.Tensor:
    """Extract and concatenate patch features from the last n_layers backbone blocks.

    Args:
        model: Frozen DINOv3 backbone.
        images: [B, 3, H, W] normalized input.
        n_layers: Number of intermediate layers to concatenate.

    Returns:
        torch.Tensor: [B, num_patches, 1024 * n_layers] concatenated features.
    """
    layers = model.get_intermediate_layers(images, n=n_layers, return_class_token=False)
    return torch.cat(layers, dim=-1)


def make_transform() -> torchvision.transforms.Normalize:
    """Build the ImageNet normalization applied to stacked slices."""
    return torchvision.transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def build_dataloaders(config: KwpTrainingConfig) -> dict[str, torch.utils.data.DataLoader]:
    """Create train/val dataloaders with a log-level holdout.

    Args:
        config: Training configuration.

    Returns:
        dict[str, torch.utils.data.DataLoader]: loaders keyed by "train" and "val".
    """
    train_ds = CTLogKwpDataset(
        config.train_logs, resolution=config.resolution, window=config.window, pith_radius=config.pith_radius
    )
    val_ds = CTLogKwpDataset(
        config.val_logs, resolution=config.resolution, window=config.window, pith_radius=config.pith_radius
    )
    return {
        "train": torch.utils.data.DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
        ),
        "val": torch.utils.data.DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
        ),
    }


def compute_loss(
    outputs: torch.Tensor,
    masks: torch.Tensor,
    config: KwpTrainingConfig,
) -> torch.Tensor:
    """Compute the combined focal + Tversky loss.

    Args:
        outputs: [B, C, H, W] logits.
        masks: [B, H, W] int64 class ids.
        config: Training configuration.

    Returns:
        torch.Tensor: Scalar loss.
    """
    distribution_loss = multiclass_focal_loss(outputs, masks, config.focal_alpha, config.focal_gamma)

    masks_one_hot = torch.nn.functional.one_hot(masks, config.num_classes + 1).permute(0, 3, 1, 2)
    district_loss = multiclass_tversky_loss(
        outputs,
        masks_one_hot,
        alpha=config.tversky_alpha,
        beta=config.tversky_beta,
        ignore_background=config.ignore_background_in_tversky,
    )

    return config.distribution_loss_weight * distribution_loss + config.district_loss_weight * district_loss


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    seg_head: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    transform: torch.nn.Module,
    device: torch.device,
    config: KwpTrainingConfig,
) -> tuple[float, dict[str, float]]:
    """Evaluate on a dataloader, returning mean loss and per-class IoU.

    Args:
        model: Frozen DINOv3 backbone.
        seg_head: Segmentation head.
        dataloader: Validation loader.
        transform: Normalization transform.
        device: Compute device.
        config: Training configuration.

    Returns:
        tuple[float, dict[str, float]]: mean loss and metric dict.
    """
    model.eval()
    seg_head.eval()
    losses = []
    iou = PerClassIoU(num_classes=config.num_classes + 1)

    for batch in dataloader:
        images = transform(batch["image"].to(device))
        masks = batch["mask"].to(device)

        features = extract_features(model, images, config.n_layers)
        outputs = seg_head(features)

        losses.append(compute_loss(outputs, masks, config).item())
        iou.update(outputs.argmax(dim=1).cpu(), masks.cpu())

    metrics = iou.compute()
    return float(torch.tensor(losses).mean().item()), metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/train_kwp.yaml")
    parser.add_argument("--window", type=int, default=None, help="Override window (0=baseline, 1=2.5D).")
    parser.add_argument("--run_name", type=str, default=None, help="Override MLflow run name.")
    parser.add_argument("--n_layers", type=int, default=None, help="Override number of backbone layers fused.")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Override checkpoint path.")
    parser.add_argument("--local_log_dir", type=str, default=None, help="Override local log directory.")
    parser.add_argument("--train_logs", type=str, nargs="+", default=None, help="Override training log dirs.")
    parser.add_argument("--val_logs", type=str, nargs="+", default=None, help="Override validation log dirs.")
    args = parser.parse_args()

    config = KwpTrainingConfig.from_yaml(args.config)
    if args.window is not None:
        config.window = args.window
    if args.run_name is not None:
        config.mlflow_run_name = args.run_name
    if args.n_layers is not None:
        config.n_layers = args.n_layers
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.checkpoint_path is not None:
        config.checkpoint_path = Path(args.checkpoint_path)
    if args.local_log_dir is not None:
        config.local_log_dir = Path(args.local_log_dir)
    if args.train_logs is not None:
        config.train_logs = args.train_logs
    if args.val_logs is not None:
        config.val_logs = args.val_logs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(config)
    tracker = MetricsTracker()

    loggers = []
    if config.use_local_logger:
        loggers.append(LocalLogger(log_dir=config.local_log_dir))
    if config.use_mlflow:
        loggers.append(
            MlflowLogger(
                experiment_name=config.mlflow_experiment_name,
                run_name=config.mlflow_run_name,
                tracking_uri=config.mlflow_tracking_uri,
            )
        )
    logger = CombinedLogger(loggers)
    logger.start()
    logger.log_params(config.model_dump())

    model, seg_head = create_dinov3_segmentor(
        backbone_weights=config.backbone_weights,
        num_classes=config.num_classes + 1,
        input_size=config.resolution[0],
        n_layers=config.n_layers,
    )
    model = model.to(device)
    seg_head = seg_head.to(device)

    transform = make_transform()
    optimizer = torch.optim.Adam(seg_head.parameters(), lr=config.lr)

    best_fg_iou = 0.0
    for epoch_idx in range(config.num_epochs):
        model.eval()
        seg_head.train()
        losses = []

        for batch_idx, batch in enumerate(loaders["train"]):
            optimizer.zero_grad()

            images = transform(batch["image"].to(device))
            masks = batch["mask"].to(device)

            with torch.no_grad():
                features = extract_features(model, images, config.n_layers)

            outputs = seg_head(features)
            loss = compute_loss(outputs, masks, config)
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().item())
            if config.log_interval and batch_idx % config.log_interval == 0:
                print(f"Epoch {epoch_idx}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        train_loss = float(torch.tensor(losses).mean().item())

        val_loss, val_metrics = evaluate(model, seg_head, loaders["val"], transform, device, config)
        fg_iou = val_metrics["mean_iou_fg"]

        empty_metrics = {key: 0.0 for key in val_metrics}
        tracker.add(epoch_idx, "train", train_loss, 0.0, **empty_metrics)
        logger.log_metrics(tracker.get(epoch_idx, "train")[-1])
        tracker.add(epoch_idx, "val", val_loss, fg_iou, **val_metrics)
        logger.log_metrics(tracker.get(epoch_idx, "val")[-1])
        iou_str = " ".join(f"{k}={v:.3f}" for k, v in val_metrics.items())
        print(f"Epoch {epoch_idx}, TrainLoss {train_loss:.4f}, ValLoss {val_loss:.4f}, {iou_str}")

        if fg_iou > best_fg_iou:
            best_fg_iou = fg_iou
            config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(seg_head.state_dict(), config.checkpoint_path)
            logger.log_model(seg_head, f"kwp_seg_head_epoch_{epoch_idx}")

    logger.end()
    print(f"Best foreground mean IoU: {best_fg_iou:.4f}")


if __name__ == "__main__":
    main()
