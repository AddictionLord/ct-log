import argparse
from dataclasses import asdict
import os
from pathlib import Path
import time
from typing import Dict, List, Optional

import torch
import torchvision

from src.configs.kwp_training_config import KwpTrainingConfig
from src.dataset.ct_log_kwp_dataset import CTLogKwpDataset
from src.dataset.kwp_mask import KwpMaskBuilder
from src.loggers import CombinedLogger, ILogger, LocalLogger, MlflowLogger
from src.loss.functional.focal_loss import multiclass_focal_loss
from src.loss.functional.tversky_loss import multiclass_tversky_loss
from src.segmentation_head import PithRegressionHead, create_dinov3_segmentor
from src.utils.checkpoints import save_best_copy
from src.utils.class_balance import (
    capped_inverse_frequency_weights,
    effective_number_weights,
    inverse_frequency_weights,
)
from src.utils.metrics import MetricsTracker
from src.utils.per_class_iou import PerClassIoU
from src.utils.prediction_panels import render_panel
from src.utils.provenance import git_provenance
from src.utils.train_info import TrainInfo

NATIVE_SLICE_SIZE = 778


def extract_features(model: torch.nn.Module, images: torch.Tensor, n_layers: int, bf16: bool = False) -> torch.Tensor:
    """Extract and concatenate patch features from the last n_layers backbone blocks.

    Args:
        model: Frozen DINOv3 backbone.
        images: [B, 3, H, W] normalized input.
        n_layers: Number of intermediate layers to concatenate.
        bf16: Run the backbone under bfloat16 autocast; features are returned as float32.

    Returns:
        torch.Tensor: [B, num_patches, 1024 * n_layers] concatenated features.
    """
    with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=bf16):
        layers = model.get_intermediate_layers(images, n=n_layers, return_class_token=False)
    return torch.cat(layers, dim=-1).float()


def make_transform() -> torchvision.transforms.Normalize:
    """Build the ImageNet normalization applied to stacked slices."""
    return torchvision.transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def build_dataloaders(config: KwpTrainingConfig) -> dict[str, torch.utils.data.DataLoader]:
    """Create train/val (and optional test) dataloaders with a log-level holdout.

    Args:
        config: Training configuration.

    Returns:
        dict[str, torch.utils.data.DataLoader]: loaders keyed by "train", "val"
            and, when test_logs is set, "test".
    """
    train_ds = CTLogKwpDataset(
        config.train_logs,
        resolution=config.resolution,
        window=config.window,
        pith_radius=config.pith_radius,
        augment=config.augment,
    )
    val_ds = CTLogKwpDataset(
        config.val_logs, resolution=config.resolution, window=config.window, pith_radius=config.pith_radius
    )
    if config.test_logs:
        test_ds = CTLogKwpDataset(
            config.test_logs, resolution=config.resolution, window=config.window, pith_radius=config.pith_radius
        )
    loaders = {
        "train": torch.utils.data.DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
        ),
        "val": torch.utils.data.DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
        ),
    }
    if config.test_logs:
        loaders["test"] = torch.utils.data.DataLoader(
            test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
        )
    if config.augment:
        train_eval_ds = CTLogKwpDataset(
            config.train_logs, resolution=config.resolution, window=config.window, pith_radius=config.pith_radius
        )
        loaders["train_eval"] = torch.utils.data.DataLoader(
            train_eval_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
        )
    if config.train_probe_logs:
        probe_ds = CTLogKwpDataset(
            config.train_probe_logs, resolution=config.resolution, window=config.window, pith_radius=config.pith_radius
        )
        loaders["train_probe"] = torch.utils.data.DataLoader(
            probe_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
        )
    return loaders


def build_class_weights(config: KwpTrainingConfig, device: torch.device) -> torch.Tensor | None:
    """Build per-class loss weights from config.class_weighting, or None to disable weighting.

    Args:
        config: Training configuration; class_pixel_counts must be set when class_weighting
            is not "none".
        device: Device the weights are moved to.

    Returns:
        torch.Tensor | None: [num_classes] weights, or None when class_weighting is "none".

    Raises:
        ValueError: If class_weighting is not "none" but class_pixel_counts is unset, or its
            length does not match num_classes.
    """
    if config.class_weighting == "none":
        return None
    if config.class_pixel_counts is None:
        msg = f"class_weighting={config.class_weighting!r} requires class_pixel_counts to be set"
        raise ValueError(msg)
    if len(config.class_pixel_counts) != config.num_classes:
        msg = (
            f"class_pixel_counts has {len(config.class_pixel_counts)} entries, "
            f"expected {config.num_classes} (num_classes includes background)"
        )
        raise ValueError(msg)

    if config.class_weighting == "effective_number":
        weights = effective_number_weights(config.class_pixel_counts, beta=config.class_weight_beta)
    elif config.class_weighting == "capped_inverse_frequency":
        weights = capped_inverse_frequency_weights(
            config.class_pixel_counts,
            floor_share=config.class_weight_floor_share,
            power=config.class_weight_power,
        )
    else:
        weights = inverse_frequency_weights(config.class_pixel_counts, power=config.class_weight_power)
    return weights.to(device)


def compute_loss(
    outputs: torch.Tensor,
    masks: torch.Tensor,
    config: KwpTrainingConfig,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the combined focal + Tversky loss.

    Args:
        outputs: [B, C, H, W] logits.
        masks: [B, H, W] int64 class ids.
        config: Training configuration.
        class_weights: Optional [num_classes] per-class weights, from build_class_weights.

    Returns:
        torch.Tensor: Scalar loss.
    """
    distribution_loss = multiclass_focal_loss(
        outputs, masks, config.focal_alpha, config.focal_gamma, class_weights=class_weights
    )

    masks_one_hot = torch.nn.functional.one_hot(masks, config.num_classes).permute(0, 3, 1, 2)
    district_loss = multiclass_tversky_loss(
        outputs,
        masks_one_hot,
        alpha=config.tversky_alpha,
        beta=config.tversky_beta,
        ignore_background=config.ignore_background_in_tversky,
        class_weights=class_weights,
    )

    return config.distribution_loss_weight * distribution_loss + config.district_loss_weight * district_loss


def pith_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE on normalized pith coordinates, skipping slices with no pith.

    Args:
        predictions: [B, 2] predicted normalized (x, y).
        targets: [B, 3] ground-truth (x, y, valid).

    Returns:
        torch.Tensor: Scalar loss; zero when no slice in the batch has a pith.
    """
    valid = targets[:, 2] > 0
    if not valid.any():
        return predictions.sum() * 0.0
    return torch.nn.functional.mse_loss(predictions[valid], targets[valid, :2])


@torch.no_grad()
def pith_pixel_errors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    config: KwpTrainingConfig,
) -> List[float]:
    """Euclidean pith error in ORIGINAL slice pixels, matching ann_pipeline/pith/eval.py.

    Args:
        predictions: [B, 2] predicted normalized (x, y).
        targets: [B, 3] ground-truth (x, y, valid).
        config: Training configuration (unused scale kept explicit below).

    Returns:
        List[float]: One error per valid slice, in pixels of the native 778x778 slice.
    """
    valid = targets[:, 2] > 0
    if not valid.any():
        return []
    # Coordinates are normalized, so scale by the native slice size rather than
    # the resized input: errors stay comparable to the detector's px numbers.
    scale = NATIVE_SLICE_SIZE
    delta = (predictions[valid] - targets[valid, :2]) * scale
    return torch.linalg.vector_norm(delta, dim=1).cpu().tolist()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    seg_head: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    transform: torch.nn.Module,
    device: torch.device,
    config: KwpTrainingConfig,
    pith_head: torch.nn.Module | None = None,
    class_weights: torch.Tensor | None = None,
) -> tuple[float, dict[str, float]]:
    """Evaluate on a dataloader, returning mean loss and per-class IoU.

    Args:
        model: Frozen DINOv3 backbone.
        seg_head: Segmentation head.
        dataloader: Validation loader.
        transform: Normalization transform.
        device: Compute device.
        config: Training configuration.
        pith_head: Optional pith coordinate regression head.
        class_weights: Optional per-class loss weights; matches training so eval loss stays
            comparable across epochs. Does not affect IoU, which is unweighted either way.

    Returns:
        tuple[float, dict[str, float]]: mean loss and metric dict. When a pith
            head is given, the metrics include Euclidean pixel errors matching
            ann_pipeline/pith/eval.py (mean/median/p90).
    """
    model.eval()
    seg_head.eval()
    if pith_head is not None:
        pith_head.eval()
    losses = []
    iou = PerClassIoU(num_classes=config.num_classes, class_names=KwpMaskBuilder.class_names())
    pith_errors: List[float] = []

    for batch in dataloader:
        images = transform(batch["image"].to(device))
        masks = batch["mask"].to(device)

        features = extract_features(model, images, config.n_layers, config.backbone_bf16)
        outputs = seg_head(features)

        if pith_head is not None:
            pith_errors.extend(pith_pixel_errors(pith_head(features), batch["pith_xy"].to(device), config))

        losses.append(compute_loss(outputs, masks, config, class_weights).item())
        iou.update(outputs.argmax(dim=1).cpu(), masks.cpu())

    metrics = iou.compute()
    if pith_errors:
        errors = torch.tensor(pith_errors)
        metrics["pith_err_mean_px"] = float(errors.mean().item())
        metrics["pith_err_median_px"] = float(errors.median().item())
        metrics["pith_err_p90_px"] = float(torch.quantile(errors, 0.9).item())
    return float(torch.tensor(losses).mean().item()), metrics


@torch.no_grad()
def log_prediction_panels(
    model: torch.nn.Module,
    seg_head: torch.nn.Module,
    pith_head: Optional[torch.nn.Module],
    dataset: CTLogKwpDataset,
    split: str,
    step: int,
    transform: torch.nn.Module,
    device: torch.device,
    config: KwpTrainingConfig,
    logger: ILogger,
) -> None:
    """Log input | ground truth | prediction | error panels for fixed, evenly spaced slices.

    The same slices are used at every step, so the MLflow image slider shows how one slice
    evolves over training. Each slice is the centre of one of viz_num_frames equal bins, which
    skips the first and last slice of the split (log ends, least informative).

    Args:
        model: Frozen DINOv3 backbone.
        seg_head: Segmentation head.
        pith_head: Optional pith regression head; its prediction is drawn as a magenta cross.
        dataset: Dataset of the split.
        split: Split name used as the image key prefix.
        step: Epoch the panels belong to.
        transform: Normalization transform.
        device: Compute device.
        config: Training configuration.
        logger: Logger receiving the images.
    """
    model.eval()
    seg_head.eval()
    if pith_head is not None:
        pith_head.eval()

    indices = (
        ((torch.arange(config.viz_num_frames) + 0.5) * len(dataset) / config.viz_num_frames).long().unique().tolist()
    )
    samples = [dataset[index] for index in indices]
    images = torch.stack([sample["image"] for sample in samples])
    features = extract_features(model, transform(images.to(device)), config.n_layers, config.backbone_bf16)
    predictions = seg_head(features).argmax(dim=1).cpu().numpy()
    pith_predictions = pith_head(features).cpu().tolist() if pith_head is not None else [None] * len(samples)

    for sample, prediction, pith_prediction in zip(samples, predictions, pith_predictions):
        pith_true = sample["pith_xy"][:2].tolist() if sample["pith_xy"][2] > 0 else None
        panel = render_panel(sample["image"], sample["mask"].numpy(), prediction, pith_true, pith_prediction)
        path = Path(sample["path"])
        logger.log_image(panel, f"{split}/log{path.parent.parent.name}_{path.stem}", step)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/train_kwp.yaml")
    parser.add_argument("--window", type=int, default=None, help="Override window (0=baseline, 1=2.5D).")
    parser.add_argument("--run_name", type=str, default=None, help="Override MLflow run name.")
    parser.add_argument("--n_layers", type=int, default=None, help="Override number of backbone layers fused.")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Override checkpoint path.")
    parser.add_argument("--local_log_dir", type=str, default=None, help="Override local log directory.")
    parser.add_argument(
        "--lr_schedule", type=str, default=None, choices=["none", "cosine", "multistep"], help="LR schedule."
    )
    parser.add_argument("--resume", action="store_true", help="Save/restore full state so a crash can resume.")
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
    if args.lr_schedule is not None:
        config.lr_schedule = args.lr_schedule
    if args.resume:
        config.resume = True
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
    run_params = config.model_dump(mode="json") | git_provenance()
    logger.log_params(run_params)
    logger.log_dict(run_params, "config.yaml")

    model, seg_head = create_dinov3_segmentor(
        backbone_weights=config.backbone_weights,
        num_classes=config.num_classes,
        input_size=config.resolution[0],
        n_layers=config.n_layers,
    )
    model = model.to(device)
    seg_head = seg_head.to(device)

    pith_head = None
    trainable = list(seg_head.parameters())
    if config.pith_regression:
        pith_head = PithRegressionHead(feature_dim=1024 * config.n_layers).to(device)
        trainable += list(pith_head.parameters())

    backbone_params = unfreeze_last_blocks(model, config.backbone_trainable_blocks)
    finetune = bool(backbone_params)
    param_groups = [{"params": trainable, "lr": config.lr}]
    if finetune:
        param_groups.append({"params": list(backbone_params.values()), "lr": config.backbone_lr})
        print(
            f"Fine-tuning last {config.backbone_trainable_blocks} backbone blocks "
            f"({sum(p.numel() for p in backbone_params.values()) / 1e6:.1f}M params, lr={config.backbone_lr})"
        )
    backbone_ckpt_path = config.checkpoint_path.with_suffix(".backbone.pth")
    pith_ckpt_path = config.checkpoint_path.with_suffix(".pith.pth")

    transform = make_transform()
    class_weights = build_class_weights(config, device)
    if class_weights is not None:
        print(f"Class weights ({config.class_weighting}): {class_weights.tolist()}")
    optimizer = torch.optim.Adam(param_groups)

    scheduler = None
    if config.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    elif config.lr_schedule == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.lr_milestones, gamma=config.lr_gamma
        )

    best_fg_iou = 0.0
    best_smoothed = 0.0
    best_epoch: Optional[int] = None
    last_uploaded_smoothed = 0.0
    fg_history: List[float] = []
    start_epoch = 0
    best_copy_dir = config.local_checkpoint_dir / (config.mlflow_run_name or config.checkpoint_path.stem)
    info = TrainInfo(
        run_name=config.mlflow_run_name or config.checkpoint_path.stem,
        num_epochs=config.num_epochs,
        checkpoint_path=str(config.checkpoint_path),
        local_best_copies=str(best_copy_dir),
    )

    # Resume state lives beside the best-model checkpoint. Without it a crash
    # restart would silently begin at epoch 0 and corrupt the plateau reading.
    resume_path = config.checkpoint_path.with_suffix(".resume.pth")
    if config.init_from is not None and not (config.resume and resume_path.exists()):
        init_weights(config.init_from, seg_head, pith_head, model if finetune else None)
        print(f"Initialized weights from {config.init_from}")
    if config.resume and resume_path.exists():
        state = torch.load(resume_path)
        seg_head.load_state_dict(state["seg_head"])
        if finetune != (state.get("backbone") is not None):
            message = (
                f"Resume mismatch: backbone_trainable_blocks={config.backbone_trainable_blocks} but checkpoint "
                f"{'has' if state.get('backbone') is not None else 'lacks'} backbone state."
            )
            raise ValueError(message)
        if finetune:
            model.load_state_dict(state["backbone"], strict=False)
        optimizer.load_state_dict(state["optimizer"])

        # Fail loudly on a mismatch rather than silently continuing with a
        # freshly-initialized head or scheduler, which would invalidate the run.
        if (pith_head is not None) != (state.get("pith_head") is not None):
            message = (
                f"Resume mismatch: pith_regression={config.pith_regression} but checkpoint "
                f"{'has' if state.get('pith_head') is not None else 'lacks'} pith head state."
            )
            raise ValueError(message)
        if (scheduler is not None) != (state.get("scheduler") is not None):
            message = (
                f"Resume mismatch: lr_schedule={config.lr_schedule} but checkpoint "
                f"{'has' if state.get('scheduler') is not None else 'lacks'} scheduler state."
            )
            raise ValueError(message)

        if pith_head is not None:
            pith_head.load_state_dict(state["pith_head"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_fg_iou = state["best_fg_iou"]
        best_smoothed = state["best_smoothed"]
        fg_history = state["fg_history"]
        best_epoch = state.get("best_epoch")
        last_uploaded_smoothed = state.get("last_uploaded_smoothed", best_smoothed)
        if state.get("train_info") is not None:
            info = TrainInfo(**state["train_info"])
            info.num_epochs = config.num_epochs
            info.status = "running"
        else:
            info.best_epoch = best_epoch
            info.best_smoothed_fg = best_smoothed if best_epoch is not None else None
        info.resumed_from_epoch = start_epoch
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    for epoch_idx in range(start_epoch, config.num_epochs):
        epoch_start = time.monotonic()
        model.eval()
        seg_head.train()
        if pith_head is not None:
            pith_head.train()
        losses = []

        for batch_idx, batch in enumerate(loaders["train"]):
            optimizer.zero_grad()

            images = transform(batch["image"].to(device))
            masks = batch["mask"].to(device)

            with torch.set_grad_enabled(finetune):
                features = extract_features(model, images, config.n_layers, config.backbone_bf16)

            outputs = seg_head(features)
            loss = compute_loss(outputs, masks, config, class_weights)
            if pith_head is not None:
                loss = loss + config.pith_loss_weight * pith_loss(pith_head(features), batch["pith_xy"].to(device))
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().item())
            if config.log_interval and batch_idx % config.log_interval == 0:
                print(f"Epoch {epoch_idx}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        train_loss = float(torch.tensor(losses).mean().item())

        val_loss, val_metrics = evaluate(
            model, seg_head, loaders["val"], transform, device, config, pith_head, class_weights
        )
        fg_iou = val_metrics["mean_iou_fg"]

        # Train IoU is only computed by the periodic train-eval below; NaN keeps the CSV columns
        # aligned and is skipped by MLflow, where zeros would read as a collapsed model.
        not_computed = {key: float("nan") for key in val_metrics}
        tracker.add(epoch_idx, "train", train_loss, float("nan"), lr=optimizer.param_groups[0]["lr"], **not_computed)
        logger.log_metrics(tracker.get(epoch_idx, "train")[-1])
        tracker.add(epoch_idx, "val", val_loss, fg_iou, **val_metrics)
        logger.log_metrics(tracker.get(epoch_idx, "val")[-1])
        fg_history.append(fg_iou)
        smoothed = float(sum(fg_history[-5:]) / len(fg_history[-5:]))
        if fg_iou > best_fg_iou:
            info.best_raw_epoch = epoch_idx
            info.best_raw_val_fg = fg_iou
        best_fg_iou = max(best_fg_iou, fg_iou)
        info.last_val = {"loss": val_loss, **val_metrics, "smoothed_fg": smoothed}

        iou_str = " ".join(f"{k}={v:.3f}" for k, v in val_metrics.items())
        print(
            f"Epoch {epoch_idx}, TrainLoss {train_loss:.4f}, ValLoss {val_loss:.4f}, "
            f"{iou_str} smoothed_fg={smoothed:.4f}"
        )

        # Train-split IoU on the same metric as val: the gap between them is the
        # bias/variance signal (both low => underfit, train >> val => overfit).
        if config.train_eval_interval and epoch_idx % config.train_eval_interval == 0:
            train_eval_loss, train_eval_metrics = evaluate(
                model, seg_head, loaders.get("train_eval", loaders["train"]), transform, device, config, pith_head, class_weights
            )
            tracker.add(
                epoch_idx, "train_eval", train_eval_loss, train_eval_metrics["mean_iou_fg"], **train_eval_metrics
            )
            logger.log_metrics(tracker.get(epoch_idx, "train_eval")[-1])
            print(
                f"Epoch {epoch_idx}, TRAIN-EVAL loss {train_eval_loss:.4f} "
                f"fg={train_eval_metrics['mean_iou_fg']:.4f} "
                f"knot={train_eval_metrics['iou_knot']:.3f} gap={train_eval_metrics['mean_iou_fg'] - fg_iou:+.4f}"
            )
            # Human-labelled train logs: separates auto-vs-human label mismatch (train_eval vs
            # train_probe) from cross-log generalisation (train_probe vs val).
            if "train_probe" in loaders:
                probe_loss, probe_metrics = evaluate(
                    model, seg_head, loaders["train_probe"], transform, device, config, pith_head, class_weights
                )
                tracker.add(epoch_idx, "train_probe", probe_loss, probe_metrics["mean_iou_fg"], **probe_metrics)
                logger.log_metrics(tracker.get(epoch_idx, "train_probe")[-1])
                print(
                    f"Epoch {epoch_idx}, TRAIN-PROBE loss {probe_loss:.4f} "
                    f"fg={probe_metrics['mean_iou_fg']:.4f} knot={probe_metrics['iou_knot']:.3f}"
                )

        if config.viz_interval and (epoch_idx % config.viz_interval == 0 or epoch_idx == config.num_epochs - 1):
            for split in ("train", "val"):
                log_prediction_panels(
                    model,
                    seg_head,
                    pith_head,
                    loaders[split].dataset,
                    split,
                    epoch_idx,
                    transform,
                    device,
                    config,
                    logger,
                )

        if scheduler is not None:
            scheduler.step()

        # Checkpoint on the 5-epoch smoothed metric: raw fg swings ~+-0.01 between
        # adjacent epochs, so a best-epoch criterion latches onto lucky outliers.
        if len(fg_history) >= 5 and smoothed > best_smoothed:
            best_smoothed = smoothed
            best_epoch = epoch_idx
            config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(seg_head.state_dict(), config.checkpoint_path)
            if finetune:
                torch.save(backbone_state(model, backbone_params), backbone_ckpt_path)
            if pith_head is not None:
                torch.save(pith_head.state_dict(), pith_ckpt_path)
            copy_path = save_best_copy(
                best_state(seg_head, pith_head, backbone_state(model, backbone_params) if finetune else None),
                best_copy_dir,
                epoch_idx,
                config.local_checkpoint_keep,
            )
            print(f"Epoch {epoch_idx}, new best smoothed_fg={smoothed:.4f}, local copy {copy_path}")

            # Each logged model costs ~160 MB of DagsHub storage that deletes never free, so
            # only upload bests that moved meaningfully; the true best is uploaded after training.
            info.best_epoch = epoch_idx
            info.best_smoothed_fg = smoothed
            info.best_epoch_val_fg = fg_iou
            if smoothed >= last_uploaded_smoothed + config.mlflow_model_min_improvement:
                logger.log_model(seg_head, f"kwp_seg_head_epoch_{epoch_idx}", seg_head.example_input(), step=epoch_idx)
                last_uploaded_smoothed = smoothed
                info.uploaded_models.append({"epoch": epoch_idx, "smoothed_fg": round(smoothed, 4)})
                print(f"Epoch {epoch_idx}, model uploaded (smoothed_fg={smoothed:.4f})")
            else:
                print(
                    f"Epoch {epoch_idx}, model upload skipped: smoothed_fg={smoothed:.4f} < last uploaded "
                    f"{last_uploaded_smoothed:.4f} + {config.mlflow_model_min_improvement}"
                )

        info.current_epoch = epoch_idx
        info.last_epoch_minutes = round((time.monotonic() - epoch_start) / 60, 2)

        if config.resume:
            config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file and rename: a kill mid-torch.save would leave
            # a truncated, unreadable checkpoint, losing exactly the state that
            # resume exists to protect. os.replace is atomic on the same filesystem.
            tmp_path = resume_path.with_suffix(".tmp")
            torch.save(
                {
                    "epoch": epoch_idx,
                    "seg_head": seg_head.state_dict(),
                    "pith_head": pith_head.state_dict() if pith_head is not None else None,
                    "backbone": backbone_state(model, backbone_params) if finetune else None,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "best_fg_iou": best_fg_iou,
                    "best_smoothed": best_smoothed,
                    "best_epoch": best_epoch,
                    "last_uploaded_smoothed": last_uploaded_smoothed,
                    "fg_history": fg_history,
                    "train_info": asdict(info),
                },
                tmp_path,
            )
            os.replace(tmp_path, resume_path)

        logger.log_dict(info.to_dict(), "train_info.yaml")

    print(f"Best foreground mean IoU: {best_fg_iou:.4f}")
    print(f"Best smoothed (5-epoch) foreground mean IoU: {best_smoothed:.4f}")

    # Test on the held-out log using the checkpointed (best-smoothed) head, not
    # the final-epoch weights.
    if best_epoch is not None:
        seg_head.load_state_dict(torch.load(config.checkpoint_path))
        if finetune:
            model.load_state_dict(torch.load(backbone_ckpt_path), strict=False)
        if pith_head is not None and pith_ckpt_path.exists():
            pith_head.load_state_dict(torch.load(pith_ckpt_path))
        if best_smoothed > last_uploaded_smoothed:
            logger.log_model(seg_head, f"kwp_seg_head_epoch_{best_epoch}", seg_head.example_input(), step=best_epoch)
            last_uploaded_smoothed = best_smoothed
            info.uploaded_models.append({"epoch": best_epoch, "smoothed_fg": round(best_smoothed, 4)})
            print(f"Final best (epoch {best_epoch}, smoothed_fg={best_smoothed:.4f}) was not uploaded; uploaded now")

    if "test" in loaders:
        if best_epoch is None:
            print("No checkpoint written; testing final-epoch weights instead.")
        test_loss, test_metrics = evaluate(
            model, seg_head, loaders["test"], transform, device, config, pith_head, class_weights
        )
        test_str = " ".join(f"{k}={v:.3f}" for k, v in test_metrics.items())
        print(f"TEST loss {test_loss:.4f} {test_str}")
        tracker.add(config.num_epochs, "test", test_loss, test_metrics["mean_iou_fg"], **test_metrics)
        info.test = {"loss": test_loss, **test_metrics}
        logger.log_metrics(tracker.get(config.num_epochs, "test")[-1])
        if config.viz_interval:
            log_prediction_panels(
                model,
                seg_head,
                pith_head,
                loaders["test"].dataset,
                "test",
                config.num_epochs,
                transform,
                device,
                config,
                logger,
            )

    info.status = "finished"
    logger.log_dict(info.to_dict(), "train_info.yaml")
    logger.end()


def best_state(
    seg_head: torch.nn.Module, pith_head: Optional[torch.nn.Module], backbone: Optional[dict] = None
) -> Dict[str, Optional[dict]]:
    """State dicts of a best model for the local best-copy history.

    Args:
        seg_head: Segmentation head.
        pith_head: Optional pith regression head.
        backbone: Optional state dict of the fine-tuned backbone parameters.

    Returns:
        Dict[str, Optional[dict]]: seg_head, pith_head and backbone state dicts (None when absent).
    """
    return {
        "seg_head": seg_head.state_dict(),
        "pith_head": pith_head.state_dict() if pith_head is not None else None,
        "backbone": backbone,
    }


def init_weights(
    path: Path,
    seg_head: torch.nn.Module,
    pith_head: Optional[torch.nn.Module],
    backbone: Optional[torch.nn.Module],
) -> None:
    """Load head (and fine-tuned backbone) weights from another run's resume or best-copy file.

    Only weights are taken; optimizer, scheduler and epoch counters start fresh, so a finished
    cosine run can be continued with a new schedule (warm restart).

    Args:
        path: A ``*.resume.pth`` or best-copy file with seg_head / pith_head / backbone entries.
        seg_head: Segmentation head to initialize.
        pith_head: Optional pith head to initialize.
        backbone: Backbone to initialize when its last blocks are fine-tuned, else None.

    Raises:
        ValueError: if the file lacks state this run needs.
    """
    state = torch.load(path, map_location="cpu")
    if (pith_head is not None and state.get("pith_head") is None) or (
        backbone is not None and state.get("backbone") is None
    ):
        message = f"init_from {path} lacks pith_head or backbone state required by this config"
        raise ValueError(message)
    seg_head.load_state_dict(state["seg_head"])
    if pith_head is not None:
        pith_head.load_state_dict(state["pith_head"])
    if backbone is not None:
        backbone.load_state_dict(state["backbone"], strict=False)


def unfreeze_last_blocks(model: torch.nn.Module, num_blocks: int) -> Dict[str, torch.nn.Parameter]:
    """Make the last num_blocks transformer blocks and the output norms trainable.

    Args:
        model: DINOv3 backbone with all parameters frozen.
        num_blocks: Number of trailing blocks to unfreeze; 0 keeps the backbone frozen.

    Returns:
        Dict[str, torch.nn.Parameter]: the unfrozen parameters keyed by their backbone name.
    """
    if num_blocks == 0:
        return {}
    first = len(model.blocks) - num_blocks
    prefixes = tuple(f"blocks.{index}." for index in range(first, len(model.blocks))) + ("norm.", "cls_norm.")
    params = {name: param for name, param in model.named_parameters() if name.startswith(prefixes)}
    for param in params.values():
        param.requires_grad = True
    return params


def backbone_state(model: torch.nn.Module, params: Dict[str, torch.nn.Parameter]) -> Dict[str, torch.Tensor]:
    """State of the fine-tuned backbone parameters only (the frozen rest is the pretrained file).

    Args:
        model: DINOv3 backbone.
        params: Trainable backbone parameters from unfreeze_last_blocks.

    Returns:
        Dict[str, torch.Tensor]: name -> tensor for the trainable parameters.
    """
    return {name: tensor for name, tensor in model.state_dict().items() if name in params}


if __name__ == "__main__":
    main()
