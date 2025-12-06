import argparse

import plotly.express as px
from segmentation_head import create_dinov3_segmentor
import torch
from torchmetrics.segmentation import MeanIoU
import torchvision

from src.configs.training_config import TrainingConfig
from src.loggers import CombinedLogger, LocalLogger, MlflowLogger
from src.loss.functional.focal_loss import multiclass_focal_loss
from src.loss.functional.tversky_loss import multiclass_tversky_loss
from src.utils.class_weights import ClassWeightTracker
from src.utils.dataloading import create_dataloaders_for_splits
from src.utils.metrics import MetricsTracker


def make_transform(resize_size: int = 224):
    resize = torchvision.transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = torchvision.transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return torchvision.transforms.Compose([resize, normalize])


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    seg_head: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> tuple[float, float]:
    """Evaluate the model and segmentation head on the given dataloader.

    Args:
        model: _description_
        seg_head: _description_
        dataloader: _description_
        device: _description_
        config: _description_

    Returns:
        tuple[float, float]: Mean loss and mean IoU.
    """
    model = model.eval()
    seg_head = seg_head.eval()
    losses = []
    ious = []
    mean_iou = MeanIoU(num_classes=config.num_classes + 1).to(device)

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        features = model.get_intermediate_layers(images, n=1, return_class_token=False)[0]

        outputs = seg_head(features)

        distribution_loss = multiclass_focal_loss(outputs, masks, config.focal_alpha, config.focal_gamma)

        masks_one_hot = torch.nn.functional.one_hot(masks, config.num_classes + 1).permute(0, 3, 1, 2)
        district_loss = multiclass_tversky_loss(outputs, masks_one_hot)

        loss = config.distribution_loss_weight * distribution_loss + config.district_loss_weight * district_loss
        losses.append(loss.item())

        preds = outputs.argmax(dim=1)
        mean_iou.update(preds, masks)

        print(f"Eval Batch {batch_idx}, Loss: {loss.item():.4f}")

    return float(torch.mean(torch.tensor(losses)).item()), float(mean_iou.compute().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/train_dino.yaml", help="Path to config file.")
    args = parser.parse_args()

    config = TrainingConfig.from_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = create_dataloaders_for_splits(config, splits=("train", "val", "test"))

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
    )
    model = model.to(device)
    seg_head = seg_head.to(device)

    transform = make_transform(resize_size=config.resolution[0])

    optimizer = torch.optim.Adam(seg_head.parameters(), lr=config.lr)

    weight_tracker: ClassWeightTracker | None = None
    if config.class_weight_mode != "none":
        weight_tracker = ClassWeightTracker(
            num_classes=config.num_classes + 1,
            mode=config.class_weight_mode,
            ema_decay=config.class_weight_ema_decay,
        )
        if config.class_weight_mode == "offline" and config.class_weights:
            weight_tracker.set_fixed_weights(config.class_weights)

    criterion = 0.0
    for epoch_idx in range(config.num_epochs):
        # Training loop ------------------------------------------------------------------------------------------------
        model.eval()
        seg_head.train()
        mean_iou = MeanIoU(num_classes=config.num_classes + 1).to(device)
        losses = []

        for batch_idx, batch in enumerate(loaders["train"]):
            optimizer.zero_grad()

            images = transform(batch["image"].to(device))
            masks = batch["mask"].to(device)

            with torch.no_grad():
                features = model.get_intermediate_layers(images, n=1, return_class_token=False)[0]

            outputs = seg_head(features)

            class_weights = None
            if weight_tracker:
                weight_tracker.update(masks)
                class_weights = weight_tracker.get_weights(device)

            distribution_loss = multiclass_focal_loss(
                outputs, masks, config.focal_alpha, config.focal_gamma, class_weights
            )

            masks_one_hot = torch.nn.functional.one_hot(masks, config.num_classes + 1).permute(0, 3, 1, 2)
            district_loss = multiclass_tversky_loss(outputs, masks_one_hot)

            loss = config.distribution_loss_weight * distribution_loss + config.district_loss_weight * district_loss
            loss.backward()
            losses.append(loss.cpu().detach().item())

            optimizer.step()

            if config.compute_train_metrics:
                preds = outputs.argmax(dim=1)
                mean_iou.update(preds, masks)

            if config.log_interval and batch_idx % config.log_interval == 0:
                print(f"Epoch {epoch_idx}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        if config.compute_train_metrics:
            train_mean_iou = mean_iou.compute().item()
            training_loss = torch.mean(torch.tensor(losses)).item()
            tracker.add(epoch_idx, "train", training_loss, train_mean_iou)
            logger.log_metrics(tracker.get(epoch_idx, "train")[-1])
            print(f"Epoch {epoch_idx}, Training Loss: {training_loss:.4f}, Mean IoU: {train_mean_iou:.4f}")

        # Training loop ------------------------------------------------------------------------------------------------

        if not config.evaluate:
            continue

        val_stats = evaluate(model, seg_head, loaders["val"], device, config)
        tracker.add(epoch_idx, "val", val_stats[0], val_stats[1])
        logger.log_metrics(tracker.get(epoch_idx, "val")[-1])
        print(f"Epoch {epoch_idx}, Validation Loss: {val_stats[0]:.4f}, Mean IoU: {val_stats[1]:.4f}")

        if not val_stats[1] > criterion:
            continue

        torch.save(seg_head.state_dict(), config.checkpoint_path)
        logger.log_model(seg_head, f"seg_head_epoch_{epoch_idx}")
        criterion = val_stats[1]

        test_stats = evaluate(model, seg_head, loaders["test"], device, config)
        tracker.add(epoch_idx, "test", test_stats[0], test_stats[1])
        logger.log_metrics(tracker.get(epoch_idx, "test")[-1])
        print(f"Epoch {epoch_idx}, Test Loss: {test_stats[0]:.4f}, Mean IoU: {test_stats[1]:.4f}")

    logger.end()

    [px.imshow(mask.cpu()).show() for mask in masks]
    [px.imshow(output.argmax(0).cpu()).show() for output in outputs]

    print()


if __name__ == "__main__":
    main()
