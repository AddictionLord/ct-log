import plotly.express as px
from segmentation_head import create_dinov3_segmentor
import torch
import torchvision

from src.configs.training_config import TrainingConfig
from src.loss.functional.focal_loss import multiclass_focal_loss
from src.loss.functional.tversky_loss import multiclass_tversky_loss
from src.utils.dataloading import create_dataloaders_for_splits


def make_transform(resize_size: int = 224):
    resize = torchvision.transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = torchvision.transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return torchvision.transforms.Compose([resize, normalize])


def main() -> None:
    config = TrainingConfig.from_yaml("src/configs/train_dino.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = create_dataloaders_for_splits(config, splits=("train", "val", "test"))

    model, seg_head = create_dinov3_segmentor(
        backbone_weights=config.backbone_weights,
        num_classes=config.num_classes + 1,
        input_size=config.resolution[0],
    )
    model = model.to(device)
    seg_head = seg_head.to(device)

    transform = make_transform(resize_size=config.resolution[0])

    optimizer = torch.optim.Adam(seg_head.parameters(), lr=config.lr)

    for epoch_idx in range(config.num_epochs):
        model.eval()
        seg_head.train()

        for batch_idx, batch in enumerate(loaders["train"]):
            optimizer.zero_grad()

            images = transform(batch["image"].to(device))
            masks = batch["mask"].to(device)

            with torch.no_grad():
                features = model.get_intermediate_layers(images, n=1, return_class_token=False)[0]

            outputs = seg_head(features)

            distribution_loss = multiclass_focal_loss(outputs, masks, config.focal_alpha, config.focal_gamma)

            masks_one_hot = torch.nn.functional.one_hot(masks, config.num_classes + 1).permute(0, 3, 1, 2)
            district_loss = multiclass_tversky_loss(outputs, masks_one_hot)

            loss = config.distribution_loss_weight * distribution_loss + config.district_loss_weight * district_loss
            loss.backward()

            optimizer.step()

            if batch_idx % config.log_interval == 0:
                print(f"Epoch {epoch_idx}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    px.imshow(masks[0].cpu()).show()
    px.imshow(outputs[0].argmax(0).cpu()).show()

    print()


if __name__ == "__main__":
    main()
