from segmentation_head import create_dinov3_segmentor
import torch
import torchvision

from src.dataset.ct_log_dataset import CTLogDataset
from src.loss.functional.focal_loss import multiclass_focal_loss
from src.loss.functional.tversky_loss import multiclass_tversky_loss


def make_transform(resize_size: int = 224):
    resize = torchvision.transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = torchvision.transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return torchvision.transforms.Compose([resize, normalize])


def main() -> None:
    num_classes = 10
    batch_size = 2
    lr = 1e-4
    num_epochs = 50
    # resolution = (458, 530)
    resolution = (224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CTLogDataset("data/processed/set_24", num_classes=num_classes, resolution=resolution)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # # model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
    # #     weights=torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
    # # )
    # # model = torchvision.models.segmentation.deeplabv3_resnet50(
    # #     weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
    # # )
    # model = torchvision.models.segmentation.deeplabv3_resnet101(
    #     weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT,
    # )
    # model.classifier[4] = torch.nn.Conv2d(256, num_classes + 1, kernel_size=1)
    # # model.load_state_dict(torch.load("models/model_weights.pth"))
    # model.train()
    # model.to(device)

    # ----- Using DINOv3 + segmentation head -----
    REPO_DIR = "/home/mary/code/dinov3"
    # model_name = "dinov3_vits16"
    model_name = "dinov3_vitl16"
    # segmentor_model_name = "dinov3_vit7b16_ms"

    # backbone_weights = "/mnt/D/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    backbone_weights = "/mnt/D/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, seg_head = create_dinov3_segmentor(
        backbone_weights=backbone_weights,
        num_classes=11,
    )
    model = model.to(device)
    seg_head = seg_head.to(device)

    transform = make_transform()
    # ----- Using DINOv3 + segmentation head -----

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batch = next(iter(dataloader))
    images = batch["image"].to(device)

    images = transform(images)

    masks = batch["mask"].to(device)
    masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=num_classes + 1).permute(
        0, 3, 1, 2
    )

    for epoch_idx in range(num_epochs):
        # for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training epoch {epoch_idx}")):
        for i in range(1000):
            optimizer.zero_grad()

            # images = batch["image"].to(device)
            # masks = batch["mask"].to(device)

            # outputs = model(images)["out"]

            features = model.get_intermediate_layers(images, n=1, return_class_token=False)[0]
            outputs = seg_head(features)

            # cross_entropy_loss = torch.nn.functional.cross_entropy(outputs, masks)
            distribution_loss = multiclass_focal_loss(outputs, masks, alpha=2.0, gamma=5.0)

            # TODO: Inspect the num_classes + 1 - feels weird.
            # masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=num_classes + 1)
            # district_loss = multiclass_dice_loss(outputs, masks_one_hot.permute(0, 3, 1, 2))
            # district_loss = multiclass_dice_loss(outputs, masks_one_hot)
            district_loss = multiclass_tversky_loss(outputs, masks_one_hot)

            loss = 0.4 * distribution_loss + 0.6 * district_loss
            # loss = dice_loss
            loss.backward()
            optimizer.step()

            # print(f"Epoch {epoch_idx}, Batch {batch_idx}, Loss: {loss.item()}")
            print(f"Epoch {epoch_idx}, Batch {i}, Loss: {loss.item()}")

    print()


if __name__ == "__main__":
    main()
