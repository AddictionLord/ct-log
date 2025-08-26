import torch
import torchvision
from tqdm import tqdm

from src.dataset.ct_log_dataset import CTLogDataset
from src.loss.functional.dice_loss import multiclass_dice_loss


def main() -> None:
    num_classes = 20
    batch_size = 2
    lr = 1e-4
    num_epochs = 10
    # resolution = (458, 530)
    resolution = (224, 260)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CTLogDataset("data/processed/set_24", num_classes=num_classes, resolution=resolution)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
    #     weights=torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
    # )
    # model = torchvision.models.segmentation.deeplabv3_resnet50(
    #     weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
    # )
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT,
    )
    model.classifier[4] = torch.nn.Conv2d(256, num_classes + 1, kernel_size=1)
    model.load_state_dict(torch.load("model_weights.pth"))
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batch = next(iter(dataloader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=num_classes + 1).permute(0, 3, 1, 2)

    for epoch_idx in range(num_epochs):
        # for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training epoch {epoch_idx}")):
        for i in range(1000):
            optimizer.zero_grad()

            # images = batch["image"].to(device)
            # masks = batch["mask"].to(device)

            outputs = model(images)["out"]

            # cross_entropy_loss = torch.nn.functional.cross_entropy(outputs, masks)

            # TODO: Inspect the num_classes + 1 - feels weird.
            # masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=num_classes + 1)
            # dice_loss = multiclass_dice_loss(outputs, masks_one_hot.permute(0, 3, 1, 2))
            dice_loss = multiclass_dice_loss(outputs, masks_one_hot)

            # loss = cross_entropy_loss + dice_loss
            loss = dice_loss
            loss.backward()
            optimizer.step()

            # print(f"Epoch {epoch_idx}, Batch {batch_idx}, Loss: {loss.item()}")
            print(f"Epoch {epoch_idx}, Batch {i}, Loss: {loss.item()}")

    print()


if __name__ == "__main__":
    main()
