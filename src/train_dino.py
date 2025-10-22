import plotly.express as px
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
    resolution = (320, 320)
    # resolution = (224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CTLogDataset("data/processed/set_24", num_classes=num_classes, resolution=resolution)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
        input_size=resolution[0],
    )
    model = model.to(device)
    seg_head = seg_head.to(device)

    transform = make_transform(resize_size=resolution[0])
    # ----- Using DINOv3 + segmentation head -----

    # Only the segmentation head is trainable; the backbone is frozen.
    optimizer = torch.optim.Adam(seg_head.parameters(), lr=lr)

    for epoch_idx in range(num_epochs):
        # Set proper modes: frozen backbone for feature extraction, trainable head
        model.eval()
        seg_head.train()

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            images = transform(batch["image"].to(device))
            masks = batch["mask"].to(device)

            # Extract features without gradients (backbone is frozen)
            with torch.no_grad():
                features = model.get_intermediate_layers(images, n=1, return_class_token=False)[0]

            outputs = seg_head(features)

            distribution_loss = multiclass_focal_loss(outputs, masks, alpha=2.0, gamma=5.0)

            masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=num_classes + 1).permute(
                0, 3, 1, 2
            )
            district_loss = multiclass_tversky_loss(outputs, masks_one_hot)

            loss = 0.4 * distribution_loss + 0.6 * district_loss
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch_idx}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    px.imshow(masks[0].cpu()).show()
    px.imshow(outputs[0].argmax(0).cpu()).show()

    print()


if __name__ == "__main__":
    main()
