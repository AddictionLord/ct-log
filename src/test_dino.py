from PIL import Image
from segmentation_head import create_dinov3_segmentor
import torch
import torchvision


def make_transform(resize_size: int = 224):
    to_tensor = torchvision.transforms.ToTensor()
    resize = torchvision.transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = torchvision.transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return torchvision.transforms.Compose([to_tensor, resize, normalize])


def main() -> None:
    REPO_DIR = "/home/mary/code/dinov3"
    # model_name = "dinov3_vits16"
    model_name = "dinov3_vitl16"
    # segmentor_model_name = "dinov3_vit7b16_ms"

    # backbone_weights = "/mnt/D/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    backbone_weights = "/mnt/D/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    segmentor_weights = "/mnt/D/models/dinov3/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"
    img_path = "/mnt/D/datasets/intelligent_chroma_key/coco/Soccer/test/JPEGImages/1.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = torch.hub.load(
    #     REPO_DIR,
    #     model_name,
    #     source="local",
    #     pretrained=True,
    #     weights=backbone_weights,
    # ).to(device)

    model, seg_head = create_dinov3_segmentor(
        backbone_weights=backbone_weights,
        num_classes=11,
    )
    model = model.to(device)
    seg_head = seg_head.to(device)

    # model = torch.hub.load(
    #     REPO_DIR,
    #     model_name,
    #     source="local",
    #     weights=segmentor_weights,
    #     backbone_weights=backbone_weights,
    # ).to(device)

    transform = make_transform()

    image = Image.open(img_path).convert("RGB")
    image = transform(image).to(device)

    # out = model(image.unsqueeze(0))
    features = model.get_intermediate_layers(image.unsqueeze(0), n=1, return_class_token=False)[0]
    out = seg_head(features)

    print(out)


if __name__ == "__main__":
    main()
