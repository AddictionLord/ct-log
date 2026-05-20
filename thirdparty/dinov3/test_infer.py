import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision


def make_transform(resize_size: int = 224):
    """Create image preprocessing transforms."""
    to_tensor = torchvision.transforms.ToTensor()
    resize = torchvision.transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = torchvision.transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return torchvision.transforms.Compose([to_tensor, resize, normalize])


def visualize_patch_features(
    patch_features: torch.Tensor, original_image: torch.Tensor, num_components: int = 6
) -> None:
    """
    Visualize patch features using PCA to reduce dimensionality.

    Args:
        patch_features: Patch features from DINOv3 [1, num_patches, feature_dim]
        original_image: Original normalized image tensor [C, H, W]
        num_components: Number of PCA components to visualize
    """
    # Convert to numpy and reshape
    features = patch_features.squeeze(0).cpu().detach().numpy()  # [num_patches, feature_dim]
    patch_size = int(np.sqrt(features.shape[0]))  # 14 for 224x224 input

    # Apply PCA to reduce dimensionality
    from sklearn.decomposition import PCA

    pca = PCA(n_components=num_components)
    features_pca = pca.fit_transform(features)  # [num_patches, num_components]

    # Reshape to spatial dimensions
    features_spatial = features_pca.reshape(patch_size, patch_size, num_components)

    # Denormalize original image for visualization
    mean = torch.tensor([0.485, 0.456, 0.406], device=original_image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=original_image.device).view(3, 1, 1)
    original_denorm = original_image * std + mean
    original_denorm = torch.clamp(original_denorm, 0, 1)

    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Show original image
    axes[0, 0].imshow(original_denorm.permute(1, 2, 0).cpu())
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Show PCA components
    for i in range(min(num_components, 6)):
        row = i // 4
        col = (i + 1) % 4
        component = features_spatial[:, :, i]
        im = axes[row, col].imshow(component, cmap="viridis")
        axes[row, col].set_title(f"PCA Component {i + 1}")
        axes[row, col].axis("off")
        plt.colorbar(im, ax=axes[row, col], fraction=0.046)

    # Hide unused subplots
    for i in range(num_components + 1, 8):
        row = i // 4
        col = i % 4
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_attention_maps(model: torch.nn.Module, image: torch.Tensor, layer_idx: int = -1) -> None:
    """
    Visualize attention maps from the model.

    Args:
        model: DINOv3 model
        image: Input image tensor [C, H, W]
        layer_idx: Which layer's attention to visualize (-1 for last layer)
    """
    with torch.no_grad():
        # Get attention weights
        attentions = model.get_intermediate_layers(
            image.unsqueeze(0), n=1, return_class_token=True, norm=False, reshape=False
        )

        # Extract attention from the specified layer
        if hasattr(model, "blocks"):
            attention_weights = model.blocks[layer_idx].attn.get_attention_map()
            if attention_weights is not None:
                # Focus on CLS token attention to patches
                cls_attention = attention_weights[0, :, 0, 5:].mean(0)  # Skip CLS and register tokens
                patch_size = int(np.sqrt(cls_attention.shape[0]))
                attention_map = cls_attention.reshape(patch_size, patch_size).cpu().numpy()

                # Denormalize original image
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                original_denorm = image * std + mean
                original_denorm = torch.clamp(original_denorm, 0, 1)

                # Visualize
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                axes[0].imshow(original_denorm.permute(1, 2, 0))
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                im = axes[1].imshow(attention_map, cmap="hot")
                axes[1].set_title("CLS Token Attention Map")
                axes[1].axis("off")
                plt.colorbar(im, ax=axes[1])

                plt.tight_layout()
                plt.show()


def analyze_feature_similarity(patch_features: torch.Tensor, original_image: torch.Tensor) -> None:
    """
    Analyze similarity between patch features to understand spatial relationships.

    Args:
        patch_features: Patch features from DINOv3 [1, num_patches, feature_dim]
        original_image: Original image tensor [C, H, W]
    """
    features = patch_features.squeeze(0)  # [num_patches, feature_dim]
    patch_size = int(np.sqrt(features.shape[0]))

    # Compute cosine similarity matrix
    features_norm = torch.nn.functional.normalize(features, dim=1)
    similarity_matrix = torch.mm(features_norm, features_norm.t())

    # Select a few reference patches and show their similarities
    reference_patches = [0, patch_size // 2, patch_size * patch_size // 2, -1]  # corners and center

    fig, axes = plt.subplots(2, len(reference_patches), figsize=(16, 8))

    # Denormalize original image
    mean = torch.tensor([0.485, 0.456, 0.406], device=original_image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=original_image.device).view(3, 1, 1)
    original_denorm = original_image * std + mean
    original_denorm = torch.clamp(original_denorm, 0, 1)

    for i, ref_patch in enumerate(reference_patches):
        # Show original image with reference patch highlighted
        axes[0, i].imshow(original_denorm.permute(1, 2, 0).cpu())
        ref_row, ref_col = ref_patch // patch_size, ref_patch % patch_size
        patch_h, patch_w = original_image.shape[1] // patch_size, original_image.shape[2] // patch_size
        rect = plt.Rectangle(
            (ref_col * patch_w, ref_row * patch_h), patch_w, patch_h, linewidth=2, edgecolor="red", facecolor="none"
        )
        axes[0, i].add_patch(rect)
        axes[0, i].set_title(f"Reference Patch {ref_patch}")
        axes[0, i].axis("off")

        # Show similarity map
        similarity_map = similarity_matrix[ref_patch].reshape(patch_size, patch_size).cpu().detach().numpy()
        im = axes[1, i].imshow(similarity_map, cmap="coolwarm", vmin=0, vmax=1)
        axes[1, i].set_title(f"Similarity to Patch {ref_patch}")
        axes[1, i].axis("off")
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)

    plt.tight_layout()
    plt.show()


def main() -> None:
    REPO_DIR = "/home/mary/code/dinov3"
    model_name = "dinov3_vitl16"
    backbone_weights = "/mnt/D/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    img_path = "/mnt/D/datasets/intelligent_chroma_key/coco/Soccer/test/JPEGImages/1.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load(
        REPO_DIR,
        model_name,
        source="local",
        pretrained=True,
        weights=backbone_weights,
    ).to(device)

    transform = make_transform()
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).to(device)

    # Get patch features
    patch_features = model.get_intermediate_layers(image_tensor.unsqueeze(0), n=1, return_class_token=False)[0]

    print(f"Patch features shape: {patch_features.shape}")

    # Visualize features
    print("Visualizing PCA components of patch features...")
    visualize_patch_features(patch_features, image_tensor)

    print("Analyzing feature similarity patterns...")
    analyze_feature_similarity(patch_features, image_tensor)

    # Uncomment if you want to see attention maps (requires model modifications)
    # visualize_attention_maps(model, image_tensor)


if __name__ == "__main__":
    main()
