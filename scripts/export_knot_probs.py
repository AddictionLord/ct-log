import argparse
from pathlib import Path
import re
from typing import List

import numpy as np
from PIL import Image
from src.configs.kwp_training_config import KwpTrainingConfig
from src.dataset.kwp_mask import KwpMaskBuilder
from src.segmentation_head import build_kwp_model
from src.train_kwp import extract_features, input_channels, make_transform
import torch
import torch.nn.functional as F

PAGE_RE = re.compile(r"page_(\d+)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-slice knot probabilities for every slice of a log.")
    parser.add_argument("--config", type=Path, required=True, help="Training config of the model.")
    parser.add_argument("--best_copy", type=Path, required=True, help="Best-model copy with seg_head (+ backbone).")
    parser.add_argument("--log_dir", type=Path, required=True, help="Log directory with img/page_*.tiff.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for page_XXX.png.")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size.")
    parser.add_argument("--tta", action="store_true", help="Average over the 8 rotations/flips of the input.")
    args = parser.parse_args()

    config = KwpTrainingConfig.from_yaml(args.config)
    device = torch.device("cuda")
    backbone, seg_head, _ = build_kwp_model(config)
    state = torch.load(args.best_copy, map_location="cpu")
    seg_head.load_state_dict(state["seg_head"])
    if state.get("backbone") is not None:
        backbone.load_state_dict(state["backbone"], strict=False)
    backbone, seg_head = backbone.to(device).eval(), seg_head.to(device).eval()
    transform = make_transform(input_channels(config.window))
    knot_id = KwpMaskBuilder.class_names().index("knot")

    pages = sorted(
        (int(PAGE_RE.search(path.stem).group(1)), path)
        for path in (args.log_dir / "img").glob("*.tiff")
        if PAGE_RE.search(path.stem)
    )
    paths = [path for _, path in pages]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(paths), args.batch_size):
        positions = list(range(start, min(start + args.batch_size, len(paths))))
        stacks = torch.stack([load_stack(paths, position, config.window) for position in positions])
        native_size = stacks.shape[-2:]
        images = F.interpolate(stacks, size=tuple(config.resolution), mode="bilinear", align_corners=False)
        with torch.no_grad():
            probs = knot_probs(backbone, seg_head, transform(images.to(device)), config, knot_id, args.tta)
            probs = F.interpolate(probs, size=tuple(native_size), mode="bilinear", align_corners=False)
        for position, prob in zip(positions, probs[:, 0].cpu().numpy()):
            image = Image.fromarray(np.round(np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8))
            image.save(args.out_dir / f"{paths[position].stem.split('.')[0]}.png")
    print("exported %d slices to %s" % (len(paths), args.out_dir))


def knot_probs(
    backbone: torch.nn.Module,
    seg_head: torch.nn.Module,
    images: torch.Tensor,
    config: KwpTrainingConfig,
    knot_id: int,
    tta: bool,
) -> torch.Tensor:
    transforms = [(k, flip) for k in range(4) for flip in (False, True)] if tta else [(0, False)]
    total = None
    for k, flip in transforms:
        view = torch.rot90(images.flip(-1) if flip else images, k, dims=(-2, -1))
        features = extract_features(backbone, view, config.n_layers, config.backbone_bf16)
        probs = seg_head(features).float().softmax(dim=1)[:, knot_id : knot_id + 1]
        probs = torch.rot90(probs, -k, dims=(-2, -1))
        probs = probs.flip(-1) if flip else probs
        total = probs if total is None else total + probs
    return total / len(transforms)


def load_stack(paths: List[Path], position: int, window: int) -> torch.Tensor:
    if window == 0:
        neighbors = [position, position, position]
    else:
        neighbors = [min(max(position + offset, 0), len(paths) - 1) for offset in range(-window, window + 1)]
    slices = [np.asarray(Image.open(paths[index]).convert("L"), dtype=np.float32) / 255.0 for index in neighbors]
    return torch.from_numpy(np.stack(slices))


if __name__ == "__main__":
    main()
