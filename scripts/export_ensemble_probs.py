import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import List, Tuple

import numpy as np
from PIL import Image
from scripts.export_knot_probs import knot_probs, load_stack
from src.configs.kwp_training_config import KwpTrainingConfig
from src.dataset.kwp_mask import KwpMaskBuilder
from src.segmentation_head import build_kwp_model
from src.train_kwp import input_channels, make_transform
import torch
import torch.nn.functional as F

PAGE_RE = re.compile(r"page_(\d+)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export averaged knot probabilities of several models for many logs.")
    parser.add_argument("--member", action="append", required=True, help="config.yaml=best_copy.pth")
    parser.add_argument("--data_root", type=Path, required=True, help="Directory with <log>/img/page_*.tiff.")
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--out_root", type=Path, required=True, help="Writes <out_root>/<log>/page_XXX.png.")
    parser.add_argument("--tta", action="store_true", help="8-way rotation/flip averaging per member.")
    parser.add_argument("--batch_size", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda")
    knot_id = KwpMaskBuilder.class_names().index("knot")
    members = [load_member(spec, device) for spec in args.member]
    for log in args.logs:
        out_dir = args.out_root / log
        if (out_dir / ".done").exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        count = export_log(args.data_root / log / "img", members, knot_id, args.tta, args.batch_size, out_dir, device)
        (out_dir / ".done").touch()
        print("log %s: %d slices" % (log, count), flush=True)

    meta = {
        "members": [
            {
                "config": spec.split("=", 1)[0],
                "checkpoint": spec.split("=", 1)[1],
                "sha256": sha256(Path(spec.split("=", 1)[1])),
            }
            for spec in args.member
        ],
        "averaging": "arithmetic mean of the members' knot softmax probabilities",
        "tta": "8-way: rot90 k=0..3 x horizontal flip, averaged per member before the ensemble mean"
        if args.tta
        else "none",
        "input": "3-slice stack [z-1, z, z+1] clamped at the ends, native 778 -> model resolution bilinear, ImageNet normalisation",
        "output": "native 778x778 bilinear, uint8 = round(p * 255); knot mask = png >= 128",
        "logs": args.logs,
    }
    (args.out_root / "META.json").write_text(json.dumps(meta, indent=2))
    print("wrote %s" % args.out_root)


def load_member(
    spec: str, device: torch.device
) -> Tuple[KwpTrainingConfig, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    config_path, checkpoint = spec.split("=", 1)
    config = KwpTrainingConfig.from_yaml(config_path)
    backbone, seg_head, _ = build_kwp_model(config)
    state = torch.load(checkpoint, map_location="cpu")
    seg_head.load_state_dict(state["seg_head"])
    if state.get("backbone") is not None:
        backbone.load_state_dict(state["backbone"], strict=False)
    return config, backbone.to(device).eval(), seg_head.to(device).eval(), make_transform(input_channels(config.window))


def export_log(
    img_dir: Path, members: List, knot_id: int, tta: bool, batch_size: int, out_dir: Path, device: torch.device
) -> int:
    pages = sorted((int(PAGE_RE.search(p.stem).group(1)), p) for p in img_dir.glob("*.tiff") if PAGE_RE.search(p.stem))
    paths = [path for _, path in pages]
    for start in range(0, len(paths), batch_size):
        positions = list(range(start, min(start + batch_size, len(paths))))
        stacks = torch.stack([load_stack(paths, position, 1) for position in positions])
        total = None
        for config, backbone, seg_head, transform in members:
            images = F.interpolate(stacks, size=tuple(config.resolution), mode="bilinear", align_corners=False)
            with torch.no_grad():
                probs = knot_probs(backbone, seg_head, transform(images.to(device)), config, knot_id, tta)
                probs = F.interpolate(probs, size=tuple(stacks.shape[-2:]), mode="bilinear", align_corners=False)
            total = probs if total is None else total + probs
        mean = (total / len(members))[:, 0].cpu().numpy()
        for position, prob in zip(positions, mean):
            image = Image.fromarray(np.round(np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8))
            image.save(out_dir / f"{paths[position].stem.split('.')[0]}.png")
    return len(paths)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
