import argparse
import json
from pathlib import Path
import re
from typing import Dict, List, Optional

import numpy as np
from scripts.export_knot_probs import load_stack
from src.configs.kwp_training_config import KwpTrainingConfig
from src.dataset.kwp_mask import KwpMaskBuilder
from src.segmentation_head import build_kwp_model
from src.train_kwp import extract_features, input_channels, make_transform
import torch
import torch.nn.functional as F

PAGE_RE = re.compile(r"page_(\d+)")
NATIVE = 778


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-slice knot/wood/pith predictions (single model or mean of several)."
    )
    parser.add_argument(
        "--member",
        action="append",
        required=True,
        help="config.yaml=checkpoint[=pith_source]; checkpoint is a best copy or a raw seg-head state dict",
    )
    parser.add_argument("--log_dir", type=Path, required=True, help="Log directory with img/page_*.tiff.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Writes page_XXX.tiff.npz (knot, wood, pith).")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda")
    members = [load_member(spec, device) for spec in args.member]
    pages = sorted(
        (int(PAGE_RE.search(p.stem).group(1)), p)
        for p in (args.log_dir / "img").glob("*.tiff")
        if PAGE_RE.search(p.stem)
    )
    paths = [path for _, path in pages]
    background = KwpMaskBuilder.class_names().index("background")
    knot_id = KwpMaskBuilder.class_names().index("knot")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(paths), args.batch_size):
        positions = list(range(start, min(start + args.batch_size, len(paths))))
        stacks = torch.stack([load_stack(paths, position, 1) for position in positions])
        probs, piths = predict(members, stacks, device)
        for i, position in enumerate(positions):
            knot = probs[i, knot_id] >= 0.5
            wood = probs[i].argmax(axis=0) != background
            pith = piths[i] if piths is not None else np.array([-1, -1])
            np.savez_compressed(
                args.out_dir / (paths[position].name + ".npz"),
                knot=np.packbits(knot.ravel()),
                wood=np.packbits(wood.ravel()),
                pith=pith.astype(np.int32),
                knot_prob=np.round(np.clip(probs[i, knot_id], 0.0, 1.0) * 255).astype(np.uint8),
            )
    meta = {
        "members": args.member,
        "knot": "mean knot softmax >= 0.5",
        "wood": "argmax of mean softmax != background (wood or knot)",
        "pith": "mean of the members' pith points, rint(p * 778), column x / row y; [-1, -1] if no pith head",
        "resize": "native 778 -> model resolution bilinear; probabilities back to 778 bilinear",
    }
    (args.out_dir / "META.json").write_text(json.dumps(meta, indent=2))
    print("exported %d slices to %s" % (len(paths), args.out_dir))


def load_member(spec: str, device: torch.device) -> Dict:
    parts = spec.split("=")
    config = KwpTrainingConfig.from_yaml(parts[0])
    backbone, seg_head, pith_head = build_kwp_model(config)
    state = torch.load(parts[1], map_location="cpu", weights_only=False)
    if "seg_head" in state:
        seg_head.load_state_dict(state["seg_head"])
        if state.get("backbone") is not None:
            backbone.load_state_dict(state["backbone"], strict=False)
        pith_state = state.get("pith_head")
    else:
        seg_head.load_state_dict(state)
        pith_state = None
    if len(parts) > 2:
        source = torch.load(parts[2], map_location="cpu", weights_only=False)
        pith_state = source["pith_head"] if "pith_head" in source else source
    if pith_head is not None and any(True for _ in pith_head.parameters()):
        if not pith_state:
            pith_head = None
        else:
            pith_head.load_state_dict(pith_state)
    return {
        "config": config,
        "backbone": backbone.to(device).eval(),
        "seg_head": seg_head.to(device).eval(),
        "pith_head": pith_head.to(device).eval() if pith_head is not None else None,
        "transform": make_transform(input_channels(config.window)),
    }


def predict(members: List[Dict], stacks: torch.Tensor, device: torch.device):
    probs_total, piths = None, []
    for member in members:
        config = member["config"]
        images = F.interpolate(stacks, size=tuple(config.resolution), mode="bilinear", align_corners=False)
        with torch.no_grad():
            features = extract_features(
                member["backbone"], member["transform"](images.to(device)), config.n_layers, config.backbone_bf16
            )
            probs = member["seg_head"](features).float().softmax(dim=1)
            probs = F.interpolate(probs, size=(NATIVE, NATIVE), mode="bilinear", align_corners=False)
            if member["pith_head"] is not None:
                piths.append(member["pith_head"](features).float().cpu().numpy())
        probs_total = probs if probs_total is None else probs_total + probs
    mean_pith: Optional[np.ndarray] = None
    if piths:
        mean_pith = np.clip(np.rint(np.mean(piths, axis=0) * NATIVE), 0, NATIVE - 1).astype(np.int32)
    return (probs_total / len(members)).cpu().numpy(), mean_pith


if __name__ == "__main__":
    main()
