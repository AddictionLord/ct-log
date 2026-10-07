import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List, Optional

import numpy as np
from scripts.export_knot_probs import load_stack
from src.configs.kwp_training_config import KwpTrainingConfig
from src.segmentation_head import build_kwp_model
from src.train_kwp import extract_features, make_transform
import torch
import torch.nn.functional as F

PAGE_RE = re.compile(r"page_(\d+)")
NATIVE = 778


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-slice pith points (pixel index) for every slice of logs.")
    parser.add_argument("--config", type=Path, required=True, help="Training config of the model.")
    parser.add_argument("--best_copy", type=Path, required=True, help="Best-model copy (seg_head, pith_head).")
    parser.add_argument("--data_root", type=Path, required=True, help="Directory with <log>/img/page_*.tiff.")
    parser.add_argument("--logs", nargs="+", required=True, help="Log names to export.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for <log>.json and META.json.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch of the best copy, recorded in META.json.")
    parser.add_argument("--batch_size", type=int, default=6, help="Inference batch size.")
    args = parser.parse_args()

    config = KwpTrainingConfig.from_yaml(args.config)
    device = torch.device("cuda")
    backbone, seg_head, pith_head = build_kwp_model(config)
    if pith_head is None:
        msg = "Config has no pith head (pith_regression is false)"
        raise ValueError(msg)
    state = torch.load(args.best_copy, map_location="cpu")
    seg_head.load_state_dict(state["seg_head"])
    if state.get("backbone") is not None:
        backbone.load_state_dict(state["backbone"], strict=False)
    if state.get("pith_head"):
        pith_head.load_state_dict(state["pith_head"])
    backbone, seg_head, pith_head = backbone.to(device).eval(), seg_head.to(device).eval(), pith_head.to(device).eval()
    transform = make_transform()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for log in args.logs:
        points = export_log(
            args.data_root / log / "img", config, backbone, seg_head, pith_head, transform, device, args.batch_size
        )
        (args.out_dir / f"{log}.json").write_text(json.dumps({"pith": points}, indent=0))
        counts[log] = len(points)
        print("log %s: %d slices" % (log, len(points)), flush=True)

    meta = {
        "model": config.mlflow_run_name,
        "config": str(args.config),
        "checkpoint": str(args.best_copy),
        "checkpoint_sha256": sha256(args.best_copy),
        "epoch": args.epoch,
        "weights": "EMA weights (best copy saved while EMA weights were swapped in)",
        "input_stack": f"[z-1, z, z+1] slices from <log>/img sorted by page index, clamped at the ends (window={config.window})",
        "resize": f"native {NATIVE}x{NATIVE} -> {config.resolution[0]}x{config.resolution[1]} bilinear, ImageNet normalisation",
        "prediction": "soft-argmax of the pith heatmap channel, normalized (x, y) in [0, 1] with pixel centres at (i + 0.5) / W",
        "rounding": f"x_px = rint(x * {NATIVE}), y_px = rint(y * {NATIVE}), clipped to [0, {NATIVE - 1}]; integer pixel index "
        "(Supervisely convention: column x, row y). Same rule as the log-10 evaluation (68% exact, 96% <= 1 px).",
        "null_pages": "none: the model always predicts a point; it does not detect absence of pith",
        "snap": "none",
        "slices_per_log": counts,
    }
    (args.out_dir / "META.json").write_text(json.dumps(meta, indent=2))
    print("wrote %s" % args.out_dir)


def export_log(
    img_dir: Path,
    config: KwpTrainingConfig,
    backbone: torch.nn.Module,
    seg_head: torch.nn.Module,
    pith_head: torch.nn.Module,
    transform: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Optional[List[int]]]:
    pages = sorted((int(PAGE_RE.search(p.stem).group(1)), p) for p in img_dir.glob("*.tiff") if PAGE_RE.search(p.stem))
    paths = [path for _, path in pages]
    points: Dict[str, Optional[List[int]]] = {}
    for start in range(0, len(paths), batch_size):
        positions = list(range(start, min(start + batch_size, len(paths))))
        stacks = torch.stack([load_stack(paths, position, config.window) for position in positions])
        images = F.interpolate(stacks, size=tuple(config.resolution), mode="bilinear", align_corners=False)
        with torch.no_grad():
            features = extract_features(backbone, transform(images.to(device)), config.n_layers, config.backbone_bf16)
            seg_head(features)
            xy = pith_head(features).float().cpu().numpy()
        for position, (x, y) in zip(positions, xy):
            px = int(np.clip(np.rint(x * NATIVE), 0, NATIVE - 1))
            py = int(np.clip(np.rint(y * NATIVE), 0, NATIVE - 1))
            points[paths[position].name] = [px, py]
    return points


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
