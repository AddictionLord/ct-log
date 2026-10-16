import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List

import numpy as np
from scripts.eval_ensemble_pith_wood import load_member
from scripts.export_knot_probs import load_stack
from src.train_kwp import extract_features
import torch
import torch.nn.functional as F

PAGE_RE = re.compile(r"page_(\d+)")
NATIVE = 778


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-slice ensemble pith points (pixel index) for logs.")
    parser.add_argument("--member", action="append", required=True, help="name=config.yaml=best_copy.pth")
    parser.add_argument("--data_root", type=Path, required=True, help="Directory with <log>/img/page_*.tiff.")
    parser.add_argument("--logs", nargs="+", required=True, help="Log names to export.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for <log>.json and META.json.")
    parser.add_argument("--batch_size", type=int, default=6, help="Inference batch size.")
    args = parser.parse_args()

    device = torch.device("cuda")
    members = [load_member(spec, device) for spec in args.member]
    windows = {m["config"].window for m in members}
    resolutions = {tuple(m["config"].resolution) for m in members}
    if len(windows) != 1 or len(resolutions) != 1:
        msg = "Members must share window and resolution, got %s and %s" % (windows, resolutions)
        raise ValueError(msg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for log in args.logs:
        points = export_log(args.data_root / log / "img", members, device, args.batch_size)
        (args.out_dir / f"{log}.json").write_text(json.dumps({"pith": points}, indent=0))
        counts[log] = len(points)
        print("log %s: %d slices" % (log, len(points)), flush=True)

    config = members[0]["config"]
    meta = {
        "members": [
            {
                "name": spec.split("=", 2)[0],
                "config": spec.split("=", 2)[1],
                "checkpoint": spec.split("=", 2)[2],
                "checkpoint_sha256": sha256(Path(spec.split("=", 2)[2])),
            }
            for spec in args.member
        ],
        "input_stack": f"[z-{config.window} .. z+{config.window}] slices from <log>/img sorted by page index, clamped at the ends",
        "resize": f"native {NATIVE}x{NATIVE} -> {config.resolution[0]}x{config.resolution[1]} bilinear, ImageNet normalisation",
        "prediction": "per member: soft-argmax of the pith heatmap channel, normalized (x, y); ensemble: arithmetic mean of "
        "the members' points in native pixels (x * 778), as in scripts/eval_ensemble_pith_wood.py",
        "rounding": f"rint of the mean point, clipped to [0, {NATIVE - 1}]; integer pixel index (Supervisely convention: "
        "column x, row y)",
        "tta": "none",
        "null_pages": "none: the models always predict a point",
        "snap": "none",
        "slices_per_log": counts,
    }
    (args.out_dir / "META.json").write_text(json.dumps(meta, indent=2))
    print("wrote %s" % args.out_dir)


def export_log(img_dir: Path, members: List[dict], device: torch.device, batch_size: int) -> Dict[str, List[int]]:
    pages = sorted((int(PAGE_RE.search(p.stem).group(1)), p) for p in img_dir.glob("*.tiff") if PAGE_RE.search(p.stem))
    paths = [path for _, path in pages]
    config = members[0]["config"]
    points: Dict[str, List[int]] = {}
    for start in range(0, len(paths), batch_size):
        positions = list(range(start, min(start + batch_size, len(paths))))
        stacks = torch.stack([load_stack(paths, position, config.window) for position in positions])
        images = F.interpolate(stacks, size=tuple(config.resolution), mode="bilinear", align_corners=False)
        member_xy = []
        with torch.no_grad():
            for member in members:
                features = extract_features(
                    member["backbone"],
                    member["transform"](images.to(device)),
                    member["config"].n_layers,
                    member["config"].backbone_bf16,
                )
                member["seg_head"](features)
                member_xy.append(member["pith_head"](features).float().cpu().numpy() * NATIVE)
        mean_xy = np.mean(member_xy, axis=0)
        for position, (x, y) in zip(positions, mean_xy):
            px = int(np.clip(np.rint(x), 0, NATIVE - 1))
            py = int(np.clip(np.rint(y), 0, NATIVE - 1))
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
