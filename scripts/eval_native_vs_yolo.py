import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from src.dataset.kwp_mask import KwpMaskBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired per-log block bootstrap of native knot IoU vs YOLO masks.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--yolo_root", type=Path, required=True, help="Dir with yolo_probs_log<log>/page_XXX_mask.png")
    parser.add_argument("--model", action="append", required=True, help="name=dir with <log>/page_XXX.png probs")
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--n_boot", type=int, default=2000)
    args = parser.parse_args()

    models = dict(m.split("=", 1) for m in args.model)
    builder = KwpMaskBuilder()
    knot = KwpMaskBuilder.class_names().index("knot")
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for log in args.logs:
        anns = sorted((args.data_root / log / "ann").glob("*.json"))
        rows = {name: [] for name in ["yolo", *models]}
        for ann in anns:
            page = ann.name.split(".")[0]
            gt = builder.build(json.load(open(ann))).numpy() == knot
            preds = {"yolo": np.asarray(Image.open(args.yolo_root / f"yolo_probs_log{log}" / f"{page}_mask.png")) > 127}
            for name, root in models.items():
                preds[name] = np.asarray(Image.open(Path(root) / log / f"{page}.png")) >= 128
            for name, pred in preds.items():
                rows[name].append(((pred & gt).sum(), (pred | gt).sum()))
        stats[log] = {name: np.array(values, dtype=np.float64) for name, values in rows.items()}
        print("log %s: %d annotated slices" % (log, len(anns)))

    rng = np.random.default_rng(0)
    for scope in [*args.logs, "pooled"]:
        logs = args.logs if scope == "pooled" else [scope]
        report(scope, logs, stats, models, rng, args.block, args.n_boot)


def report(scope: str, logs: List[str], stats, models, rng, block: int, n_boot: int) -> None:
    print("== %s" % scope)
    blocks = {
        log: [np.arange(i, min(i + block, len(stats[log]["yolo"]))) for i in range(0, len(stats[log]["yolo"]), block)]
        for log in logs
    }
    samples = []
    for _ in range(n_boot):
        samples.append(
            {
                log: np.concatenate([blocks[log][j] for j in rng.integers(0, len(blocks[log]), len(blocks[log]))])
                for log in logs
            }
        )

    def iou(name: str, index=None) -> float:
        inter = union = 0.0
        for log in logs:
            s = stats[log][name] if index is None else stats[log][name][index[log]]
            inter += s[:, 0].sum()
            union += s[:, 1].sum()
        return inter / union

    print("  %-16s IoU %.4f" % ("yolo", iou("yolo")))
    for name in models:
        diffs = np.array([iou(name, idx) - iou("yolo", idx) for idx in samples])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(
            "  %-16s IoU %.4f  diff vs yolo %+.4f [%+.4f, %+.4f]" % (name, iou(name), iou(name) - iou("yolo"), lo, hi)
        )


if __name__ == "__main__":
    main()
