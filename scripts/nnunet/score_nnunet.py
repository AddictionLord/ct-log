import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from src.dataset.kwp_mask import KwpMaskBuilder
from src.eval.knot_instances import S8, THIN_MINOR_AXIS, KnotInstanceEvaluator

N = 778


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score nnU-Net (or M4 npz) knot predictions per log with knot_instances."
    )
    parser.add_argument(
        "--pred_dir", type=Path, required=True, help="nnUNetv2_predict output with ctlog_<log>.npz, or an M4 npz dir"
    )
    parser.add_argument("--kind", choices=["nnunet", "m4"], required=True)
    parser.add_argument("--gt_root", type=Path, required=True, help="kwp_ds_v3 root with <log>/ann")
    parser.add_argument(
        "--pages_dir", type=Path, required=True, help="Dir with ctlog_<log>_pages.json (slice order of the volumes)"
    )
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    builder = KwpMaskBuilder()
    knot_id = KwpMaskBuilder.class_names().index("knot")
    out: Dict[str, dict] = {}
    for log in args.logs:
        pages = json.loads((args.pages_dir / ("ctlog_%s_pages.json" % log)).read_text())
        knot_prob, wood = load_predictions(args.pred_dir, args.kind, log, pages)
        knot = (knot_prob >= 0.5) & wood
        evaluator = KnotInstanceEvaluator()
        per_slice: List[Tuple[str, int, int]] = []
        for z, page in enumerate(pages):
            ann = args.gt_root / log / "ann" / (page + ".json")
            if not ann.exists():
                continue
            gt = builder.build(json.loads(ann.read_text())).numpy() == knot_id
            evaluator.add(gt, knot_prob[z] >= 0.5, wood=wood[z], score_map=knot_prob[z])
            per_slice.append((page, int((knot[z] & gt).sum()), int((knot[z] | gt).sum())))
        r = evaluator.result()
        out[log] = {
            "n_frames": r["n_frames"],
            "knot_iou": r["pixel_iou"],
            "inst_f1_30": r["iou_0.3"]["f1"],
            "map50": r["map50"],
            "map50_95": r["map50_95"],
            "thin_recall": r["recall_thin_iou0.3"]["recall"],
            "n_thin": r["recall_thin_iou0.3"]["n"],
            "thin_map50": r["ap_by_stratum"]["thin"]["map50"],
            "thin_map50_95": r["ap_by_stratum"]["thin"]["map50_95"],
            "thick_map50": r["ap_by_stratum"]["thick"]["map50"],
            "thick_map50_95": r["ap_by_stratum"]["thick"]["map50_95"],
            **isolated_thin(knot),
            "per_slice": per_slice,
        }
        print(log, {k: v for k, v in out[log].items() if k != "per_slice"}, flush=True)
    args.out.write_text(json.dumps(out))


def load_predictions(pred_dir: Path, kind: str, log: str, pages: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if kind == "nnunet":
        probs = np.load(pred_dir / ("ctlog_%s.npz" % log))["probabilities"]
        return probs[2].astype(np.float32), probs.argmax(axis=0) != 0
    knot_prob, wood = [], []
    for page in pages:
        d = np.load(pred_dir / (page + ".npz"))
        knot_prob.append(d["knot_prob"].astype(np.float32) / 255.0)
        wood.append(np.unpackbits(d["wood"])[: N * N].reshape(N, N).astype(bool))
    return np.stack(knot_prob), np.stack(wood)


def isolated_thin(knot: np.ndarray) -> Dict[str, float]:
    thin_total = thin_isolated = 0
    for z in range(knot.shape[0]):
        labels, _ = ndi.label(knot[z], structure=S8)
        neighbours = np.zeros_like(knot[z])
        if z > 0:
            neighbours |= knot[z - 1]
        if z + 1 < knot.shape[0]:
            neighbours |= knot[z + 1]
        for region in measure.regionprops(labels):
            if region.axis_minor_length >= THIN_MINOR_AXIS:
                continue
            thin_total += 1
            rows, cols = region.coords[:, 0], region.coords[:, 1]
            thin_isolated += int(not neighbours[rows, cols].any())
    return {
        "pred_thin_components": thin_total,
        "pred_thin_isolated": thin_isolated,
        "isolated_thin_rate": round(thin_isolated / max(thin_total, 1), 4),
    }


if __name__ == "__main__":
    main()
