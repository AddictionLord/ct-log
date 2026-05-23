"""80/20 holdout evaluation: YOLO+SAM2 vs anchor propagation on subset 4.

Holdout split (deterministic, every 5th annotated page):
    holdout (9): [7, 49, 103, 152, 175, 214, 236, 247, 285]
    train  (36): the rest

Both methods are evaluated only on the 9 holdout pages, neither having seen
them during training/anchor-fitting:
  * YOLO:        retrained without holdouts; predicts knot bboxes -> SAM2 image
                 predictor produces per-instance masks.
  * Propagation: re-run without holdouts as anchors; per-frame semantic label
                 map then decomposed into per-knot connected components.

Per-knot Hungarian matching is by IoU (cost = 1 - IoU); a match requires
IoU >= --iou_thr (default 0.30).

Reported metrics:
  - knot precision / recall / F1 per regime (anchor-distance <= 5 vs > 5)
  - knot mean IoU / Dice on matched instances
  - pith centroid distance (px) vs GT point
  - PR sweep for YOLO across conf thresholds (single number for propagation)
  - per-size stratification (small <200 px, medium 200-2000, large >=2000)

Run from repo root:
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.holdout_eval
"""

import argparse
import json
import pathlib
from typing import Dict, List, Optional, Tuple

from ann_pipeline.knot.data_prep import knot_mask_from_ann
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor_npz
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import torch
from ultralytics import YOLO

from experiments.sm2025_subset4_propagate.run import (
    CLASS_IDS,
    list_pages,
    load_tiff_gray,
    page_ann_path,
    page_img_path,
    propagate_segment,
    render_annotation,
)

HOLDOUT_PAGES = [7, 49, 103, 152, 175, 214, 236, 247, 285]

CHECKPOINT = "/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt"
MODEL_CFG_PATH = "/home/mary/code/ct-log/thirdparty/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"
OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/holdout_eval"

SIZE_BINS = [("small", 0, 200), ("medium", 200, 2000), ("large", 2000, 10**9)]


def split_components(mask: np.ndarray, min_px: int = 8) -> List[np.ndarray]:
    """Return a list of binary masks, one per connected component."""
    lab, n = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    out = []
    for k in range(1, n + 1):
        comp = lab == k
        if comp.sum() >= min_px:
            out.append(comp)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2.0 * inter / denom) if denom > 0 else 0.0


def match_components(
    preds: List[np.ndarray], gts: List[np.ndarray], iou_thr: float
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """IoU-based Hungarian matching. Returns (matches[(pi, gi, iou)], unmatched_pred, unmatched_gt)."""
    if not preds and not gts:
        return [], [], []
    if not preds:
        return [], [], list(range(len(gts)))
    if not gts:
        return [], list(range(len(preds))), []
    iou_mat = np.zeros((len(preds), len(gts)), dtype=np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou_mat[i, j] = iou(p, g)
    cost = 1.0 - iou_mat
    ri, ci = linear_sum_assignment(cost)
    matches = []
    matched_p, matched_g = set(), set()
    for r, c in zip(ri, ci):
        if iou_mat[r, c] >= iou_thr:
            matches.append((int(r), int(c), float(iou_mat[r, c])))
            matched_p.add(int(r))
            matched_g.add(int(c))
    return (
        matches,
        [i for i in range(len(preds)) if i not in matched_p],
        [j for j in range(len(gts)) if j not in matched_g],
    )


def size_bin(n_px: int) -> str:
    for name, lo, hi in SIZE_BINS:
        if lo <= n_px < hi:
            return name
    return "large"


def nearest_anchor_distance(page: int, anchors: List[int]) -> int:
    return min(abs(page - a) for a in anchors)


def run_propagation_holdout(holdouts: List[int], anchor_pages: List[int], size: int = 512) -> Dict[int, np.ndarray]:
    """Propagate across the full subset 4 volume using `anchor_pages` (already
    holdout-free) and return the prediction array for each holdout page."""
    pages = list_pages()
    page_to_idx = {p: i for i, p in enumerate(pages)}
    anchor_idx = sorted(page_to_idx[p] for p in anchor_pages)
    anchors = [pages[i] for i in anchor_idx]
    print("frames: %d  kept anchors (%d)  holdouts (%d)" % (len(pages), len(anchors), len(holdouts)))

    sample = load_tiff_gray(page_img_path(pages[0]))
    h, w = sample.shape
    vol = np.stack([load_tiff_gray(page_img_path(p)) for p in pages])

    anchor_masks: Dict[int, Dict[str, np.ndarray]] = {}
    anchor_pith: Dict[int, List[Tuple[int, int]]] = {}
    for p in anchors:
        m, pith = render_annotation(page_ann_path(p), h, w)
        anchor_masks[p] = m
        anchor_pith[p] = pith

    repo_root = pathlib.Path(__file__).resolve().parents[2] / "thirdparty" / "MedSAM2"
    model_cfg = "//" + str(repo_root / "sam2/configs/sam2.1_hiera_t512.yaml")
    predictor = build_sam2_video_predictor_npz(model_cfg, CHECKPOINT)

    pred = np.zeros_like(vol, dtype=np.uint8)
    first_a = anchor_idx[0]
    if first_a > 0:
        sub = vol[: first_a + 1]
        pa = anchors[0]
        seg = propagate_segment(
            predictor,
            sub,
            seed_masks_a={c: np.zeros((h, w), np.uint8) for c in CLASS_IDS},
            seed_pith_a=[],
            seed_masks_b=anchor_masks[pa],
            seed_pith_b=anchor_pith[pa],
            size=size,
        )
        pred[: first_a + 1] = seg

    for s in range(len(anchor_idx) - 1):
        a, b = anchor_idx[s], anchor_idx[s + 1]
        sub = vol[a : b + 1]
        pa, pb = anchors[s], anchors[s + 1]
        seg = propagate_segment(
            predictor,
            sub,
            seed_masks_a=anchor_masks[pa],
            seed_pith_a=anchor_pith[pa],
            seed_masks_b=anchor_masks[pb],
            seed_pith_b=anchor_pith[pb],
            size=size,
        )
        pred[a : b + 1] = seg
        print("  segment %d..%d done" % (pa, pb))

    last_a = anchor_idx[-1]
    if last_a < len(pages) - 1:
        sub = vol[last_a:]
        pa = anchors[-1]
        seg = propagate_segment(
            predictor,
            sub,
            seed_masks_a=anchor_masks[pa],
            seed_pith_a=anchor_pith[pa],
            seed_masks_b=None,
            seed_pith_b=None,
            size=size,
        )
        pred[last_a:] = seg

    return {p: pred[page_to_idx[p]] for p in holdouts}, pages, pred


def mask_nms(masks: List[np.ndarray], confs: List[float], iou_thr: float) -> Tuple[List[np.ndarray], List[float]]:
    """Greedy mask-IoU NMS: keep highest-conf, drop any later mask with IoU>=thr against a kept one."""
    if not masks:
        return [], []
    order = sorted(range(len(masks)), key=lambda i: -confs[i])
    kept_idx: List[int] = []
    for i in order:
        drop = False
        for k in kept_idx:
            if iou(masks[i], masks[k]) >= iou_thr:
                drop = True
                break
        if not drop:
            kept_idx.append(i)
    return [masks[k] for k in kept_idx], [confs[k] for k in kept_idx]


def yolo_sam_predict(
    yolo_model: YOLO,
    sam_predictor: SAM2ImagePredictor,
    img_path: str,
    conf: float,
    yolo_nms_iou: float = 0.7,
    mask_nms_iou: Optional[float] = None,
) -> Tuple[List[np.ndarray], List[float], Optional[Tuple[float, float]]]:
    """Returns (knot_masks_per_instance, knot_confs, pith_point_or_None).

    - yolo_nms_iou: bbox-IoU threshold for YOLO's internal NMS (default 0.7).
    - mask_nms_iou: if set, apply greedy mask-IoU NMS to SAM2 outputs at this threshold.
    """
    arr = np.array(Image.open(img_path))
    if arr.ndim == 2:
        rgb = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        rgb = arr[..., :3]
    else:
        rgb = arr
    rgb = rgb.astype(np.uint8)

    res = yolo_model.predict(rgb, conf=conf, iou=yolo_nms_iou, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return [], [], None
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()

    knot_boxes, knot_confs = [], []
    pith_pt: Optional[Tuple[float, float]] = None
    best_pith_conf = -1.0
    for b, c, cf in zip(xyxy, cls, confs):
        if c == 0:
            knot_boxes.append(b)
            knot_confs.append(float(cf))
        elif c == 1 and cf > best_pith_conf:
            best_pith_conf = float(cf)
            pith_pt = ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)

    knot_masks: List[np.ndarray] = []
    if knot_boxes:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam_predictor.set_image(rgb)
            for box in knot_boxes:
                masks, _, _ = sam_predictor.predict(box=np.array(box, dtype=np.float32), multimask_output=False)
                knot_masks.append(masks[0].astype(bool))

    if mask_nms_iou is not None and knot_masks:
        knot_masks, knot_confs = mask_nms(knot_masks, knot_confs, mask_nms_iou)

    return knot_masks, knot_confs, pith_pt


def gt_pith_point(ann_path: str) -> Optional[Tuple[float, float]]:
    with open(ann_path) as f:
        ann = json.load(f)
    for obj in ann.get("objects", []):
        if obj.get("classTitle") == "Pith":
            pts = obj.get("points", {}).get("exterior", [])
            if pts:
                return float(pts[0][0]), float(pts[0][1])
    return None


def gt_knot_components(ann_path: str) -> List[np.ndarray]:
    with open(ann_path) as f:
        ann = json.load(f)
    mask = knot_mask_from_ann(ann)
    return split_components(mask.astype(bool))


def evaluate(
    yolo_results: Dict[int, Tuple[List[np.ndarray], List[float], Optional[Tuple[float, float]]]],
    prop_results: Dict[int, np.ndarray],
    train_anchors: List[int],
    iou_thr: float,
    out_dir: pathlib.Path,
) -> None:
    rows = []
    matched_by_method: Dict[str, List[Tuple[float, float, int, int]]] = {"yolo": [], "prop": []}

    for page in HOLDOUT_PAGES:
        ann_path = page_ann_path(page)
        gt_components = gt_knot_components(ann_path)
        gt_pith = gt_pith_point(ann_path)
        dist_anchor = nearest_anchor_distance(page, train_anchors)
        regime = "near" if dist_anchor <= 5 else "far"

        yolo_masks, yolo_confs, yolo_pith = yolo_results[page]
        prop_pred = prop_results[page]
        prop_components = split_components(prop_pred == CLASS_IDS["Knot"])

        for method, preds in [("yolo", yolo_masks), ("prop", prop_components)]:
            matches, unmatched_p, unmatched_g = match_components(preds, gt_components, iou_thr)
            tp = len(matches)
            fp = len(unmatched_p)
            fn = len(unmatched_g)
            mean_iou = float(np.mean([m[2] for m in matches])) if matches else float("nan")
            mean_dice = (
                float(np.mean([dice(preds[pi], gt_components[gi]) for pi, gi, _ in matches]))
                if matches
                else float("nan")
            )
            for pi, gi, iv in matches:
                matched_by_method[method].append(
                    (iv, dice(preds[pi], gt_components[gi]), int(gt_components[gi].sum()), dist_anchor)
                )

            pith_pred: Optional[Tuple[float, float]] = None
            if method == "yolo":
                pith_pred = yolo_pith
            else:
                ys, xs = np.nonzero(prop_pred == CLASS_IDS["Pith"])
                if len(xs) > 0:
                    pith_pred = (float(xs.mean()), float(ys.mean()))

            pith_dist = float("nan")
            if pith_pred is not None and gt_pith is not None:
                pith_dist = float(np.hypot(pith_pred[0] - gt_pith[0], pith_pred[1] - gt_pith[1]))

            rows.append(
                {
                    "page": page,
                    "method": method,
                    "dist_to_anchor": dist_anchor,
                    "regime": regime,
                    "n_gt": len(gt_components),
                    "n_pred": len(preds),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
                    "recall": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
                    "mean_iou": mean_iou,
                    "mean_dice": mean_dice,
                    "pith_dist_px": pith_dist,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_page.csv", index=False)

    summary_rows = []
    for method in ("yolo", "prop"):
        for regime in ("all", "near", "far"):
            sub = df[df.method == method]
            if regime != "all":
                sub = sub[sub.regime == regime]
            tp = sub.tp.sum()
            fp = sub.fp.sum()
            fn = sub.fn.sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            f1 = 2 * precision * recall / (precision + recall) if precision and recall else float("nan")
            summary_rows.append(
                {
                    "method": method,
                    "regime": regime,
                    "n_frames": sub.page.nunique(),
                    "n_gt": int(sub.n_gt.sum()),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mean_iou": float(sub.mean_iou.dropna().mean()) if sub.mean_iou.notna().any() else float("nan"),
                    "mean_dice": float(sub.mean_dice.dropna().mean()) if sub.mean_dice.notna().any() else float("nan"),
                    "pith_dist_mean_px": float(sub.pith_dist_px.dropna().mean())
                    if sub.pith_dist_px.notna().any()
                    else float("nan"),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("\nper-page:")
    print(df.to_string(index=False))
    print("\nsummary:")
    print(summary.to_string(index=False))

    size_rows = []
    for method, matches in matched_by_method.items():
        for name, lo, hi in SIZE_BINS:
            sub = [m for m in matches if lo <= m[2] < hi]
            size_rows.append(
                {
                    "method": method,
                    "size_bin": name,
                    "n_matched": len(sub),
                    "mean_iou": float(np.mean([m[0] for m in sub])) if sub else float("nan"),
                    "mean_dice": float(np.mean([m[1] for m in sub])) if sub else float("nan"),
                }
            )
    size_df = pd.DataFrame(size_rows)
    size_df.to_csv(out_dir / "size_strat.csv", index=False)
    print("\nmatched-instance quality by knot size:")
    print(size_df.to_string(index=False))


def yolo_pr_sweep(
    yolo_path: str,
    img_paths: Dict[int, str],
    gt_components_by_page: Dict[int, List[np.ndarray]],
    iou_thr: float,
    out_dir: pathlib.Path,
) -> None:
    """For YOLO only: sweep confidence and report PR. Uses bbox IoU (not SAM mask)
    so we don't need to re-run SAM for every conf threshold."""
    model = YOLO(yolo_path)
    rows = []
    for conf in np.linspace(0.05, 0.9, 18):
        tp = fp = fn = 0
        for page, img_path in img_paths.items():
            arr = np.array(Image.open(img_path))
            if arr.ndim == 2:
                rgb = np.stack([arr] * 3, axis=-1)
            else:
                rgb = arr[..., :3]
            res = model.predict(rgb.astype(np.uint8), conf=float(conf), verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                pred_bboxes = []
            else:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                pred_bboxes = [tuple(b) for b, c in zip(xyxy, cls) if c == 0]
            gt_bboxes = []
            for comp in gt_components_by_page[page]:
                ys, xs = np.nonzero(comp)
                gt_bboxes.append((float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())))

            if not pred_bboxes and not gt_bboxes:
                continue
            if not pred_bboxes:
                fn += len(gt_bboxes)
                continue
            if not gt_bboxes:
                fp += len(pred_bboxes)
                continue

            iou_mat = np.zeros((len(pred_bboxes), len(gt_bboxes)), dtype=np.float32)
            for i, pb in enumerate(pred_bboxes):
                for j, gb in enumerate(gt_bboxes):
                    iou_mat[i, j] = _bbox_iou(pb, gb)
            ri, ci = linear_sum_assignment(1.0 - iou_mat)
            matched_p, matched_g = set(), set()
            for r, c in zip(ri, ci):
                if iou_mat[r, c] >= iou_thr:
                    matched_p.add(int(r))
                    matched_g.add(int(c))
            tp += len(matched_p)
            fp += len(pred_bboxes) - len(matched_p)
            fn += len(gt_bboxes) - len(matched_g)

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else float("nan")
        rows.append(
            {"conf": float(conf), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
        )

    pr_df = pd.DataFrame(rows)
    pr_df.to_csv(out_dir / "yolo_pr_sweep.csv", index=False)
    print("\nYOLO PR sweep (bbox IoU):")
    print(pr_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(pr_df.recall, pr_df.precision, "-o")
    for _, r in pr_df.iterrows():
        ax.annotate("%.2f" % r["conf"], (r["recall"], r["precision"]), fontsize=7)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("YOLO knot PR (bbox IoU>=%.2f, 9 holdout pages)" % iou_thr)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "yolo_pr_curve.png", dpi=120)
    plt.close()


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolo_weights",
        required=True,
        help="YOLO weights retrained without subset-4 holdouts.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou_thr", type=float, default=0.30)
    parser.add_argument(
        "--yolo_nms_iou",
        type=float,
        default=0.7,
        help="bbox-IoU threshold for YOLO's internal NMS (default 0.7 == ultralytics default).",
    )
    parser.add_argument(
        "--mask_nms_iou",
        type=float,
        default=None,
        help="If set, apply greedy mask-IoU NMS to SAM2 outputs at this threshold (e.g. 0.5).",
    )
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument(
        "--prop_npz",
        default=None,
        help="Optional path to a saved npz with keys {page, pred} for the holdout-free propagation. "
        "If unset, re-runs propagation.",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_annotated = [
        7,
        18,
        28,
        35,
        39,
        49,
        64,
        74,
        84,
        101,
        103,
        111,
        120,
        136,
        145,
        152,
        154,
        157,
        166,
        167,
        175,
        188,
        193,
        196,
        201,
        214,
        223,
        232,
        234,
        235,
        236,
        238,
        241,
        243,
        245,
        247,
        254,
        257,
        265,
        276,
        285,
        291,
        293,
        295,
        297,
    ]
    train_anchors = [p for p in all_annotated if p not in HOLDOUT_PAGES]
    print("train anchors (%d): %s" % (len(train_anchors), train_anchors))
    print("holdout pages (%d): %s" % (len(HOLDOUT_PAGES), HOLDOUT_PAGES))

    if args.prop_npz and pathlib.Path(args.prop_npz).exists():
        data = np.load(args.prop_npz, allow_pickle=False)
        prop_pages = data["pages"].tolist()
        prop_pred_arr = data["pred"]
        prop_results = {p: prop_pred_arr[prop_pages.index(p)] for p in HOLDOUT_PAGES}
        print("loaded propagation predictions from", args.prop_npz)
    else:
        prop_results, pages, full_pred = run_propagation_holdout(HOLDOUT_PAGES, train_anchors)
        npz_path = out_dir / "prop_pred_holdout_free.npz"
        np.savez_compressed(npz_path, pages=np.array(pages), pred=full_pred)
        print("saved propagation predictions to %s" % npz_path)

    yolo_model = YOLO(args.yolo_weights)
    print("building SAM2 image predictor ...")
    cfg = "//" + MODEL_CFG_PATH
    sam = build_sam2(cfg, CHECKPOINT)
    sam_predictor = SAM2ImagePredictor(sam)

    yolo_results: Dict[int, Tuple[List[np.ndarray], List[float], Optional[Tuple[float, float]]]] = {}
    img_paths: Dict[int, str] = {}
    gt_components_by_page: Dict[int, List[np.ndarray]] = {}
    for page in HOLDOUT_PAGES:
        img_paths[page] = page_img_path(page)
        gt_components_by_page[page] = gt_knot_components(page_ann_path(page))
        yolo_results[page] = yolo_sam_predict(
            yolo_model,
            sam_predictor,
            img_paths[page],
            args.conf,
            yolo_nms_iou=args.yolo_nms_iou,
            mask_nms_iou=args.mask_nms_iou,
        )

    evaluate(yolo_results, prop_results, train_anchors, args.iou_thr, out_dir)
    yolo_pr_sweep(args.yolo_weights, img_paths, gt_components_by_page, args.iou_thr, out_dir)


if __name__ == "__main__":
    main()
