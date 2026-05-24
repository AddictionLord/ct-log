"""Quantify how much OBB-only propagation diverges from the anchor-augmented
baseline on subset 4.

Treats `result_obb_aug_ellipse.npz` as pseudo-GT (trusted) and
`result_obb_only.npz` as the prediction. Per frame we compute:

  - Pixel-level IoU and Dice on the binary union of knot CCs.
  - Instance-level Hungarian matching by mask IoU at threshold >= --iou_thr.
    Reports precision/recall/F1, where the anchor-aug CCs are "GT".

Aggregates by anchor distance regime:
  - dist == 0  (anchor frames)
  - dist in [1, 5]
  - dist >  5

Also writes the 8 worst-disagreement frames (by pixel IoU) as side-by-side
panels.

Run from repo root:
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.compare_anchor_vs_obb_only
"""

import argparse
import pathlib
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from experiments.sm2025_subset4_propagate.run import CLASS_IDS, page_img_path

ANCHORS = [
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

NPZ_AUG = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/result_obb_aug_ellipse.npz"
NPZ_OBB_ONLY = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/result_obb_only.npz"
OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/compare_anchor_vs_obb_only"


def dist_to_anchor(page: int) -> int:
    return min(abs(page - a) for a in ANCHORS)


def regime(d: int) -> str:
    if d == 0:
        return "anchor"
    if d <= 5:
        return "near"
    return "far"


def split_cc(mask: np.ndarray, min_px: int = 50) -> List[np.ndarray]:
    lab, n = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    out: List[np.ndarray] = []
    for k in range(1, n + 1):
        comp = lab == k
        if comp.sum() >= min_px:
            out.append(comp)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2.0 * inter / denom) if denom > 0 else (1.0 if inter == 0 else 0.0)


def match_ccs(preds: List[np.ndarray], gts: List[np.ndarray], iou_thr: float) -> Tuple[int, int, int]:
    if not preds and not gts:
        return 0, 0, 0
    if not preds:
        return 0, 0, len(gts)
    if not gts:
        return 0, len(preds), 0
    mat = np.zeros((len(preds), len(gts)), dtype=np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            mat[i, j] = iou(p, g)
    ri, ci = linear_sum_assignment(1.0 - mat)
    tp = 0
    matched_p, matched_g = set(), set()
    for r, c in zip(ri, ci):
        if mat[r, c] >= iou_thr:
            tp += 1
            matched_p.add(int(r))
            matched_g.add(int(c))
    fp = len(preds) - len(matched_p)
    fn = len(gts) - len(matched_g)
    return tp, fp, fn


def render_worst(
    rows: pd.DataFrame, aug_pred: np.ndarray, obb_pred: np.ndarray, pages_arr: np.ndarray, n: int, out_dir: pathlib.Path
) -> None:
    worst = rows.sort_values("pixel_iou").head(n)
    for _, row in worst.iterrows():
        page = int(row["page"])
        idx = int(np.where(pages_arr == page)[0][0])
        img_path = page_img_path(page)
        gray = np.array(Image.open(img_path))
        if gray.ndim == 3:
            gray = gray[..., :3].mean(axis=-1)
        gray = gray.astype(np.uint8)
        aug_m = aug_pred[idx] == CLASS_IDS["Knot"]
        obb_m = obb_pred[idx] == CLASS_IDS["Knot"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        for ax, mask, title in [
            (axes[0], aug_m, "anchor-aug (pseudo-GT)"),
            (axes[1], obb_m, "OBB-only (pred)"),
            (axes[2], aug_m ^ obb_m, "diff (XOR)"),
        ]:
            ax.imshow(gray, cmap="gray")
            overlay = np.zeros((*gray.shape, 4), dtype=np.float32)
            overlay[mask] = (1.0, 0.55, 0.0, 0.45)
            ax.imshow(overlay)
            ax.set_title("%s (%d px)" % (title, int(mask.sum())), fontsize=10)
            ax.axis("off")
        fig.suptitle(
            "page %d  d=%d  IoU=%.3f  Dice=%.3f  TP/FP/FN=%d/%d/%d"
            % (
                page,
                int(row["dist"]),
                row["pixel_iou"],
                row["pixel_dice"],
                int(row["tp"]),
                int(row["fp"]),
                int(row["fn"]),
            ),
            fontsize=12,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = out_dir / ("page_%03d.png" % page)
        plt.savefig(out_path, dpi=120)
        plt.close()
        print("  wrote %s" % out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iou_thr", type=float, default=0.30)
    parser.add_argument("--min_px", type=int, default=50)
    parser.add_argument("--n_worst", type=int, default=8)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aug = np.load(NPZ_AUG, allow_pickle=False)
    obb = np.load(NPZ_OBB_ONLY, allow_pickle=False)
    pages = aug["pages"]
    assert list(pages) == list(obb["pages"]), "page lists differ"
    aug_pred = aug["pred"]
    obb_pred = obb["pred"]

    rows = []
    for i, p in enumerate(pages):
        d = dist_to_anchor(int(p))
        aug_mask = aug_pred[i] == CLASS_IDS["Knot"]
        obb_mask = obb_pred[i] == CLASS_IDS["Knot"]

        aug_ccs = split_cc(aug_mask, args.min_px)
        obb_ccs = split_cc(obb_mask, args.min_px)
        tp, fp, fn = match_ccs(obb_ccs, aug_ccs, args.iou_thr)

        rows.append(
            {
                "page": int(p),
                "dist": d,
                "regime": regime(d),
                "n_aug_ccs": len(aug_ccs),
                "n_obb_ccs": len(obb_ccs),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "pixel_iou": iou(aug_mask, obb_mask),
                "pixel_dice": dice(aug_mask, obb_mask),
                "px_aug": int(aug_mask.sum()),
                "px_obb": int(obb_mask.sum()),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_frame.csv", index=False)

    print("\n=== aggregate (all 292 frames) ===")
    print("mean pixel IoU:  %.3f" % df["pixel_iou"].mean())
    print("mean pixel Dice: %.3f" % df["pixel_dice"].mean())
    tp_total, fp_total, fn_total = df["tp"].sum(), df["fp"].sum(), df["fn"].sum()
    prec = tp_total / max(1, tp_total + fp_total)
    rec = tp_total / max(1, tp_total + fn_total)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    print("instance precision / recall / F1 (vs anchor-aug as GT): %.3f / %.3f / %.3f" % (prec, rec, f1))

    print("\n=== by regime ===")
    summary_rows = []
    for reg in ("anchor", "near", "far"):
        sub = df[df.regime == reg]
        if not len(sub):
            continue
        tp_, fp_, fn_ = sub["tp"].sum(), sub["fp"].sum(), sub["fn"].sum()
        p_ = tp_ / max(1, tp_ + fp_)
        r_ = tp_ / max(1, tp_ + fn_)
        f_ = 2 * p_ * r_ / max(1e-9, p_ + r_)
        summary_rows.append(
            {
                "regime": reg,
                "n_frames": len(sub),
                "mean_iou": float(sub["pixel_iou"].mean()),
                "mean_dice": float(sub["pixel_dice"].mean()),
                "precision": p_,
                "recall": r_,
                "f1": f_,
                "tp": int(tp_),
                "fp": int(fp_),
                "fn": int(fn_),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "by_regime.csv", index=False)
    print(summary_df.to_string(index=False))

    print("\n=== writing %d worst-disagreement panels ===" % args.n_worst)
    render_worst(df, aug_pred, obb_pred, pages, args.n_worst, out_dir)


if __name__ == "__main__":
    main()
