"""Quantify propagation error as a function of distance-to-nearest-anchor.

Method: leave-one-out across all 16 annotated pages in subset 4. For each
annotated page p we re-run propagation with p removed from the anchor set,
then measure on page p:
  * pith centroid distance vs GT pith point
  * knot recall + mean centroid distance vs GT knots (Hungarian matching)
  * wood Dice vs GT wood mask

Each held-out page also has a "distance to nearest remaining anchor" measured
in *page index space* (frames are spaced 1 apart by page number).

Run from repo root:
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.sweep_distance
"""

import argparse
import os
from os.path import join
import pathlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sam2.build_sam import build_sam2_video_predictor_npz

from experiments.sm2025_subset4_propagate.compare_methods import (
    hungarian_match,
    knot_components_centroids,
    render_gt_knot_centroids,
)
from experiments.sm2025_subset4_propagate.run import (
    CLASS_IDS,
    find_anchor_pages,
    list_pages,
    load_tiff_gray,
    page_ann_path,
    page_img_path,
    propagate_segment,
    render_annotation,
)

CHECKPOINT = "/mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt"
OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out/sweep"


def propagate_all(
    predictor,
    vol: np.ndarray,
    anchors: List[int],
    anchor_masks: Dict[int, Dict[str, np.ndarray]],
    anchor_pith: Dict[int, List[Tuple[int, int]]],
    pages: List[int],
    size: int,
) -> np.ndarray:
    """Run the full anchor-segmented propagation over the whole volume."""
    page_to_idx = {p: i for i, p in enumerate(pages)}
    anchor_idx = [page_to_idx[p] for p in anchors]
    h, w = vol.shape[1], vol.shape[2]
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

    return pred


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    pages = list_pages()
    all_anchors = find_anchor_pages(pages)
    page_to_idx = {p: i for i, p in enumerate(pages)}
    sample = load_tiff_gray(page_img_path(pages[0]))
    h, w = sample.shape
    vol = np.stack([load_tiff_gray(page_img_path(p)) for p in pages])
    print("loaded %d frames, %d annotated anchors" % (len(pages), len(all_anchors)))

    full_masks: Dict[int, Dict[str, np.ndarray]] = {}
    full_pith: Dict[int, List[Tuple[int, int]]] = {}
    for p in all_anchors:
        m, pith = render_annotation(page_ann_path(p), h, w)
        full_masks[p] = m
        full_pith[p] = pith

    repo_root = os.path.abspath(join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty", "MedSAM2"))
    model_cfg = "//" + join(repo_root, "sam2/configs/sam2.1_hiera_t512.yaml")
    predictor = build_sam2_video_predictor_npz(model_cfg, CHECKPOINT)

    rows = []
    for held in all_anchors:
        kept = [a for a in all_anchors if a != held]
        kept_masks = {a: full_masks[a] for a in kept}
        kept_pith = {a: full_pith[a] for a in kept}
        dist_to_nearest = min(abs(held - a) for a in kept)
        print("hold-out page=%d nearest_anchor_dist=%d frames" % (held, dist_to_nearest))

        pred = propagate_all(predictor, vol, kept, kept_masks, kept_pith, pages, size=args.size)
        pf = pred[page_to_idx[held]]

        gt_masks = full_masks[held]
        gt_pith_pts = full_pith[held]
        gt_knots = render_gt_knot_centroids(page_ann_path(held), h, w)

        prop_knots = knot_components_centroids(pf == CLASS_IDS["Knot"])
        ys, xs = np.nonzero(pf == CLASS_IDS["Pith"])
        prop_pith = (float(xs.mean()), float(ys.mean())) if len(xs) > 0 else None
        gt_pith = gt_pith_pts[0] if gt_pith_pts else None

        pith_dist = (
            float(np.hypot(prop_pith[0] - gt_pith[0], prop_pith[1] - gt_pith[1]))
            if (prop_pith is not None and gt_pith is not None)
            else float("nan")
        )
        matched, fp, fn = hungarian_match(prop_knots, gt_knots)
        knot_recall = len(matched) / len(gt_knots) if gt_knots else float("nan")
        knot_mean_dist = float(np.mean([d for *_, d in matched])) if matched else float("nan")

        wood_dice = dice(pf == CLASS_IDS["Wood"], gt_masks["Wood"])

        rows.append(
            {
                "page": held,
                "nearest_anchor_dist": dist_to_nearest,
                "n_gt_knots": len(gt_knots),
                "pith_dist_px": pith_dist,
                "knot_recall": knot_recall,
                "knot_mean_dist_px": knot_mean_dist,
                "knot_fp": len(fp),
                "knot_fn": len(fn),
                "wood_dice": wood_dice,
            }
        )

    df = pd.DataFrame(rows).sort_values("nearest_anchor_dist").reset_index(drop=True)
    df.to_csv(join(OUT_DIR, "loo_sweep.csv"), index=False)
    print("\n", df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(df["nearest_anchor_dist"], df["pith_dist_px"], c="tab:blue")
    for _, r in df.iterrows():
        axes[0].annotate(
            str(r["page"]),
            (r["nearest_anchor_dist"], r["pith_dist_px"]),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )
    axes[0].set_xlabel("dist to nearest anchor (frames)")
    axes[0].set_ylabel("pith centroid err (px)")
    axes[0].set_title("Pith: propagation error vs anchor distance")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(df["nearest_anchor_dist"], df["knot_recall"], c="tab:orange", label="recall")
    axes[1].set_xlabel("dist to nearest anchor (frames)")
    axes[1].set_ylabel("knot recall")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Knot recall vs anchor distance")
    axes[1].grid(alpha=0.3)
    for _, r in df.iterrows():
        axes[1].annotate(
            str(r["page"]),
            (r["nearest_anchor_dist"], r["knot_recall"]),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )

    axes[2].scatter(df["nearest_anchor_dist"], df["wood_dice"], c="tab:green")
    axes[2].set_xlabel("dist to nearest anchor (frames)")
    axes[2].set_ylabel("wood Dice")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("Wood Dice vs anchor distance")
    axes[2].grid(alpha=0.3)
    for _, r in df.iterrows():
        axes[2].annotate(
            str(r["page"]),
            (r["nearest_anchor_dist"], r["wood_dice"]),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(join(OUT_DIR, "loo_sweep.png"), dpi=120)
    plt.close()
    print("wrote %s" % join(OUT_DIR, "loo_sweep.png"))


if __name__ == "__main__":
    main()
