"""Run all pith detectors on every annotated slice and produce:

  - A CSV with per-slice pixel error per detector
  - A summary CSV with mean/median/p90/max error per detector
  - A montage of per-slice visualizations (GT in green, prediction in red)
    for the BEST and WORST detector by mean pixel error
  - A per-slice grid for the recommended detector for quick eyeballing

Run from ct-log repo root:
    conda run -n ct-log python -m ann_pipeline.scripts.eval_pith
"""

import argparse
import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ann_pipeline.data import DEFAULT_SM_DIR, PithSlice, collect_pith_slices
from ann_pipeline.pith.detectors import DETECTORS
from ann_pipeline.pith.eval import DetectorReport, evaluate


def make_montage(
    report: DetectorReport,
    slices_by_page: Dict[int, PithSlice],
    out_path: str,
    cols: int = 5,
) -> None:
    """Grid of (img, GT, prediction) for every slice in the report."""
    n = len(report.results)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, r in zip(axes, report.results):
        s = slices_by_page[r.page]
        ax.imshow(s.img, cmap="gray")
        ax.plot(r.gt_xy[0], r.gt_xy[1], "o", color="lime", ms=8, mew=2, mfc="none", label="GT")
        ax.plot(r.pred_xy[0], r.pred_xy[1], "x", color="red", ms=8, mew=2, label="pred")
        ax.set_title(f"p{r.page} err={r.pixel_error:.1f}", fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(report.name, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def make_side_by_side(
    reports: List[DetectorReport],
    slices_by_page: Dict[int, PithSlice],
    page: int,
    out_path: str,
) -> None:
    """For a single page, show every detector's prediction side by side."""
    s = slices_by_page[page]
    n = len(reports)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, rep in zip(axes, reports):
        per_slice = {r.page: r for r in rep.results}
        r = per_slice[page]
        ax.imshow(s.img, cmap="gray")
        ax.plot(r.gt_xy[0], r.gt_xy[1], "o", color="lime", ms=10, mew=2, mfc="none")
        ax.plot(r.pred_xy[0], r.pred_xy[1], "x", color="red", ms=10, mew=2)
        ax.set_title(f"{rep.name}\nerr={r.pixel_error:.1f}px", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"page_{page:03d}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sm_dir", default=DEFAULT_SM_DIR)
    parser.add_argument("--out_dir", default="/home/mary/code/ct-log/ann_pipeline/out/pith_eval")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "montage").mkdir(exist_ok=True)
    (out_dir / "side_by_side").mkdir(exist_ok=True)

    print(f"loading slices from {args.sm_dir}")
    slices = collect_pith_slices(args.sm_dir)
    slices_by_page = {s.page: s for s in slices}
    print(f"got {len(slices)} slices with Pith annotation")

    reports: List[DetectorReport] = []
    for name, det in DETECTORS.items():
        print(f"  running {name}")
        rep = evaluate(det, slices, name)
        reports.append(rep)

    # per-slice CSV
    per_slice_rows = []
    for rep in reports:
        for r in rep.results:
            per_slice_rows.append(
                {
                    "page": r.page,
                    "detector": rep.name,
                    "gt_x": r.gt_xy[0],
                    "gt_y": r.gt_xy[1],
                    "pred_x": r.pred_xy[0],
                    "pred_y": r.pred_xy[1],
                    "pixel_error": r.pixel_error,
                    "inference_ms": r.inference_ms,
                }
            )
    df_per_slice = pd.DataFrame(per_slice_rows)
    df_per_slice.to_csv(out_dir / "per_slice.csv", index=False)

    # summary CSV
    summary = pd.DataFrame([rep.summary() for rep in reports]).sort_values("mean_err_px")
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== Summary (sorted by mean error, ascending) ===")
    print(summary.to_string(index=False))

    # per-detector montage
    for rep in reports:
        make_montage(rep, slices_by_page, str(out_dir / "montage" / f"{rep.name}.png"))
    print(f"\nmontages written to {out_dir / 'montage'}")

    # side-by-side per slice
    for page in sorted(slices_by_page):
        make_side_by_side(reports, slices_by_page, page, str(out_dir / "side_by_side" / f"page_{page:03d}.png"))
    print(f"side-by-side written to {out_dir / 'side_by_side'}")


if __name__ == "__main__":
    main()
