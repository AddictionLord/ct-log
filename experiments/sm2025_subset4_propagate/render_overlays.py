"""Re-render overlay PNGs from a saved result.npz (no re-propagation).

Draws Wood + Knot as filled masks and Pith as a single 'x' marker at the
centroid of the predicted blob.

Run from repo root:
    conda run -n ct-log python -m experiments.sm2025_subset4_propagate.render_overlays
"""

import argparse
from os.path import join
import pathlib
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "/home/mary/code/ct-log/experiments/sm2025_subset4_propagate/out"
CLASS_IDS = {"Wood": 1, "Knot": 2, "Pith": 3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=join(OUT_DIR, "result.npz"))
    parser.add_argument("--n_vis", type=int, default=12)
    parser.add_argument("--overlays_dir", default=join(OUT_DIR, "overlays"))
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    vol = data["imgs"]
    pred = data["pred"]
    pages = data["pages"]
    is_anchor = data["is_anchor"]
    anchor_idx = [i for i, a in enumerate(is_anchor) if a]
    h, w = vol.shape[1], vol.shape[2]

    pathlib.Path(args.overlays_dir).mkdir(parents=True, exist_ok=True)

    nonanchor_indices = [i for i in range(len(pages)) if not is_anchor[i]]
    step = max(1, len(nonanchor_indices) // args.n_vis)
    chosen = sorted(set(nonanchor_indices[::step][: args.n_vis]) | set(anchor_idx))
    cmap = plt.get_cmap("tab10")

    for i in chosen:
        p = int(pages[i])
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(vol[i], cmap="gray")
        ax[0].set_title("page_%03d %s" % (p, "(ANCHOR)" if is_anchor[i] else ""))
        ax[0].axis("off")

        ax[1].imshow(vol[i], cmap="gray")
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        pith_xy: Tuple[float, float] = None
        for cls_name, cid in CLASS_IDS.items():
            mask = pred[i] == cid
            if not mask.any():
                continue
            if cls_name == "Pith":
                ys, xs = np.nonzero(mask)
                pith_xy = (float(xs.mean()), float(ys.mean()))
                continue
            colour = np.array(cmap(cid - 1))
            overlay[mask] = (*colour[:3], 0.5)
        ax[1].imshow(overlay)
        if pith_xy is not None:
            colour = cmap(CLASS_IDS["Pith"] - 1)
            ax[1].scatter([pith_xy[0]], [pith_xy[1]], s=60, c=[colour], marker="x", linewidths=2)
        ax[1].set_title("pred (W=blue, K=orange, P=green x)")
        ax[1].axis("off")
        plt.tight_layout()
        plt.savefig(join(args.overlays_dir, "page_%03d.png" % p), dpi=110)
        plt.close()
    print("wrote %d overlays to %s" % (len(chosen), args.overlays_dir))


if __name__ == "__main__":
    main()
