"""Render every frame with its propagated timber id drawn on each board."""

import argparse
from os.path import join
import pathlib

import cv2
import numpy as np
from tqdm import tqdm

from ann_pipeline.timber.data import DEFAULT_PROJECT_ROOT, DEFAULT_SUBSET, load_gray
from ann_pipeline.timber.track import propagate

PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--out_dir", default="/home/mary/code/ct-log/ann_pipeline/out/timber_tracks_vis")
    args = parser.parse_args()

    subset_dir = join(args.project_root, args.subset)
    labelled, df = propagate(subset_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ids = sorted({tid for ids in labelled.values() for tid in ids})
    color_of = {tid: PALETTE[i % len(PALETTE)] for i, tid in enumerate(all_ids)}
    anchors = set(df[df.is_anchor].frame)

    for fname in tqdm(sorted(labelled), desc="rendering"):
        gray = load_gray(join(subset_dir, "img", fname))
        vis = np.stack([gray] * 3, axis=-1).astype(np.uint8)
        for tid, mask in labelled[fname].items():
            color = color_of[tid]
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2)
            ys, xs = np.nonzero(mask)
            cv2.putText(
                vis, tid, (int(xs.mean()) - 18, int(ys.mean()) + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
        stem = fname.rsplit(".", 1)[0]
        tag = "anchor" if fname in anchors else "prop"
        flag = "" if len(labelled[fname]) == 8 else "FLAG_"
        cv2.putText(vis, f"{stem} {tag} ids={len(labelled[fname])}", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"{flag}{stem}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"wrote {len(labelled)} PNGs to {out_dir}")


if __name__ == "__main__":
    main()
