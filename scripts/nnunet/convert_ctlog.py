import argparse
import json
from pathlib import Path
import re
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import SimpleITK as sitk
from src.dataset.ct_log_kwp_dataset import HUMAN_REVIEW_TAG
from src.dataset.kwp_mask import KwpMaskBuilder

PAGE_RE = re.compile(r"page_(\d+)")
IGNORE = 3
SPACING_XYZ = (1.0, 1.0, 10.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert kwp-ds logs to an nnU-Net v2 raw dataset (one 3D volume per log)."
    )
    parser.add_argument("--data_root", type=Path, required=True, help="kwp_ds_v4 root with <log>/img and <log>/ann.")
    parser.add_argument("--train_logs", nargs="+", required=True)
    parser.add_argument(
        "--human_only_logs",
        nargs="*",
        default=["08"],
        help="Logs where only bumaska slices are labelled; the rest is ignore.",
    )
    parser.add_argument("--predict_logs", nargs="+", default=["10", "1", "2", "4"])
    parser.add_argument(
        "--predict_root", type=Path, required=True, help="Data root for the prediction volumes (e.g. kwp_ds_v3)."
    )
    parser.add_argument("--out", type=Path, required=True, help="nnUNet_raw/DatasetXXX_Name")
    parser.add_argument("--predict_out", type=Path, required=True)
    args = parser.parse_args()

    for sub in ("imagesTr", "labelsTr"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    args.predict_out.mkdir(parents=True, exist_ok=True)
    builder = KwpMaskBuilder()
    summary: Dict[str, dict] = {}
    for log in args.train_logs:
        paths = pages(args.data_root / log)
        volume = np.stack([gray(p) for p in paths])
        labels = np.stack(
            [
                label_slice(builder, args.data_root / log / "ann" / (p.name + ".json"), log in args.human_only_logs)
                for p in paths
            ]
        )
        write(volume, args.out / "imagesTr" / ("ctlog_%s_0000.nii.gz" % log))
        write(labels, args.out / "labelsTr" / ("ctlog_%s.nii.gz" % log))
        summary[log] = {"slices": len(paths), "ignored_slices": int((labels == IGNORE).all(axis=(1, 2)).sum())}
        print(log, summary[log], flush=True)
    for log in args.predict_logs:
        paths = pages(args.predict_root / log)
        write(np.stack([gray(p) for p in paths]), args.predict_out / ("ctlog_%s_0000.nii.gz" % log))
        (args.predict_out / ("ctlog_%s_pages.json" % log)).write_text(json.dumps([p.name for p in paths]))
    dataset = {
        "channel_names": {"0": "zscore_uint8_ct"},
        "labels": {"background": 0, "wood": 1, "knot": 2, "ignore": IGNORE},
        "numTraining": len(args.train_logs),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
        "notes": "kwp-ds-v4 teacher split; one volume per log, spacing (1, 1, 10) mm; wood = wood not covered by knot; ignore = slices without a label (and non-human slices of %s)"
        % ",".join(args.human_only_logs),
    }
    (args.out / "dataset.json").write_text(json.dumps(dataset, indent=2))
    (args.out / "conversion_summary.json").write_text(json.dumps(summary, indent=1))


def pages(log_dir: Path) -> List[Path]:
    found = sorted(
        (int(PAGE_RE.search(p.stem).group(1)), p) for p in (log_dir / "img").glob("*.tiff") if PAGE_RE.search(p.stem)
    )
    return [path for _, path in found]


def gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def label_slice(builder: KwpMaskBuilder, ann: Path, human_only: bool) -> np.ndarray:
    annotation: Optional[dict] = json.loads(ann.read_text()) if ann.exists() else None
    if annotation is None:
        return np.full((778, 778), IGNORE, np.uint8)
    if human_only and not any(t.get("name") == HUMAN_REVIEW_TAG for t in annotation.get("tags", [])):
        return np.full((annotation["size"]["height"], annotation["size"]["width"]), IGNORE, np.uint8)
    return builder.build(annotation).numpy().astype(np.uint8)


def write(array: np.ndarray, path: Path) -> None:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(SPACING_XYZ)
    sitk.WriteImage(image, str(path), useCompression=True)


if __name__ == "__main__":
    main()
