"""Train a YOLO11n knot detector on the Supervisely-derived dataset.

Assumes data_prep.py has already been run to populate the YOLO directory.
Aggressive rotation/flip augmentation is enabled because knots are radially
symmetric around the pith — rotating a slice yields a valid new sample.

Run from ct-log repo root:
    conda run -n ct-log python -m ann_pipeline.knot.train
"""

import argparse
import os
import pathlib

os.environ.setdefault("MLFLOW_TRACKING_URI", "")
os.environ.setdefault("YOLO_MLFLOW", "False")

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

SETTINGS["mlflow"] = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/mary/code/ct-log/ann_pipeline/out/knot_yolo/knots.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="/home/mary/code/ct-log/ann_pipeline/out/knot_runs")
    parser.add_argument("--name", default="yolo11n_knots")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    pathlib.Path(args.project).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        mosaic=0.5,
        translate=0.05,
        scale=0.2,
        patience=50,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
