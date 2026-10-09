import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from src.dataset.kwp_mask import KwpMaskBuilder

CROP = 200
COLORS = {"tp": (40, 200, 70), "fp": (230, 40, 40), "fn": (40, 120, 255)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Knot-type gallery: ConvNeXt-S U-Net vs YOLO vs human GT.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--unet_root", type=Path, required=True)
    parser.add_argument("--yolo_root", type=Path, required=True)
    parser.add_argument("--logs", nargs="+", default=["1", "4", "10"])
    parser.add_argument("--per_category", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    builder = KwpMaskBuilder()
    knot = KwpMaskBuilder.class_names().index("knot")
    candidates: Dict[str, List[Tuple[float, dict]]] = {"thick": [], "thin": [], "straight": [], "worst": []}
    for log in args.logs:
        for ann in sorted((args.data_root / log / "ann").glob("*.json")):
            page = ann.name.split(".")[0]
            gt = builder.build(json.load(open(ann))).numpy() == knot
            unet = np.asarray(Image.open(args.unet_root / log / f"{page}.png")) >= 128
            yolo = np.asarray(Image.open(args.yolo_root / f"yolo_probs_log{log}" / f"{page}_mask.png")) > 127
            item = {"log": log, "page": page, "gt": gt, "unet": unet, "yolo": yolo}
            classify(item, candidates)
    chosen = select(candidates, args.logs)
    figure = render(chosen, args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(args.out) + ".html", include_plotlyjs="cdn")
    figure.write_image(str(args.out) + ".png", width=1000, height=figure.layout.height, scale=1.5)
    print("wrote %s (.html, .png), %d rows" % (args.out, len(chosen)))


def classify(item: dict, candidates: Dict[str, List[Tuple[float, dict]]]) -> None:
    gt, unet = item["gt"], item["unet"]
    if gt.sum() == 0:
        return
    dist = ndi.distance_transform_edt(gt)
    for region in regionprops(label(gt, connectivity=2)):
        if region.area < 30:
            continue
        mask = np.zeros_like(gt)
        mask[tuple(region.coords.T)] = True
        width = 2 * dist[mask].max()
        elongation = region.major_axis_length / max(region.minor_axis_length, 1.0)
        entry = dict(item, center=tuple(int(v) for v in region.centroid), width=width, elongation=elongation)
        if width >= 20:
            candidates["thick"].append((region.area, entry))
        elif width < 8 and region.major_axis_length >= 25:
            candidates["thin"].append((region.major_axis_length, entry))
        if elongation >= 3.5 and width >= 8:
            candidates["straight"].append((elongation, entry))
    union = (gt | unet).sum()
    if gt.sum() > 400:
        entry = dict(item, center=error_center(gt, unet), width=0.0, elongation=0.0)
        candidates["worst"].append((-(gt & unet).sum() / union, entry))


def error_center(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, int]:
    errors = ndi.uniform_filter((gt ^ pred).astype(np.float32), CROP // 2)
    return tuple(int(v) for v in np.unravel_index(errors.argmax(), errors.shape))


def select(candidates: Dict[str, List[Tuple[float, dict]]], logs: List[str]) -> List[Tuple[str, dict]]:
    quota = {"thick": 1, "thin": 1, "straight": 1, "worst": 1}
    chosen, used = [], []
    for category, per_log in quota.items():
        for log in logs:
            ranked = sorted((c for c in candidates[category] if c[1]["log"] == log), key=lambda c: -c[0])
            taken = 0
            for _, entry in ranked:
                page = int(entry["page"][5:])
                if any(entry["log"] == other_log and abs(page - other_page) < 15 for other_log, other_page in used):
                    continue
                used.append((entry["log"], page))
                chosen.append((category, entry))
                taken += 1
                if taken == per_log:
                    break
    return chosen


def render(chosen: List[Tuple[str, dict]], data_root: Path) -> go.Figure:
    titles = []
    for category, e in chosen:
        for name in ("image", "ConvNeXt-S U-Net", "YOLO11n-seg"):
            pred = e["unet"] if "U-Net" in name else e["yolo"]
            score = "" if name == "image" else " IoU %.2f" % iou(e["gt"], pred)
            titles.append(
                "log %s %s · %s%s" % (e["log"], e["page"], category, score) if name == "image" else name + score
            )
    figure = make_subplots(
        rows=len(chosen), cols=3, subplot_titles=titles, horizontal_spacing=0.01, vertical_spacing=0.02
    )
    for row, (_, e) in enumerate(chosen, start=1):
        gray = np.asarray(Image.open(next((data_root / e["log"] / "img").glob(e["page"] + ".*"))).convert("L"))
        window = crop_window(e["center"], gray.shape)
        panels = [
            np.repeat(gray[window][..., None], 3, axis=2),
            overlay(gray, e["gt"], e["unet"])[window],
            overlay(gray, e["gt"], e["yolo"])[window],
        ]
        for col, panel in enumerate(panels, start=1):
            figure.add_trace(go.Image(z=panel), row=row, col=col)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(
        height=330 * len(chosen),
        width=1000,
        margin=dict(l=5, r=5, t=60, b=5),
        title="Knots on held-out human logs 1/4/10 (original split). Green = correct, red = false positive, blue = missed.",
    )
    return figure


def crop_window(center: Tuple[int, int], shape: Tuple[int, int]) -> Tuple[slice, slice]:
    half = CROP // 2
    y = min(max(center[0] - half, 0), shape[0] - CROP)
    x = min(max(center[1] - half, 0), shape[1] - CROP)
    return slice(y, y + CROP), slice(x, x + CROP)


def overlay(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    for key, mask in (("tp", gt & pred), ("fp", pred & ~gt), ("fn", gt & ~pred)):
        rgb[mask] = 0.45 * rgb[mask] + 0.55 * np.array(COLORS[key], dtype=np.float32)
    return rgb.astype(np.uint8)


def iou(gt: np.ndarray, pred: np.ndarray) -> float:
    union = (gt | pred).sum()
    return (gt & pred).sum() / union if union else 1.0


if __name__ == "__main__":
    main()
