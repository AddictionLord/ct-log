import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from tqdm import tqdm
from ultralytics import YOLO

from ann_pipeline.knot.data_prep import decode_bitmap
from ann_pipeline.scripts.upload_ct_logs_to_phases import _bitmap_object, _largest_cc_fill, _point_object, _yolo_pith
from ann_pipeline.wood.detectors import threshold_peel

RAW_ROOTS = [pathlib.Path("/mnt/D/datasets/ct_log/generated"), pathlib.Path("/mnt/D/datasets/ct_log/375492_SM_2025")]
SNAP = pathlib.Path("/mnt/D/datasets/ct_log/human_reviewed_20260929")
DEFAULT_SEG_WEIGHTS = "ann_pipeline/out/knot_runs/yolo11n_seg_v5_val10_800/weights/best.pt"
DEFAULT_SEG_IMGSZ = 800
DEFAULT_OBB_WEIGHTS = "ann_pipeline/out/knot_runs/yolo11n_obb_2cls_v5_n640/weights/best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--seg_weights", default=DEFAULT_SEG_WEIGHTS)
    parser.add_argument("--seg_imgsz", type=int, default=DEFAULT_SEG_IMGSZ)
    parser.add_argument("--obb_weights", default=DEFAULT_OBB_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--auto_logs", nargs="+", required=True)
    parser.add_argument("--human_logs", nargs="+", required=True)
    parser.add_argument("--human_root", default=str(SNAP))
    parser.add_argument("--pith_root", default="")
    parser.add_argument("--knot_root", default="")
    parser.add_argument("--knot_threshold", type=int, default=128)
    parser.add_argument("--knot_source", default="")
    args = parser.parse_args()

    out_root = pathlib.Path(args.out_root)
    human_root = pathlib.Path(args.human_root)
    seg, obb = YOLO(args.seg_weights), YOLO(args.obb_weights)
    stats: Dict[str, dict] = {}
    for log in args.auto_logs:
        piths = load_piths(pathlib.Path(args.pith_root), log) if args.pith_root else None
        knots = KnotSource(pathlib.Path(args.knot_root) / log, args.knot_threshold) if args.knot_root else None
        stats[log] = build_auto_log(log, out_root, human_root, seg, obb, args.conf, args.seg_imgsz, piths, knots)
        if knots is not None:
            stats[log]["knot_inputs_sha256"] = knots.digest()
    for log in args.human_logs:
        stats[log] = build_human_log(log, out_root, human_root)
    write_record(args, out_root, stats)


def raw_img_dir(log: str) -> pathlib.Path:
    for root in RAW_ROOTS:
        d = root / log / "img"
        if d.is_dir() and any(d.glob("page_*.tiff")):
            return d
    msg = "no raw image dir for log %s" % log
    raise ValueError(msg)


def link_images(log: str, out_dir: pathlib.Path) -> List[pathlib.Path]:
    src = raw_img_dir(log)
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    pages = sorted(src.glob("page_*.tiff"))
    for p in pages:
        target = img_dir / p.name
        if not target.exists():
            target.symlink_to(p.resolve())
    return pages


def build_auto_log(
    log: str,
    out_root: pathlib.Path,
    human_root: pathlib.Path,
    seg: YOLO,
    obb: YOLO,
    conf: float,
    imgsz: int,
    piths: Optional[Dict[str, Optional[List[float]]]],
    knots: Optional["KnotSource"] = None,
) -> dict:
    out_dir = out_root / log
    pages = link_images(log, out_dir)
    ann_dir = out_dir / "ann"
    ann_dir.mkdir(parents=True, exist_ok=True)
    human_ann = human_root / log / "ann"
    n_human = n_knot = knot_px = 0
    for p in tqdm(pages, desc="auto %s" % log):
        hp = human_ann / (p.name + ".json")
        if hp.exists():
            ann = json.loads(hp.read_text())
            n_human += 1
        else:
            ann = auto_annotation(p, seg, obb, conf, imgsz, piths, knots)
        objs = [o for o in ann["objects"] if o.get("classTitle") == "Knot"]
        n_knot += bool(objs)
        knot_px += sum(knot_pixels(o) for o in objs)
        (ann_dir / (p.name + ".json")).write_text(json.dumps(ann))
    return {
        "kind": "auto",
        "frames": len(pages),
        "human_frames": n_human,
        "frames_with_knot": n_knot,
        "knot_px": knot_px,
        "img_source": str(raw_img_dir(log)),
    }


def auto_annotation(
    page: pathlib.Path,
    seg: YOLO,
    obb: YOLO,
    conf: float,
    imgsz: int,
    piths: Optional[Dict[str, Optional[List[float]]]] = None,
    knots: Optional["KnotSource"] = None,
) -> dict:
    arr = np.array(Image.open(page))
    gray = (arr[..., :3].mean(-1) if arr.ndim == 3 else arr).astype(np.uint8)
    rgb = np.stack([gray] * 3, -1)
    h, w = gray.shape
    if knots is not None:
        knot = knots.mask(page.name, (h, w))
    else:
        r = seg.predict(rgb, conf=conf, imgsz=imgsz, retina_masks=True, verbose=False)[0]
        has = r.masks is not None and len(r.masks)
        knot = (r.masks.data.cpu().numpy() > 0.5).any(0) if has else np.zeros((h, w), bool)
    wood = _largest_cc_fill(threshold_peel(gray, 30, 60, 5).astype(bool) | knot).astype(bool)
    knot &= wood
    objects: List[dict] = []
    lab, n = ndi.label(knot, structure=np.ones((3, 3), np.uint8))
    for k in range(1, n + 1):
        obj = _bitmap_object("Knot", (lab == k).astype(np.uint8))
        if obj is not None:
            objects.append(obj)
    wood_obj = _bitmap_object("Wood", wood.astype(np.uint8))
    if wood_obj is not None:
        objects.append(wood_obj)
    pith: Optional[tuple] = _yolo_pith(obb, rgb) if piths is None else piths.get(page.name)
    if pith is not None:
        objects.append(_point_object("Pith", *pith))
    return {"size": {"height": h, "width": w}, "description": "", "tags": [], "objects": objects}


def load_piths(root: pathlib.Path, log: str) -> Dict[str, Optional[List[float]]]:
    path = root / ("%s.json" % log)
    if not path.exists():
        msg = "no external pith file %s" % path
        raise ValueError(msg)
    return json.loads(path.read_text())["pith"]


def knot_pixels(obj: dict) -> int:
    return int(decode_bitmap(obj["bitmap"]["data"]).sum())


def build_human_log(log: str, out_root: pathlib.Path, human_root: pathlib.Path) -> dict:
    out_dir = out_root / log
    pages = link_images(log, out_dir)
    ann_dir = out_dir / "ann"
    ann_dir.mkdir(parents=True, exist_ok=True)
    n = n_knot = knot_px = 0
    for hp in sorted((human_root / log / "ann").glob("*.json")):
        ann = json.loads(hp.read_text())
        objs = [o for o in ann["objects"] if o.get("classTitle") == "Knot"]
        n += 1
        n_knot += bool(objs)
        knot_px += sum(knot_pixels(o) for o in objs)
        (ann_dir / hp.name).write_text(json.dumps(ann))
    return {
        "kind": "human",
        "frames": n,
        "img_pages": len(pages),
        "frames_with_knot": n_knot,
        "knot_px": knot_px,
        "img_source": str(raw_img_dir(log)),
    }


class KnotSource:
    def __init__(self, log_dir: pathlib.Path, threshold: int) -> None:
        if not log_dir.is_dir():
            msg = "no external knot dir %s" % log_dir
            raise ValueError(msg)
        self.log_dir = log_dir
        self.threshold = threshold
        self.hashes: List[str] = []

    def mask(self, page_name: str, shape: Tuple[int, int]) -> np.ndarray:
        path = self.log_dir / page_name.replace(".tiff", ".png")
        if not path.exists():
            msg = "missing knot probability %s" % path
            raise ValueError(msg)
        raw = path.read_bytes()
        self.hashes.append("%s %s" % (hashlib.sha256(raw).hexdigest(), path.name))
        prob = np.array(Image.open(path))
        if prob.shape != shape:
            msg = "knot probability %s has shape %s, expected %s" % (path, prob.shape, shape)
            raise ValueError(msg)
        return prob >= self.threshold

    def digest(self) -> str:
        return hashlib.sha256("\n".join(sorted(self.hashes)).encode()).hexdigest()


def knot_record(args: argparse.Namespace) -> dict:
    if args.knot_root:
        return {
            "source": "external per-slice knot probabilities from %s" % args.knot_root,
            "description": args.knot_source,
            "threshold": "p >= %d/255" % args.knot_threshold,
            "post": "clipped to wood, one Knot object per 8-connected component, no other filters",
            "per_log_inputs_sha256": "see logs.<log>.knot_inputs_sha256 (sha256 of sorted per-file sha256 lines)",
        }
    return {
        "model": args.seg_weights,
        "sha256": sha256(pathlib.Path(args.seg_weights)),
        "conf": args.conf,
        "imgsz": args.seg_imgsz,
        "post": "union of instance masks, clipped to wood, one Knot object per 8-connected component, no other filters",
    }


def pith_record(args: argparse.Namespace) -> dict:
    if args.pith_root:
        root = pathlib.Path(args.pith_root)
        meta_path = root / "META.json"
        return {
            "source": "external per-slice pith points from %s" % root,
            "meta": json.loads(meta_path.read_text()) if meta_path.exists() else None,
        }
    return {
        "model": args.obb_weights,
        "sha256": sha256(pathlib.Path(args.obb_weights)),
        "conf": 0.10,
        "post": "highest-confidence pith box centre snapped to the darkest pixel in a 5x5 window (upload_ct_logs_to_phases._yolo_pith)",
    }


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_record(args: argparse.Namespace, out_root: pathlib.Path, stats: Dict[str, dict]) -> None:
    manifest = sorted(out_root.glob("*/ann/*.json"))
    lines = ["%s  %s" % (sha256(p), p.relative_to(out_root)) for p in manifest]
    (out_root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    record = {
        "version": args.version,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "generator": {
            "knot": knot_record(args),
            "pith": pith_record(args),
            "wood": "largest_cc_fill(threshold_peel(gray, 30, 60, 5) | knot)",
            "human_root": args.human_root,
        },
        "logs": stats,
        "manifest": "MANIFEST.sha256 (%d annotation files)" % len(manifest),
        "manifest_sha256": sha256(out_root / "MANIFEST.sha256"),
    }
    (out_root / "DATASET.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "logs"}, indent=1))
    for log, s in stats.items():
        print("%-4s %s" % (log, s))


if __name__ == "__main__":
    main()
