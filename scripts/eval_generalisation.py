import argparse
import json
from pathlib import Path
import pickle
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scripts.export_knot_probs import load_stack
from scripts.export_report_preds import load_member
from src.dataset.ct_log_kwp_dataset import HUMAN_REVIEW_TAG
from src.dataset.kwp_mask import KwpMaskBuilder
from src.eval.knot_instances import KnotInstanceEvaluator
from src.train_kwp import extract_features
import torch
import torch.nn.functional as F

PAGE_RE = re.compile(r"page_(\d+)")
NATIVE = 778


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Knot/pith metrics of one model or an ensemble on train-human, train-auto, val and test frames."
    )
    parser.add_argument("--member", action="append", required=True, help="name=config.yaml=checkpoint[=pith_source]")
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        help="name:filter:n:log_dir,log_dir,...  filter = human|auto|all, n = 0 for all",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dump_dir", type=Path, default=None, help="Pickle per-slice instance data (<split>.pkl) for bootstraps."
    )
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    names = [spec.split("=", 1)[0] for spec in args.member]
    members = [load_member(spec.split("=", 1)[1], device) for spec in args.member]
    results: Dict[str, dict] = {}
    for split in args.split:
        name, kind, n, logs = split.split(":", 3)
        frames = select_frames([Path(p) for p in logs.split(",")], kind, int(n), args.seed)
        start = time.monotonic()
        dump = args.dump_dir / (name + ".pkl") if args.dump_dir else None
        results[name] = evaluate_frames(frames, names, members, device, args.batch_size, dump)
        results[name]["frame_filter"] = {
            "kind": kind,
            "n_requested": int(n),
            "logs": [Path(p).name for p in logs.split(",")],
            "seed": args.seed,
        }
        results[name]["seconds"] = round(time.monotonic() - start, 1)
        print(
            name,
            {k: v.get("knot_iou") for k, v in results[name].items() if isinstance(v, dict) and "knot_iou" in v},
            flush=True,
        )
    args.out.write_text(json.dumps(results, indent=1))


def select_frames(log_dirs: List[Path], kind: str, n: int, seed: int) -> List[Tuple[Path, Path]]:
    frames = []
    for log_dir in log_dirs:
        for ann in sorted((log_dir / "ann").glob("*.json")):
            human = any(tag.get("name") == HUMAN_REVIEW_TAG for tag in json.loads(ann.read_text()).get("tags", []))
            if kind == "all" or (kind == "human") == human:
                frames.append((log_dir, ann))
    if n and n < len(frames):
        frames = sorted(random.Random(seed).sample(frames, n))
    return frames


def evaluate_frames(
    frames: List[Tuple[Path, Path]],
    names: List[str],
    members: List[dict],
    device: torch.device,
    batch_size: int,
    dump: Optional[Path] = None,
) -> dict:
    keys = [*names, "ensemble"] if len(names) > 1 else names
    evaluators = {k: KnotInstanceEvaluator() for k in keys}
    pith_errors: Dict[str, List[float]] = {k: [] for k in keys}
    per_slice: Dict[str, List[Tuple[str, int, int]]] = {k: [] for k in keys}
    builder = KwpMaskBuilder()
    knot_id, background = KwpMaskBuilder.class_names().index("knot"), KwpMaskBuilder.class_names().index("background")
    by_log: Dict[Path, List[Path]] = {}
    for log_dir, ann in frames:
        by_log.setdefault(log_dir, []).append(ann)
    for log_dir, anns in by_log.items():
        pages = sorted(
            (int(PAGE_RE.search(p.stem).group(1)), p)
            for p in (log_dir / "img").glob("*.tiff")
            if PAGE_RE.search(p.stem)
        )
        paths = [path for _, path in pages]
        position = {path.name: i for i, path in enumerate(paths)}
        for start in range(0, len(anns), batch_size):
            chunk = anns[start : start + batch_size]
            stacks = torch.stack([load_stack(paths, position[ann.name[: -len(".json")]], 1) for ann in chunk])
            outputs = [predict_member(member, stacks, device) for member in members]
            if len(members) > 1:
                piths = [o[1] for o in outputs if o[1] is not None]
                outputs.append((np.mean([o[0] for o in outputs], axis=0), np.mean(piths, axis=0) if piths else None))
            for i, ann in enumerate(chunk):
                annotation = json.loads(ann.read_text())
                gt = builder.build(annotation).numpy() == knot_id
                click = gt_click(annotation)
                for key, (probs, pith) in zip(keys, outputs):
                    wood = probs[i].argmax(axis=0) != background
                    evaluators[key].add(gt, probs[i, knot_id] >= 0.5, wood=wood, score_map=probs[i, knot_id])
                    knot = (probs[i, knot_id] >= 0.5) & wood
                    per_slice[key].append(
                        (
                            "%s/%s" % (log_dir.name, ann.name[: -len(".json")]),
                            int((knot & gt).sum()),
                            int((knot | gt).sum()),
                        )
                    )
                    if pith is not None and click is not None:
                        x, y = np.clip(np.rint(pith[i]), 0, NATIVE - 1)
                        pith_errors[key].append(float(np.hypot(x - click[0], y - click[1])))
    if dump is not None:
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.write_bytes(pickle.dumps({k: {"per_slice": per_slice[k], "slices": evaluators[k].slices} for k in keys}))
    out = {}
    for key in keys:
        r = evaluators[key].result()
        out[key] = {
            "n_frames": r["n_frames"],
            "knot_iou": r["pixel_iou"],
            "inst_f1_30": r["iou_0.3"]["f1"],
            "mean_per_gt_iou": r["iou_0.3"]["mean_per_gt_iou"],
            "map50": r["map50"],
            "map50_95": r["map50_95"],
            "thin_recall": r["recall_thin_iou0.3"]["recall"],
            "n_thin": r["recall_thin_iou0.3"]["n"],
            "n_gt_knots": r["n_gt"],
            "thin_map50": r["ap_by_stratum"]["thin"]["map50"],
            "thin_map50_95": r["ap_by_stratum"]["thin"]["map50_95"],
            "thick_map50": r["ap_by_stratum"]["thick"]["map50"],
            "thick_map50_95": r["ap_by_stratum"]["thick"]["map50_95"],
            "n_thick": r["ap_by_stratum"]["thick"]["n_gt"],
            "pith_mean_px_rounded": round(float(np.mean(pith_errors[key])), 3) if pith_errors[key] else None,
            "n_pith": len(pith_errors[key]),
            "per_slice": per_slice[key],
        }
    return out


def predict_member(member: dict, stacks: torch.Tensor, device: torch.device) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    config = member["config"]
    images = F.interpolate(stacks, size=tuple(config.resolution), mode="bilinear", align_corners=False)
    with torch.no_grad():
        features = extract_features(
            member["backbone"], member["transform"](images.to(device)), config.n_layers, config.backbone_bf16
        )
        probs = member["seg_head"](features).float().softmax(dim=1)
        pith = member["pith_head"](features).float().cpu().numpy() * NATIVE if member["pith_head"] is not None else None
        probs = F.interpolate(probs, size=(NATIVE, NATIVE), mode="bilinear", align_corners=False)
    return probs[:, :3].cpu().numpy() if probs.shape[1] > 3 else probs.cpu().numpy(), pith


def gt_click(annotation: dict) -> Optional[Tuple[float, float]]:
    for obj in annotation.get("objects", []):
        if obj.get("classTitle", "").lower() == "pith":
            x, y = obj["points"]["exterior"][0]
            return float(x), float(y)
    return None


if __name__ == "__main__":
    main()
