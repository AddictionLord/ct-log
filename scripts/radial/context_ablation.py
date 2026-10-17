import argparse
import json
import logging
import pathlib
from typing import Dict, List

import numpy as np
from scripts.export_ensemble_probs import load_member
from scripts.export_knot_probs import knot_probs, load_stack
from scripts.radial.common import OUT, SRC_LOG, components, match, page_name, read_png, write_png
from src.dataset.kwp_mask import KwpMaskBuilder
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
M5_CONFIG = "src/configs/train_kwp_v4_unet_effv2s_w1_dice_ts_48ep_dec512.yaml"
M5_CKPT = "/mnt/D/models/ct_log/m5/seg_head_epoch_47.pth"
M5_PROBS = pathlib.Path("/mnt/D/datasets/ct_log/knot_probs_m5/10")


def main() -> None:
    parser = argparse.ArgumentParser(description="M5 on log 10 with and without the z+-1 context channels.")
    parser.add_argument("--root", type=pathlib.Path, default=OUT)
    parser.add_argument("--check_pages", type=int, nargs="+", default=[26, 110, 215])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = torch.device("cuda")
    knot_id = KwpMaskBuilder.class_names().index("knot")
    member = load_member("%s=%s" % (M5_CONFIG, M5_CKPT), device)
    paths = sorted((SRC_LOG / "img").glob("page_*.tiff"))
    out_dir = args.root / "predictions" / "m5_nocontext"

    for z in args.check_pages:
        mine = infer(member, paths, z, 1, knot_id, device)
        ref = read_png(M5_PROBS / ("%s.png" % page_name(z)))
        logger.info("check page %d: max |diff| vs knot_probs_m5 = %d", z, int(np.abs(mine.astype(int) - ref).max()))

    for z in range(len(paths)):
        path = out_dir / ("%s.png" % page_name(z))
        if not path.exists():
            write_png(path, infer(member, paths, z, 0, knot_id, device))

    results = evaluate(args.root, out_dir)
    (args.root / "context_ablation.json").write_text(json.dumps(results, indent=1))
    logger.info(json.dumps(results["summary"], indent=1))


def infer(member, paths: List[pathlib.Path], z: int, window: int, knot_id: int, device: torch.device) -> np.ndarray:
    config, backbone, seg_head, transform = member
    stack = load_stack(paths, z, window)[None]
    images = F.interpolate(stack, size=tuple(config.resolution), mode="bilinear", align_corners=False)
    with torch.no_grad():
        probs = knot_probs(backbone, seg_head, transform(images.to(device)), config, knot_id, False)
        probs = F.interpolate(probs, size=tuple(stack.shape[-2:]), mode="bilinear", align_corners=False)
    return np.round(np.clip(probs[0, 0].cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)


def evaluate(root: pathlib.Path, nocontext_dir: pathlib.Path) -> Dict[str, object]:
    knots = json.loads((root / "knots.json").read_text())["knots"]
    gt_ids = {k["id"]: k for k in knots if k["source"] == "gt"}
    pages = sorted({k["page"] for k in gt_ids.values()})
    per_knot: List[Dict[str, object]] = []
    pixel = {name: [0, 0, 0] for name in ("m5", "m5_nocontext")}
    for page in pages:
        label = read_png(root / "annotations" / ("%s.png" % page))
        wood = label > 0
        gt_comps = components(label == 2)
        preds = {
            "m5": (read_png(M5_PROBS / ("%s.png" % page)) >= 128) & wood,
            "m5_nocontext": (read_png(nocontext_dir / ("%s.png" % page)) >= 128) & wood,
        }
        probs = {
            "m5": read_png(M5_PROBS / ("%s.png" % page)),
            "m5_nocontext": read_png(nocontext_dir / ("%s.png" % page)),
        }
        ious = {}
        for name, pred in preds.items():
            gt_iou, _ = match(gt_comps, components(pred))
            ious[name] = gt_iou
            g = label == 2
            pixel[name][0] += int((g & pred).sum())
            pixel[name][1] += int((~g & pred).sum())
            pixel[name][2] += int((g & ~pred).sum())
        for i, comp in enumerate(gt_comps):
            kid = "gt_%s_%02d" % (page, i)
            k = gt_ids.get(kid)
            per_knot.append(
                {
                    "id": kid,
                    "thin": bool(k["thin"]) if k else None,
                    "m4_iou": k["iou"] if k else None,
                    "m5_iou": float(ious["m5"][i]),
                    "m5_nocontext_iou": float(ious["m5_nocontext"][i]),
                    "m5_pmax": float(probs["m5"][comp].max()) / 255.0,
                    "m5_nocontext_pmax": float(probs["m5_nocontext"][comp].max()) / 255.0,
                }
            )
    return {"summary": summarise(per_knot, pixel), "knots": per_knot}


def summarise(per_knot: List[Dict[str, object]], pixel: Dict[str, List[int]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name in ("m5", "m5_nocontext"):
        tp, fp, fn = pixel[name]
        thin = [r for r in per_knot if r["thin"]]
        out[name] = {
            "pixel_iou": round(tp / (tp + fp + fn), 4),
            "precision": round(tp / (tp + fp), 4),
            "recall": round(tp / (tp + fn), 4),
            "per_knot_iou": round(float(np.mean([r["%s_iou" % name] for r in per_knot])), 4),
            "thin_found": sum(r["%s_iou" % name] >= 0.3 for r in thin),
            "n_thin": len(thin),
        }
    m4_missed = [r for r in per_knot if r["thin"] and r["m4_iou"] is not None and r["m4_iou"] < 0.3]
    out["m4_thin_misses"] = [
        {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "thin"} for r in m4_missed
    ]
    return out


if __name__ == "__main__":
    main()
