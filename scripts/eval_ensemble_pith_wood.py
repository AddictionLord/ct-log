import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from src.configs.kwp_training_config import KwpTrainingConfig
from src.dataset.ct_log_kwp_dataset import CTLogKwpDataset
from src.dataset.kwp_mask import KwpMaskBuilder
from src.segmentation_head import build_kwp_model
from src.train_kwp import extract_features, input_channels, make_transform
import torch

NATIVE = 778


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pith vs human clicks and wood IoU for single models and their average."
    )
    parser.add_argument("--member", action="append", required=True, help="name=config.yaml=best_copy.pth")
    parser.add_argument("--log", type=Path, required=True, help="Human-labelled log directory.")
    parser.add_argument("--out", type=Path, required=True, help="JSON with per-slice pith points for the gallery.")
    args = parser.parse_args()

    device = torch.device("cuda")
    members = [load_member(spec, device) for spec in args.member]
    config = members[0]["config"]
    dataset = CTLogKwpDataset([str(args.log)], resolution=tuple(config.resolution), window=config.window)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=4)
    wood_id = KwpMaskBuilder.class_names().index("wood")
    points: Dict[str, List[np.ndarray]] = {m["name"]: [] for m in members}
    softmax_sum: Dict[str, List[int]] = {name: [0, 0] for name in [*points, "ensemble"]}
    gts, paths = [], []
    with torch.no_grad():
        for batch in loader:
            probs_total = None
            for member in members:
                features = extract_features(
                    member["backbone"],
                    member["transform"](batch["image"].to(device)),
                    config.n_layers,
                    config.backbone_bf16,
                )
                probs = member["seg_head"](features).float().softmax(dim=1)
                points[member["name"]].append(member["pith_head"](features).float().cpu().numpy())
                accumulate(softmax_sum[member["name"]], probs, batch["mask"], wood_id)
                probs_total = probs if probs_total is None else probs_total + probs
            accumulate(softmax_sum["ensemble"], probs_total, batch["mask"], wood_id)
            gts.append(batch["pith_xy"].numpy())
            paths += batch["path"]
    gt = np.concatenate(gts)
    valid = gt[:, 2] > 0
    gt_px = np.rint(gt[valid, :2] * NATIVE).astype(int)
    predictions = {name: np.concatenate(values)[valid] * NATIVE for name, values in points.items()}
    predictions["ensemble"] = np.mean([predictions[name] for name in points], axis=0)
    for name, pred in predictions.items():
        report(name, np.rint(pred).astype(int), gt_px, softmax_sum[name])
    valid_paths = [path for path, keep in zip(paths, valid) if keep]
    rows = [
        {
            "path": path,
            "gt": gt_px[i].tolist(),
            "pred": np.rint(predictions["ensemble"][i]).astype(int).tolist(),
            "pred_float": predictions["ensemble"][i].round(3).tolist(),
        }
        for i, path in enumerate(valid_paths)
    ]
    args.out.write_text(json.dumps(rows))
    print("wrote %s (%d slices)" % (args.out, len(rows)))


def load_member(spec: str, device: torch.device) -> dict:
    name, config_path, checkpoint = spec.split("=", 2)
    config = KwpTrainingConfig.from_yaml(config_path)
    backbone, seg_head, pith_head = build_kwp_model(config)
    state = torch.load(checkpoint, map_location="cpu")
    seg_head.load_state_dict(state["seg_head"])
    if state.get("backbone") is not None:
        backbone.load_state_dict(state["backbone"], strict=False)
    if state.get("pith_head"):
        pith_head.load_state_dict(state["pith_head"])
    return {
        "name": name,
        "config": config,
        "backbone": backbone.to(device).eval(),
        "seg_head": seg_head.to(device).eval(),
        "pith_head": pith_head.to(device).eval(),
        "transform": make_transform(input_channels(config.window)),
    }


def accumulate(totals: List[int], probs: torch.Tensor, mask: torch.Tensor, wood_id: int) -> None:
    pred = probs.argmax(dim=1).cpu() == wood_id
    gt = mask == wood_id
    totals[0] += int((pred & gt).sum())
    totals[1] += int((pred | gt).sum())


def report(name: str, pred: np.ndarray, gt: np.ndarray, wood: List[int]) -> None:
    dist = np.hypot(pred[:, 0] - gt[:, 0], pred[:, 1] - gt[:, 1])
    print(
        "%-12s pith n=%d exact %.0f%%  <=1px %.0f%%  <=2px %.0f%%  mean %.2f px | wood IoU %.4f"
        % (
            name,
            len(dist),
            100 * np.mean(dist == 0),
            100 * np.mean(dist <= 1),
            100 * np.mean(dist <= 2),
            dist.mean(),
            wood[0] / wood[1],
        )
    )


if __name__ == "__main__":
    main()
