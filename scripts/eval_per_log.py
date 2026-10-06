import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from src.configs.kwp_training_config import KwpTrainingConfig
from src.dataset.ct_log_kwp_dataset import CTLogKwpDataset
from src.dataset.kwp_mask import KwpMaskBuilder
from src.segmentation_head import build_kwp_model
from src.train_kwp import extract_features, make_transform, pith_pixel_errors
import torch

FG_CLASSES = ("wood", "knot")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-log evaluation of best-model copies with a paired block bootstrap over slices."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="name=config.yaml=best_copy.pth; the first model is the reference for paired differences.",
    )
    parser.add_argument("--logs", nargs="+", required=True, help="Human-labelled log directories.")
    parser.add_argument("--block", type=int, default=20, help="Consecutive slices per bootstrap block.")
    parser.add_argument("--n_boot", type=int, default=2000, help="Bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON with per-frame stats and summary.")
    args = parser.parse_args()

    device = torch.device("cuda")
    class_names = KwpMaskBuilder.class_names()
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for spec in args.model:
        name, config_path, copy_path = spec.split("=")
        stats[name] = evaluate_model(Path(config_path), Path(copy_path), args.logs, class_names, device)
        print("evaluated %s" % name, flush=True)

    rng = np.random.default_rng(args.seed)
    logs = [Path(log).name for log in args.logs]
    blocks = {log: block_indices(len(stats[next(iter(stats))][log]["pith"]), args.block) for log in logs}
    draws = [{log: rng.integers(0, len(blocks[log]), len(blocks[log])) for log in logs} for _ in range(args.n_boot)]

    summary = summarize(stats, logs, blocks, draws, class_names)
    print_summary(summary, logs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_frame = {
        name: {log: {key: values.tolist() for key, values in arrays.items()} for log, arrays in by_log.items()}
        for name, by_log in stats.items()
    }
    args.out.write_text(json.dumps({"summary": summary, "per_frame": per_frame, "args": vars(args)}, default=str))
    print("wrote %s" % args.out)


def evaluate_model(
    config_path: Path, copy_path: Path, logs: List[str], class_names: List[str], device: torch.device
) -> Dict[str, Dict[str, np.ndarray]]:
    config = KwpTrainingConfig.from_yaml(config_path)
    backbone, seg_head, pith_head = build_kwp_model(config)
    state = torch.load(copy_path, map_location="cpu")
    seg_head.load_state_dict(state["seg_head"])
    if state.get("backbone") is not None:
        backbone.load_state_dict(state["backbone"], strict=False)
    if pith_head is not None and state.get("pith_head") is not None:
        pith_head.load_state_dict(state["pith_head"])
        pith_head = pith_head.to(device).eval()
    else:
        pith_head = None
    backbone, seg_head = backbone.to(device).eval(), seg_head.to(device).eval()
    transform = make_transform()

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for log in logs:
        dataset = CTLogKwpDataset(
            [log], resolution=config.resolution, window=config.window, pith_radius=config.pith_radius
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
        results[Path(log).name] = frame_stats(
            backbone, seg_head, pith_head, loader, transform, device, config, class_names
        )
    return results


@torch.no_grad()
def frame_stats(
    backbone: torch.nn.Module,
    seg_head: torch.nn.Module,
    pith_head: Optional[torch.nn.Module],
    loader: torch.utils.data.DataLoader,
    transform: torch.nn.Module,
    device: torch.device,
    config: KwpTrainingConfig,
    class_names: List[str],
) -> Dict[str, np.ndarray]:
    inter: Dict[str, List[float]] = {name: [] for name in FG_CLASSES}
    union: Dict[str, List[float]] = {name: [] for name in FG_CLASSES}
    pith: List[float] = []
    for batch in loader:
        images = transform(batch["image"].to(device))
        masks = batch["mask"].to(device)
        features = extract_features(backbone, images, config.n_layers, config.backbone_bf16)
        preds = seg_head(features).argmax(dim=1)
        for name in FG_CLASSES:
            class_id = class_names.index(name)
            pred_c, target_c = preds == class_id, masks == class_id
            inter[name].extend((pred_c & target_c).flatten(1).sum(1).tolist())
            union[name].extend((pred_c | target_c).flatten(1).sum(1).tolist())
        targets = batch["pith_xy"].to(device)
        if pith_head is None:
            pith.extend([float("nan")] * len(targets))
            continue
        errors = iter(pith_pixel_errors(pith_head(features), targets, config))
        pith.extend(next(errors) if valid > 0 else float("nan") for valid in targets[:, 2].tolist())
    arrays = {f"inter_{name}": np.array(inter[name]) for name in FG_CLASSES}
    arrays |= {f"union_{name}": np.array(union[name]) for name in FG_CLASSES}
    arrays["pith"] = np.array(pith)
    return arrays


def block_indices(n_frames: int, block: int) -> List[np.ndarray]:
    return [np.arange(start, min(start + block, n_frames)) for start in range(0, n_frames, block)]


def metrics(arrays_by_log: Dict[str, Dict[str, np.ndarray]], frames: Dict[str, np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in FG_CLASSES:
        inter = sum(arrays_by_log[log][f"inter_{name}"][idx].sum() for log, idx in frames.items())
        union = sum(arrays_by_log[log][f"union_{name}"][idx].sum() for log, idx in frames.items())
        out[f"iou_{name}"] = float(inter / union) if union > 0 else float("nan")
    out["mean_iou_fg"] = float(np.mean([out[f"iou_{name}"] for name in FG_CLASSES]))
    pith = np.concatenate([arrays_by_log[log]["pith"][idx] for log, idx in frames.items()])
    out["pith_median_px"] = float(np.nanmedian(pith)) if np.isfinite(pith).any() else float("nan")
    return out


def resample(
    blocks: Dict[str, List[np.ndarray]], draw: Dict[str, np.ndarray], logs: List[str]
) -> Dict[str, np.ndarray]:
    return {log: np.concatenate([blocks[log][i] for i in draw[log]]) for log in logs}


def summarize(
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    logs: List[str],
    blocks: Dict[str, List[np.ndarray]],
    draws: List[Dict[str, np.ndarray]],
    class_names: List[str],
) -> Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]]:
    scopes = {log: [log] for log in logs} | {"pooled": logs}
    names = list(stats)
    reference = names[0]
    summary: Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]] = {}
    for scope, scope_logs in scopes.items():
        full = {log: np.concatenate(blocks[log]) for log in scope_logs}
        boot = {name: [metrics(stats[name], resample(blocks, draw, scope_logs)) for draw in draws] for name in names}
        summary[scope] = {}
        for name in names:
            point = metrics(stats[name], full)
            entry = {key: ci(point[key], [b[key] for b in boot[name]]) for key in point}
            if name != reference:
                ref_point = metrics(stats[reference], full)
                for key in point:
                    diffs = [b[key] - r[key] for b, r in zip(boot[name], boot[reference])]
                    entry[f"diff_{key}"] = ci(point[key] - ref_point[key], diffs)
            summary[scope][name] = entry
    return summary


def ci(point: float, samples: List[float]) -> Tuple[float, float, float]:
    values = np.array(samples, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (point, float("nan"), float("nan"))
    return (point, float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def print_summary(summary: Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]], logs: List[str]) -> None:
    for scope in [*logs, "pooled"]:
        print("== %s" % scope)
        for name, entry in summary[scope].items():
            line = "  %-14s" % name
            for key in ("iou_knot", "mean_iou_fg", "pith_median_px"):
                point, low, high = entry[key]
                line += "  %s %.3f [%.3f, %.3f]" % (key, point, low, high)
            print(line)
            for key in ("iou_knot", "mean_iou_fg"):
                if f"diff_{key}" in entry:
                    point, low, high = entry[f"diff_{key}"]
                    print("  %-14s  diff_%s %+.4f [%+.4f, %+.4f]" % ("", key, point, low, high))


if __name__ == "__main__":
    main()
