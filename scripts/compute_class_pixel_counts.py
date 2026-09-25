import argparse
import json
from pathlib import Path

from src.configs.kwp_training_config import KwpTrainingConfig
from src.dataset.kwp_mask import KwpMaskBuilder
from src.utils.class_balance import effective_number_weights, inverse_frequency_weights
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="src/configs/train_kwp_v1.yaml")
    parser.add_argument("--out", type=str, default=None, help="Write class_pixel_counts as YAML here.")
    args = parser.parse_args()

    config = KwpTrainingConfig.from_yaml(args.config)
    builder = KwpMaskBuilder(pith_radius=config.pith_radius)
    names = KwpMaskBuilder.class_names()
    counts = [0] * config.num_classes
    n_frames = 0

    for log_dir in config.train_logs:
        ann_dir = Path(log_dir) / "ann"
        for ann_path in sorted(ann_dir.glob("*.json")):
            with ann_path.open() as f:
                mask = builder.build(json.load(f))
            for class_id in range(len(counts)):
                counts[class_id] += int((mask == class_id).sum())
            n_frames += 1

    total = sum(counts)
    print(f"n_frames: {n_frames}  train_logs: {len(config.train_logs)}")
    for class_id, name in enumerate(names):
        print(f"  {name:<12} {counts[class_id]:>14d} px  ({100 * counts[class_id] / total:.4f}%)")

    print("\nEffective-number weights (Cui et al. 2019):")
    for beta in (0.9, 0.99, 0.999, 0.9999):
        weights = effective_number_weights(counts, beta=beta)
        print(f"  beta={beta:<7} {[round(w, 4) for w in weights.tolist()]}")

    print("\nInverse-frequency weights:")
    for power in (1.0, 2.0):
        weights = inverse_frequency_weights(counts, power=power)
        print(f"  power={power:<5} {[round(w, 4) for w in weights.tolist()]}")

    if args.out:
        Path(args.out).write_text(yaml.safe_dump({"class_pixel_counts": counts}, sort_keys=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
