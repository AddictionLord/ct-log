import argparse
import json
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a thin-z 3D configuration (no pooling along z) to nnUNetPlans.json."
    )
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--z", type=int, required=True)
    parser.add_argument("--xy", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument(
        "--z_kernel_stages", type=int, default=2, help="Leading stages with (3, 3, 3) kernels; the rest use (1, 3, 3)."
    )
    args = parser.parse_args()

    backup = args.plans.with_suffix(".json.orig")
    if not backup.exists():
        shutil.copy(args.plans, backup)
    plans = json.loads(args.plans.read_text())
    base = plans["configurations"]["3d_fullres"]["architecture"]
    n_stages = 7
    if args.xy % 2 ** (n_stages - 1):
        msg = "xy %d is not divisible by %d" % (args.xy, 2 ** (n_stages - 1))
        raise ValueError(msg)
    arch = json.loads(json.dumps(base))
    arch["arch_kwargs"]["n_stages"] = n_stages
    arch["arch_kwargs"]["kernel_sizes"] = [
        [3, 3, 3] if s < args.z_kernel_stages else [1, 3, 3] for s in range(n_stages)
    ]
    arch["arch_kwargs"]["strides"] = [[1, 1, 1]] + [[1, 2, 2]] * (n_stages - 1)
    plans["configurations"][args.name] = {
        "inherits_from": "3d_fullres",
        "data_identifier": "nnUNetPlans_3d_fullres",
        "patch_size": [args.z, args.xy, args.xy],
        "batch_size": args.batch,
        "architecture": arch,
    }
    args.plans.write_text(json.dumps(plans, indent=4))
    print(
        args.name,
        json.dumps(plans["configurations"][args.name]["patch_size"]),
        "batch",
        args.batch,
        "kernels",
        arch["arch_kwargs"]["kernel_sizes"],
        "strides",
        arch["arch_kwargs"]["strides"],
    )


if __name__ == "__main__":
    main()
