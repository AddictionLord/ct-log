import argparse
import json
from pathlib import Path
import shutil

Z_POOL_STAGES = (2, 3, 4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a deep-z 3D configuration: (3,3,3) everywhere, z pooling at three stages."
    )
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--z", type=int, required=True)
    parser.add_argument("--xy", type=int, default=384)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    n_stages = 7
    if args.z % 2 ** len(Z_POOL_STAGES) or args.xy % 2 ** (n_stages - 1):
        msg = "z must be divisible by %d and xy by %d" % (2 ** len(Z_POOL_STAGES), 2 ** (n_stages - 1))
        raise ValueError(msg)
    backup = args.plans.with_suffix(".json.orig")
    if not backup.exists():
        shutil.copy(args.plans, backup)
    plans = json.loads(args.plans.read_text())
    arch = json.loads(json.dumps(plans["configurations"]["3d_fullres"]["architecture"]))
    arch["arch_kwargs"]["n_stages"] = n_stages
    arch["arch_kwargs"]["kernel_sizes"] = [[3, 3, 3]] * n_stages
    arch["arch_kwargs"]["strides"] = [[1, 1, 1]] + [[2 if s in Z_POOL_STAGES else 1, 2, 2] for s in range(1, n_stages)]
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
        [args.z, args.xy, args.xy],
        "batch",
        args.batch,
        "strides",
        arch["arch_kwargs"]["strides"],
        "z_rf",
        z_receptive_field(arch["arch_kwargs"]),
    )


def z_receptive_field(kwargs: dict) -> int:
    rf, jump, jumps = 1, 1, []
    for kernel, stride in zip(kwargs["kernel_sizes"], kwargs["strides"]):
        rf += (kernel[0] - 1) * jump
        jump *= stride[0]
        rf += (kernel[0] - 1) * jump
        jumps.append(jump)
    for kernel, jump in zip(reversed(kwargs["kernel_sizes"][:-1]), reversed(jumps[:-1])):
        rf += 2 * (kernel[0] - 1) * jump
    return rf


if __name__ == "__main__":
    main()
