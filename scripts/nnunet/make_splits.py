import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Duplicate human cases (symlinks) in preprocessed folders and write a single custom fold.")
    parser.add_argument("--preprocessed", type=Path, required=True, help="nnUNet_preprocessed/DatasetXXX")
    parser.add_argument("--configs", nargs="+", default=["nnUNetPlans_2d", "nnUNetPlans_3d_fullres"])
    parser.add_argument("--human", nargs="+", default=["ctlog_1", "ctlog_2", "ctlog_4", "ctlog_08"])
    parser.add_argument("--copies", type=int, default=4, help="Extra copies per human case (4 -> 5x total).")
    parser.add_argument("--val", nargs="+", default=["ctlog_58", "ctlog_59", "ctlog_60"])
    args = parser.parse_args()
    cases = sorted(p.name[: -len(".pkl")] for p in (args.preprocessed / args.configs[0]).glob("*.pkl") if "_dup" not in p.name)
    duplicates = []
    for config in args.configs:
        folder = args.preprocessed / config
        for case in args.human:
            for k in range(1, args.copies + 1):
                dup = "%s_dup%d" % (case, k)
                for suffix in (".b2nd", ".pkl", "_seg.b2nd"):
                    link = folder / (dup + suffix)
                    if not link.exists():
                        os.symlink(case + suffix, link)
                if config == args.configs[0]:
                    duplicates.append(dup)
    train = [c for c in cases if c not in args.val] + duplicates
    (args.preprocessed / "splits_final.json").write_text(json.dumps([{"train": train, "val": args.val}], indent=1))
    print("cases %d, duplicates %d, train %d, val %s" % (len(cases), len(duplicates), len(train), args.val))


if __name__ == "__main__":
    main()
