#!/bin/bash
# Wait for generated logs, convert them, then launch v1 training - with a hard
# cutoff so the night always produces a trained model.
#
# Failure mode this guards against: waiting forever on data that never lands and
# training nothing at all. At CUTOFF we stop waiting and train on whatever is
# ready, falling back to the existing 6 Phase2 logs if none of the new ones are.
#
# Usage: bash scripts/await_data_and_train.sh [cutoff_HH:MM] [min_logs]

set -u
cd /home/mary/code/ct-log
set -a
source .env 2>/dev/null
set +a

CUTOFF="${1:-01:30}"
MIN_LOGS="${2:-6}"
LOGS="13 14 15 16 17 18 19 20 21 23 24 25 26 27"
NPZ_DIR=experiments/sm2025_subset4_propagate/out
GEN_ROOT=/mnt/D/datasets/ct_log/generated
STATE=logs/kwp_v1_extended
mkdir -p "$STATE" "$GEN_ROOT"
LOG="$STATE/launcher.log"

say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

cutoff_epoch=$(date -d "today $CUTOFF" +%s)
[ "$cutoff_epoch" -lt "$(date +%s)" ] && cutoff_epoch=$(date -d "tomorrow $CUTOFF" +%s)
say "cutoff $CUTOFF; will train on whatever is ready by then (min $MIN_LOGS logs)"

ready_npz() {
    local out=""
    for lg in $LOGS; do
        local f="$NPZ_DIR/result_ct${lg}_obb_only.npz"
        [ -f "$f" ] || continue
        local a b
        a=$(stat -c%s "$f"); sleep 0.3; b=$(stat -c%s "$f")
        [ "$a" = "$b" ] && [ "$a" -gt 1000000 ] && out="$out $lg"
    done
    echo "$out"
}

while true; do
    have=$(ready_npz)
    count=$(echo $have | wc -w)
    now=$(date +%s)
    if [ "$count" -ge 14 ]; then
        say "all 14 npz ready"
        break
    fi
    if [ "$now" -ge "$cutoff_epoch" ]; then
        say "CUTOFF reached with $count/14 logs - proceeding with what we have"
        break
    fi
    say "waiting: $count/14 ready, $(( (cutoff_epoch - now) / 60 ))min to cutoff"
    sleep 120
done

have=$(ready_npz)
say "converting:${have:-（none)}"
for lg in $have; do
    if [ -d "$GEN_ROOT/$lg/ann" ] && [ "$(ls "$GEN_ROOT/$lg/ann" 2>/dev/null | wc -l)" -gt 250 ]; then
        continue
    fi
    CUDA_VISIBLE_DEVICES="" conda run -n ct-log python -m scripts.npz_to_dataset \
        --npz "$NPZ_DIR/result_ct${lg}_obb_only.npz" --log "$lg" \
        --src_img_dir "/mnt/D/datasets/ct_log/CT/$lg" --out_root "$GEN_ROOT" >> "$LOG" 2>&1
    say "converted log $lg ($(ls "$GEN_ROOT/$lg/ann" 2>/dev/null | wc -l) frames)"
done

# Build the train list from logs that actually converted, so a partial batch
# still trains rather than failing on a missing directory.
python3 - "$GEN_ROOT" $have <<'PY' >> "$LOG" 2>&1
import pathlib
import sys

gen_root = pathlib.Path(sys.argv[1])
generated = [
    f"  - {gen_root / log}"
    for log in sys.argv[2:]
    if (gen_root / log / "ann").is_dir() and len(list((gen_root / log / "ann").iterdir())) > 250
]
phase2 = [f"  - /mnt/D/datasets/ct_log/377328_phase2/{log}" for log in ["2", "3", "05", "06", "08", "09"]]

config = pathlib.Path("src/configs/train_kwp_v1.yaml").read_text()
start = config.index("train_logs:")
end = config.index("val_logs:")
config = config[:start] + "train_logs:\n" + "\n".join(phase2 + generated) + "\n" + config[end:]
pathlib.Path("src/configs/train_kwp_v1_active.yaml").write_text(config)
print(f"active config: {len(phase2) + len(generated)} train logs ({len(generated)} generated)")
PY

while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do
    say "GPU busy, waiting"
    sleep 60
done

say "launching training"
exec bash scripts/supervise_kwp.sh src/configs/train_kwp_v1_active.yaml v1_extended "$STATE"
