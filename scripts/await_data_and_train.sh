#!/bin/bash
# Wait for the 14 generated logs, convert them, then launch v1 training.
#
# Runs unattended overnight: polls for the propagation .npz files, converts any
# that are not yet on disk (CPU only, so it never competes with generation for
# the GPU), and starts training under the supervisor once all are present.
#
# Usage: bash scripts/await_data_and_train.sh [deadline_hour]

set -u
cd /home/mary/code/ct-log
set -a
source .env 2>/dev/null
set +a

LOGS="13 14 15 16 17 18 19 20 21 23 24 25 26 27"
NPZ_DIR=experiments/sm2025_subset4_propagate/out
GEN_ROOT=/mnt/D/datasets/ct_log/generated
STATE=logs/kwp_v1_extended
mkdir -p "$STATE" "$GEN_ROOT"
LOG="$STATE/launcher.log"

say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

say "waiting for ${LOGS// /, }"

# Wait until every npz exists and has stopped growing (write finished).
while true; do
    missing=0
    for lg in $LOGS; do
        f="$NPZ_DIR/result_ct${lg}_obb_only.npz"
        [ -f "$f" ] || { missing=$((missing + 1)); continue; }
        a=$(stat -c%s "$f"); sleep 1; b=$(stat -c%s "$f")
        [ "$a" = "$b" ] || missing=$((missing + 1))
    done
    [ "$missing" -eq 0 ] && break
    say "still waiting: $missing/14 not ready"
    sleep 120
done

say "all 14 npz present; converting (CPU only)"
for lg in $LOGS; do
    if [ -d "$GEN_ROOT/$lg/ann" ] && [ "$(ls "$GEN_ROOT/$lg/ann" | wc -l)" -gt 250 ]; then
        say "log $lg already converted, skipping"
        continue
    fi
    CUDA_VISIBLE_DEVICES="" conda run -n ct-log python -m scripts.npz_to_dataset \
        --npz "$NPZ_DIR/result_ct${lg}_obb_only.npz" \
        --log "$lg" \
        --src_img_dir "/mnt/D/datasets/ct_log/CT/$lg" \
        --out_root "$GEN_ROOT" >> "$LOG" 2>&1
    say "converted log $lg ($(ls "$GEN_ROOT/$lg/ann" 2>/dev/null | wc -l) frames)"
done

# Do not start training while the GPU is still busy generating.
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do
    say "GPU still busy, waiting before training"
    sleep 60
done

say "launching v1 training"
exec bash scripts/supervise_kwp.sh src/configs/train_kwp_v1.yaml v1_extended_20logs "$STATE"
