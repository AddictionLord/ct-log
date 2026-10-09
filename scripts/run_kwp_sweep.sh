#!/bin/bash
# Sequential KWP experiment queue. Each run writes its own log + checkpoint.
# Runs are sequential so the 4GB GPU is never shared between two ViT-L runs.
#
# Usage: bash scripts/run_kwp_sweep.sh <stage>
#   stage1  multi-layer sweep (n_layers 4 vs 2), window fixed at 0
#   stage2  more training data: logs 4+1 vs log 4 alone, val stays log 10

set -u
cd /home/mary/code/ct-log
set -a
source .env 2>/dev/null
set +a

CONFIG=src/configs/train_kwp.yaml
EPOCHS=30
MODELS=/mnt/D/models/ct-log
HC=/mnt/D/datasets/ct_log/378193_human_collection
STAGE="${1:-stage1}"

run() {
    local name="$1" window="$2" n_layers="$3"
    shift 3
    local logdir="logs/kwp_${name}"
    mkdir -p "$logdir"
    if grep -q "Best foreground" "$logdir/run.log" 2>/dev/null; then
        echo "[skip] $name already complete"
        return 0
    fi
    echo "[start] $name window=$window n_layers=$n_layers $(date '+%H:%M:%S')"
    conda run -n ct-log --no-capture-output python -m src.train_kwp \
        --config "$CONFIG" \
        --window "$window" \
        --n_layers "$n_layers" \
        --num_epochs "$EPOCHS" \
        --run_name "$name" \
        --checkpoint_path "$MODELS/kwp_seg_head_${name}.pth" \
        --local_log_dir "$logdir" \
        "$@" \
        > "$logdir/run.log" 2>&1
    echo "[done ] $name rc=$? $(grep -E 'Best foreground' "$logdir/run.log" || echo 'NO RESULT')"
}

case "$STAGE" in
    stage1)
        run multilayer4_window0 0 4
        run multilayer2_window0 0 2
        ;;
    stage2)
        # Two fully human-reviewed training logs instead of one; val stays log 10
        # so every number is comparable with stage1 and the original baseline.
        run twolog_n4_window0 0 4 \
            --train_logs "$HC/4" "$HC/1" --val_logs "$HC/10"
        # 2.5D on top of the best head + more data: does axial context pay once
        # the head and the data floor are both improved?
        run twolog_n4_window1 1 4 \
            --train_logs "$HC/4" "$HC/1" --val_logs "$HC/10"
        ;;
    *)
        echo "unknown stage: $STAGE" >&2
        exit 1
        ;;
esac

echo "[sweep complete] $STAGE $(date '+%H:%M:%S')"
