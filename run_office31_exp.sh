#!/bin/bash
# run_office31_exp.sh
# Run Office-31 domain adaptation experiments for all 6 directions

# Configuration
PYTHON_CMD="python"
SCRIPT="main.py"
OUTPUT_BASE="./output/office31-"
DATA_ROOT="./data/office31"
SEED=42

# Epoch settings
SOURCE_EPOCHS=30
TARGET_EPOCHS=50

# Batch size settings
# Set SOURCE_BATCH_SIZE/TARGET_BATCH_SIZE to override independently
# If set to 0, uses BATCH_SIZE for both
SOURCE_BATCH_SIZE=64
TARGET_BATCH_SIZE=64

# Enumerate all 6 directions (SOURCE -> TARGET)
# Comment out any line to skip that experiment
DIRECTIONS=(
    "amazon:webcam"
    "amazon:dslr"
    "webcam:amazon"
    "webcam:dslr"
    "dslr:amazon"
    "dslr:webcam"
)

mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "Office-31 Domain Adaptation Experiments"
echo "=========================================="
echo ""

EXP_NUM=1
for DIRECTION in "${DIRECTIONS[@]}"; do
    SOURCE="${DIRECTION%%:*}"
    TARGET="${DIRECTION##*:}"

    echo "------------------------------------------"
    echo "Experiment [$EXP_NUM]: $SOURCE -> $TARGET"
    echo "------------------------------------------"

    OUTPUT_DIR="$OUTPUT_BASE/${SOURCE}_to_${TARGET}"

    # Build batch_size args
    if [ "$SOURCE_BATCH_SIZE" -gt 0 ]; then
        BS_ARGS="--source_batch_size $SOURCE_BATCH_SIZE --target_batch_size ${TARGET_BATCH_SIZE:-$SOURCE_BATCH_SIZE}"
    else
        BS_ARGS=""
    fi

    $PYTHON_CMD $SCRIPT \
        --source "$SOURCE" \
        --target "$TARGET" \
        --output_dir "$OUTPUT_DIR" \
        --data_root "$DATA_ROOT" \
        --seed "$SEED" \
        --source_epochs "$SOURCE_EPOCHS" \
        --target_epochs "$TARGET_EPOCHS" \
        $BS_ARGS

    echo "✓ Completed: $SOURCE -> $TARGET"
    echo ""

    EXP_NUM=$((EXP_NUM + 1))
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
