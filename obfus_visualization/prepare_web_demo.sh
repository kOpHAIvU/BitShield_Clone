#!/bin/bash
# Prepare web demo models for Linux/Mac

echo "========================================"
echo "Preparing Web Demo Models"
echo "========================================"

# Default parameters
MODEL_NAME=${1:-ResNetSEBlockIoT}
DATASET_NAME=${2:-IoTID20}
ATTACK_MODE=${3:-pbs}
ATTACK_ITERS=${4:-25}
DEVICE=${5:-cuda}

echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Attack Mode: $ATTACK_MODE"
echo "Attack Iterations: $ATTACK_ITERS"
echo "Device: $DEVICE"
echo ""

python prepare_web_demo_models.py "$MODEL_NAME" "$DATASET_NAME" \
    --attack-mode "$ATTACK_MODE" \
    --attack-iters "$ATTACK_ITERS" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "SUCCESS! Models created successfully"
    echo "========================================"
    echo ""
    echo "Models saved to: models/web_demo/${DATASET_NAME}_${MODEL_NAME}/"
    echo "  - original.pt"
    echo "  - attacked.pt"
    echo "  - protected.pt"
    echo "  - obfus_config.json"
else
    echo ""
    echo "========================================"
    echo "ERROR! Failed to create models"
    echo "========================================"
    exit 1
fi

