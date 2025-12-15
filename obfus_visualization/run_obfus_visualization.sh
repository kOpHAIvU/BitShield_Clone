#!/bin/bash
# ========================================================================
# OBFUS Defense Visualization Pipeline for Linux/Mac
# ========================================================================

set -e  # Exit on error

echo ""
echo "========================================================================"
echo "  OBFUS Defense Experiment and Visualization"
echo "========================================================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
fi

echo "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Default parameters
MODEL="${1:-ResNetSEBlockIoT}"
DATASET="${2:-CICIoT2023}"
DEVICE="${3:-cuda}"
ATTACK_ITERS="${4:-25}"
ATTACK_MODES="${5:-pbs,random,pbs2random,random2pbs}"
SIG_PERIOD="${6:-20}"
SIG_K="${7:-3.0}"
OBFUS_TARGETS="${8:-linear,conv1d}"
OBFUS_AUTO_RESEED="${9:-10}"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Device: $DEVICE"
echo "  Attack Iterations: $ATTACK_ITERS"
echo "  Attack Modes: $ATTACK_MODES"
echo ""

# Activate virtual environment if exists
if [ -d "bitshield/bin" ]; then
    echo "Activating virtual environment: bitshield"
    source bitshield/bin/activate
elif [ -d "venv/bin" ]; then
    echo "Activating virtual environment: venv"
    source venv/bin/activate
else
    echo "Warning: No virtual environment found, using system Python"
fi

echo ""
echo "========================================================================"
echo "  Running OBFUS Experiment Pipeline"
echo "========================================================================"
echo ""

# Change to project root
cd ..

# Run the full pipeline
$PYTHON_CMD obfus_visualization/run_full_obfus_pipeline.py "$MODEL" "$DATASET" \
  --device "$DEVICE" \
  --attack-iters "$ATTACK_ITERS" \
  --attack-modes "$ATTACK_MODES" \
  --sig-period "$SIG_PERIOD" \
  --sig-k "$SIG_K" \
  --obfus-targets "$OBFUS_TARGETS" \
  --obfus-auto-reseed "$OBFUS_AUTO_RESEED"

echo ""
echo "========================================================================"
echo "  Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Results saved to: results/obfus_experiments/${DATASET}_${MODEL}_obfus_experiment.json"
echo "Visualizations saved to: results/obfus_visualizations/${DATASET}_${MODEL}/"
echo ""
echo "To view visualizations, open the PNG files in:"
echo "  results/obfus_visualizations/${DATASET}_${MODEL}/"
echo ""

