#!/bin/bash
# Script để build TVM .so files từ models
# Usage: ./build_so.sh [options]

set -e  # Exit on error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

COMPILER="tvm"
COMPILER_VER="main"
MODEL="ResNetSEBlockIoT"
DATASET="IoTID20"
CIG="ncnp"
DIG="nd"
AVX=true
OPT_LEVEL=3
CHECK_ACC=false
FORCE=false

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Build TVM/Glow/NNFusion .so files from trained models"
    echo ""
    echo "Options:"
    echo "  -c, --compiler COMPILER      Compiler to use (tvm, glow, nnfusion) [default: tvm]"
    echo "  -v, --compiler-ver VERSION   Compiler version [default: main]"
    echo "  -m, --model MODEL            Model name (ResNetSEBlockIoT, SimpleCNNIoT, PureCNN, EfficientCNN) [default: ResNetSEBlockIoT]"
    echo "  -d, --dataset DATASET        Dataset name (IoTID20, WUSTL, CICIoT2023) [default: IoTID20]"
    echo "  -i, --cig CIG                CIG mode (nc, ncnp, cc1, cc2) [default: ncnp]"
    echo "  -I, --dig DIG                DIG mode (nd, gn1, gn2, gninf, id, rb, cb) [default: nd]"
    echo "  -X, --no-avx                 Disable AVX optimization"
    echo "  -O, --opt-level LEVEL        Optimization level (0-3) [default: 3]"
    echo "  -A, --check-acc              Check accuracy after build"
    echo "  -f, --force                  Force rebuild even if file exists"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Build ResNetSEBlockIoT on IoTID20 with no defense"
    echo "  $0 -m ResNetSEBlockIoT -d IoTID20 -I nd"
    echo ""
    echo "  # Build with gradient norm defense"
    echo "  $0 -m ResNetSEBlockIoT -d IoTID20 -I gn1"
    echo ""
    echo "  # Build SimpleCNNIoT on WUSTL dataset"
    echo "  $0 -m SimpleCNNIoT -d WUSTL -I nd"
    echo ""
    echo "  # Build with force rebuild"
    echo "  $0 -m ResNetSEBlockIoT -d IoTID20 -I nd -f"
    echo ""
    echo "  # Build multiple models"
    echo "  for model in ResNetSEBlockIoT SimpleCNNIoT PureCNN EfficientCNN; do"
    echo "    $0 -m \$model -d IoTID20 -I nd"
    echo "  done"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--compiler)
            COMPILER="$2"
            shift 2
            ;;
        -v|--compiler-ver)
            COMPILER_VER="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -i|--cig)
            CIG="$2"
            shift 2
            ;;
        -I|--dig)
            DIG="$2"
            shift 2
            ;;
        -X|--no-avx)
            AVX=false
            shift
            ;;
        -O|--opt-level)
            OPT_LEVEL="$2"
            shift 2
            ;;
        -A|--check-acc)
            CHECK_ACC=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Build command
BUILD_CMD="python buildmodels.py"
BUILD_CMD="$BUILD_CMD --compiler $COMPILER"
BUILD_CMD="$BUILD_CMD --compiler_ver $COMPILER_VER"
BUILD_CMD="$BUILD_CMD --model $MODEL"
BUILD_CMD="$BUILD_CMD --dataset $DATASET"
BUILD_CMD="$BUILD_CMD --cig $CIG"
BUILD_CMD="$BUILD_CMD --dig $DIG"
BUILD_CMD="$BUILD_CMD --opt-level $OPT_LEVEL"

if [ "$AVX" = false ]; then
    BUILD_CMD="$BUILD_CMD --no-avx"
fi

if [ "$CHECK_ACC" = true ]; then
    BUILD_CMD="$BUILD_CMD --check-acc"
else
    BUILD_CMD="$BUILD_CMD --no-check-acc"
fi

if [ "$FORCE" = true ]; then
    BUILD_CMD="$BUILD_CMD --force"
fi

# Print build information
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Building .so File${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Compiler:     ${GREEN}$COMPILER${NC} ($COMPILER_VER)"
echo -e "Model:        ${GREEN}$MODEL${NC}"
echo -e "Dataset:      ${GREEN}$DATASET${NC}"
echo -e "CIG Mode:     ${GREEN}$CIG${NC}"
echo -e "DIG Mode:     ${GREEN}$DIG${NC}"
echo -e "AVX:          ${GREEN}$AVX${NC}"
echo -e "Opt Level:    ${GREEN}$OPT_LEVEL${NC}"
echo -e "Check Acc:    ${GREEN}$CHECK_ACC${NC}"
echo -e "Force:        ${GREEN}$FORCE${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if TVM is available (for TVM compiler)
if [ "$COMPILER" = "tvm" ]; then
    if ! python -c "import tvm" 2>/dev/null; then
        echo -e "${RED}Error: TVM is not available!${NC}"
        echo -e "${YELLOW}Please install TVM or activate the TVM environment.${NC}"
        echo -e "${YELLOW}See README_BUILD.md for setup instructions.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ TVM is available${NC}"
fi

# Check if model file exists
MODEL_FILE="models/$DATASET/$MODEL/$MODEL.pt"
if [ ! -f "$MODEL_FILE" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL_FILE${NC}"
    echo -e "${YELLOW}Please train the model first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Model file found: $MODEL_FILE${NC}"
echo ""

# Run build
echo -e "${BLUE}Running build command...${NC}"
echo -e "${YELLOW}$BUILD_CMD${NC}"
echo ""

if eval $BUILD_CMD; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # Show output file
    OUTPUT_FILE="built/${COMPILER}-${COMPILER_VER}-${MODEL}-${DATASET}-${CIG}-${DIG}.so"
    if [ "$AVX" = false ]; then
        OUTPUT_FILE="built/${COMPILER}-${COMPILER_VER}-${MODEL}-${DATASET}-${CIG}-${DIG}-noavx.so"
    fi
    if [ "$OPT_LEVEL" != "3" ]; then
        OUTPUT_FILE="${OUTPUT_FILE%.so}-${OPT_LEVEL}.so"
    fi
    
    if [ -f "$OUTPUT_FILE" ]; then
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo -e "Output file: ${GREEN}$OUTPUT_FILE${NC}"
        echo -e "File size:   ${GREEN}$FILE_SIZE${NC}"
    fi
    
    # Show output defs file
    OUTPUT_DEFS="${OUTPUT_FILE}.json"
    OUTPUT_DEFS="built-aux/output-defs/$(basename $OUTPUT_DEFS)"
    if [ -f "$OUTPUT_DEFS" ]; then
        echo -e "Output defs:  ${GREEN}$OUTPUT_DEFS${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}You can now use this .so file for:${NC}"
    echo -e "  - Attack simulation (flipsweep.py, attack_with_defense.py)"
    echo -e "  - Ghidra analysis"
    echo -e "  - Performance benchmarking"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Build failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

