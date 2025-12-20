#!/bin/bash

# Validation script for Improved YOLOv8
# Usage: ./val.sh [model_path] [options]

set -e  # Exit on error

# ==================== Configuration ====================
MODEL_PATH="${1:-runs/train/exp/weights/best.pt}"
DATA_CONFIG="dataset/data.yaml"
IMG_SIZE=640
DEVICE=0
PROJECT="runs"
NAME="val"

# ==================== Functions ====================

print_usage() {
    echo "Usage: $0 [MODEL_PATH] [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  MODEL_PATH              Path to model weights (default: $MODEL_PATH)"
    echo ""
    echo "Options:"
    echo "  -d, --data PATH        Data config file (default: $DATA_CONFIG)"
    echo "  -s, --size N           Image size (default: $IMG_SIZE)"
    echo "  --device N             GPU device ID or 'cpu' (default: $DEVICE)"
    echo "  --project NAME         Project name (default: $PROJECT)"
    echo "  --name NAME            Experiment name (default: $NAME)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Validate with default model"
    echo "  $0 runs/train/exp2/weights/best.pt    # Validate specific model"
    echo "  $0 --device cpu                      # Use CPU"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data)
            DATA_CONFIG="$2"
            shift 2
            ;;
        -s|--size)
            IMG_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --name)
            NAME="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            if [ -z "$MODEL_PATH_SET" ]; then
                MODEL_PATH="$1"
                MODEL_PATH_SET=1
            fi
            shift
            ;;
    esac
done

# ==================== Validation ====================

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$DATA_CONFIG" ]; then
    echo "Error: Data config file not found: $DATA_CONFIG"
    exit 1
fi

# ==================== Validation ====================

echo "=========================================="
echo "Improved YOLOv8 Validation"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model:          $MODEL_PATH"
echo "  Data config:    $DATA_CONFIG"
echo "  Image size:     $IMG_SIZE"
echo "  Device:         $DEVICE"
echo ""

# Import and register custom modules
echo "Importing custom modules..."
python -c "from improved_yolov8 import utils" || {
    echo "Error: Failed to import custom modules"
    exit 1
}
echo "✓ Custom modules registered"
echo ""

# Run validation
echo "Running validation..."
yolo val \
    model="$MODEL_PATH" \
    data="$DATA_CONFIG" \
    imgsz=$IMG_SIZE \
    device=$DEVICE \
    project="$PROJECT" \
    name="$NAME"

echo ""
echo "=========================================="
echo "Validation completed!"
echo "=========================================="
echo ""
echo "Results saved to: $PROJECT/$NAME/"

