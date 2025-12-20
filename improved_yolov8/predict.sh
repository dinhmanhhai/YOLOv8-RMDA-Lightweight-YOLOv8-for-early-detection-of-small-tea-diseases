#!/bin/bash

# Prediction script for Improved YOLOv8
# Usage: ./predict.sh [source] [options]

set -e  # Exit on error

# ==================== Configuration ====================
MODEL_PATH="${1:-runs/train/exp/weights/best.pt}"
SOURCE="${2:-dataset/test/images}"
IMG_SIZE=640
DEVICE=0
PROJECT="runs"
NAME="predict"
CONF=0.25  # Confidence threshold
IOU=0.45   # IoU threshold for NMS

# ==================== Functions ====================

print_usage() {
    echo "Usage: $0 [MODEL_PATH] [SOURCE] [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  MODEL_PATH              Path to model weights (default: $MODEL_PATH)"
    echo "  SOURCE                  Source for prediction: image, video, folder, or webcam (default: $SOURCE)"
    echo ""
    echo "Options:"
    echo "  -s, --size N           Image size (default: $IMG_SIZE)"
    echo "  --device N              GPU device ID or 'cpu' (default: $DEVICE)"
    echo "  --conf N                Confidence threshold (default: $CONF)"
    echo "  --iou N                 IoU threshold for NMS (default: $IOU)"
    echo "  --project NAME          Project name (default: $PROJECT)"
    echo "  --name NAME             Experiment name (default: $NAME)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Predict with defaults"
    echo "  $0 runs/train/exp/weights/best.pt dataset/test/images  # Specify model and source"
    echo "  $0 --source 0                        # Use webcam (device 0)"
    echo "  $0 --source video.mp4                # Predict on video"
    echo "  $0 --conf 0.5 --iou 0.5              # Adjust thresholds"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--size)
            IMG_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --conf)
            CONF="$2"
            shift 2
            ;;
        --iou)
            IOU="$2"
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
        --source)
            SOURCE="$2"
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
            elif [ -z "$SOURCE_SET" ]; then
                SOURCE="$1"
                SOURCE_SET=1
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

# ==================== Prediction ====================

echo "=========================================="
echo "Improved YOLOv8 Prediction"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model:          $MODEL_PATH"
echo "  Source:         $SOURCE"
echo "  Image size:     $IMG_SIZE"
echo "  Device:         $DEVICE"
echo "  Confidence:     $CONF"
echo "  IoU threshold:  $IOU"
echo ""

# Import and register custom modules
echo "Importing custom modules..."
python -c "from improved_yolov8 import utils" || {
    echo "Error: Failed to import custom modules"
    exit 1
}
echo "✓ Custom modules registered"
echo ""

# Run prediction
echo "Running prediction..."
yolo predict \
    model="$MODEL_PATH" \
    source="$SOURCE" \
    imgsz=$IMG_SIZE \
    device=$DEVICE \
    conf=$CONF \
    iou=$IOU \
    project="$PROJECT" \
    name="$NAME"

echo ""
echo "=========================================="
echo "Prediction completed!"
echo "=========================================="
echo ""
echo "Results saved to: $PROJECT/$NAME/"

