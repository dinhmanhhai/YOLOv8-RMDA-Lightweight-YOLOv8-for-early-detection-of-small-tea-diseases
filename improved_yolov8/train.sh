#!/bin/bash

# Training script for Improved YOLOv8 with RFCBAM, MixSPPF, RepGFPN, and Dynamic Head
# Usage: ./train.sh [options]

set -e  # Exit on error

# ==================== WandB API Key ====================
# API Key for WandB
# Get API Key from https://wandb.ai/settings
# Save it to ~/.netrc file
# machine wandb.ai
# login <your_wandb_username>
# password <your_wandb_api_key>
# Save it to ~/.netrc file
# machine wandb.ai
# login <your_wandb_username>
# password <your_wandb_api_key>
export WANDB_API_KEY="654322757bc621b514dc2592badff0c6eeefe6ad"

# ==================== Configuration ====================
# Model configuration
MODEL_CONFIG="improved_yolov8/configs/yolov8-rfcbam-dynamic.yaml"
DATA_CONFIG="dataset/data.yaml"
PRETRAINED="yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Training parameters
EPOCHS=150
IMG_SIZE=640
BATCH=16
DEVICE=0  # GPU device ID, use "cpu" for CPU training

# Project and experiment names
PROJECT="runs"
NAME="exp"

# WandB configuration (set to "true" to enable, "false" to disable)
USE_WANDB="true"
WANDB_PROJECT="yolov8-rfcbam"
WANDB_NAME="experiment_1"

# Resume training (set to checkpoint path or empty to start new training)
RESUME=""

# Additional arguments (add any other YOLO CLI arguments here)
EXTRA_ARGS=""

# ==================== Functions ====================

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model PATH          Model config file (default: $MODEL_CONFIG)"
    echo "  -d, --data PATH           Data config file (default: $DATA_CONFIG)"
    echo "  -p, --pretrained PATH     Pretrained weights (default: $PRETRAINED)"
    echo "  -e, --epochs N            Number of epochs (default: $EPOCHS)"
    echo "  -s, --size N              Image size (default: $IMG_SIZE)"
    echo "  -b, --batch N             Batch size (default: $BATCH)"
    echo "  --device N                GPU device ID or 'cpu' (default: $DEVICE)"
    echo "  --project NAME            Project name (default: $PROJECT)"
    echo "  --name NAME               Experiment name (default: $NAME)"
    echo "  --wandb                   Enable WandB logging (default: enabled)"
    echo "  --no-wandb                Disable WandB logging"
    echo "  --wandb-project NAME     WandB project name (default: $WANDB_PROJECT)"
    echo "  --wandb-name NAME         WandB run name (default: $WANDB_NAME)"
    echo "  -r, --resume PATH         Resume from checkpoint"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run with default settings"
    echo "  $0 -e 200 -b 32                      # Train for 200 epochs with batch 32"
    echo "  $0 --resume runs/train/exp/weights/last.pt  # Resume training"
    echo "  $0 --no-wandb                         # Disable WandB"
    echo "  $0 --device cpu                       # Use CPU"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        -d|--data)
            DATA_CONFIG="$2"
            shift 2
            ;;
        -p|--pretrained)
            PRETRAINED="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -s|--size)
            IMG_SIZE="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH="$2"
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
        --wandb)
            USE_WANDB="true"
            shift
            ;;
        --no-wandb)
            USE_WANDB="false"
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-name)
            WANDB_NAME="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# ==================== Validation ====================

# Check if model config exists
if [ ! -f "$MODEL_CONFIG" ]; then
    echo "Error: Model config file not found: $MODEL_CONFIG"
    exit 1
fi

# Check if data config exists
if [ ! -f "$DATA_CONFIG" ]; then
    echo "Error: Data config file not found: $DATA_CONFIG"
    exit 1
fi

# Check if pretrained weights exist (if not using resume)
if [ -z "$RESUME" ] && [ ! -f "$PRETRAINED" ] && [ "$PRETRAINED" != "yolov8n.pt" ] && [ "$PRETRAINED" != "yolov8s.pt" ] && [ "$PRETRAINED" != "yolov8m.pt" ] && [ "$PRETRAINED" != "yolov8l.pt" ] && [ "$PRETRAINED" != "yolov8x.pt" ]; then
    echo "Warning: Pretrained weights file not found: $PRETRAINED"
    echo "YOLO will download it automatically if it's a standard model name."
fi

# ==================== Setup ====================

echo "=========================================="
echo "Improved YOLOv8 Training Script"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model config:    $MODEL_CONFIG"
echo "  Data config:     $DATA_CONFIG"
echo "  Pretrained:      $PRETRAINED"
echo "  Epochs:          $EPOCHS"
echo "  Image size:      $IMG_SIZE"
echo "  Batch size:      $BATCH"
echo "  Device:          $DEVICE"
echo "  Project:         $PROJECT"
echo "  Name:            $NAME"
echo "  WandB:           $USE_WANDB"
if [ "$USE_WANDB" = "true" ]; then
    echo "  WandB project:   $WANDB_PROJECT"
    echo "  WandB name:      $WANDB_NAME"
fi
if [ -n "$RESUME" ]; then
    echo "  Resume:          $RESUME"
fi
echo ""

# Import and register custom modules
echo "Step 1: Importing and registering custom modules..."
python -c "from improved_yolov8 import utils" || {
    echo "Error: Failed to import custom modules"
    exit 1
}
echo "✓ Custom modules registered successfully"
echo ""

# ==================== Training ====================

echo "Step 2: Starting training..."
echo ""

# Build YOLO command
YOLO_CMD="yolo train"

if [ -n "$RESUME" ]; then
    # Resume training
    YOLO_CMD="$YOLO_CMD resume=$RESUME"
else
    # New training
    YOLO_CMD="$YOLO_CMD model=$MODEL_CONFIG data=$DATA_CONFIG pretrained=$PRETRAINED"
fi

YOLO_CMD="$YOLO_CMD epochs=$EPOCHS imgsz=$IMG_SIZE batch=$BATCH device=$DEVICE"
YOLO_CMD="$YOLO_CMD project=$PROJECT name=$NAME"

# WandB configuration
if [ "$USE_WANDB" = "true" ]; then
    YOLO_CMD="$YOLO_CMD project=$WANDB_PROJECT name=$WANDB_NAME"
else
    YOLO_CMD="$YOLO_CMD wandb=False"
fi

# Add extra arguments
if [ -n "$EXTRA_ARGS" ]; then
    YOLO_CMD="$YOLO_CMD $EXTRA_ARGS"
fi

echo "Command: $YOLO_CMD"
echo ""

# Execute training
eval $YOLO_CMD

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Results saved to: $PROJECT/$NAME/"
echo "Best model: $PROJECT/$NAME/weights/best.pt"
echo "Last model: $PROJECT/$NAME/weights/last.pt"
if [ "$USE_WANDB" = "true" ]; then
    echo ""
    echo "View results on WandB: https://wandb.ai"
fi

