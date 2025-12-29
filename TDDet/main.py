from ultralytics import YOLO
import wandb

# Optional: Login to wandb explicitly in code if not done via CLI
# wandb.login(key="YOUR_WANDB_API_KEY") 

# Load model custom (MobileNetV4-MQA version)
model = YOLO("ultralytics/cfg/models/v8/yolov8-mobilenetv4-mqa.yaml") 

# Train with all configurable parameters
model.train(
    data="data.yaml",           # path to dataset YAML
    epochs=150,                 # number of epochs to train for
    patience=50,                # epochs to wait for no observable improvement for early stopping of training
    batch=10,                    # number of images per batch (-1 for AutoBatch)
    imgsz=640,                  # size of input images as integer
    save=True,                  # save train checkpoints and predict results
    save_period=-1,             # Save checkpoint every x epochs (disabled if < 1)
    cache=False,                # True/ram, disk or False. Use cache for data loading
    device=0,                   # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
    workers=8,                  # number of worker threads for data loading (per RANK if DDP)
    project="works-haidinh-ptit",# [WANDB] Project name
    name="tddet_mqa",             # [WANDB] Run name
    exist_ok=False,             # whether to overwrite existing experiment
    pretrained=True,            # (bool or str) whether to use a pretrained model (bool) or a model to load weights from (str)
    optimizer="auto",           # optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
    verbose=True,               # whether to print verbose output
    seed=0,                     # random seed for reproducibility
    deterministic=True,         # whether to enable deterministic mode
    single_cls=False,           # train multi-class data as single-class
    rect=False,                 # rectangular training if mode='train' or rectangular validation if mode='val'
    cos_lr=False,               # use cosine learning rate scheduler
    close_mosaic=10,            # (int) disable mosaic augmentation for final epochs (0 to disable)
    resume=False,               # resume training from last checkpoint
    amp=True,                   # Automatic Mixed Precision (AMP) training, choices=[True, False]
    fraction=1.0,               # dataset fraction to train on (default is 1.0, all images in train set)
    profile=False,              # profile ONNX and TensorRT speeds during training for loggers
    freeze=None,                # (int or list, optional) freeze first n layers, or freeze list of layer indices
    overlap_mask=True,          # masks should overlap during training (segment train only)
    mask_ratio=4,               # mask downsample ratio (segment train only)
    dropout=0.0,                # use dropout regularization (classify train only)
    val=True,                   # validate/test during training
    split="val",                # dataset split to use for validation, i.e. 'val', 'test' or 'train'
    save_json=True,            # save results to JSON file
    save_hybrid=False,          # save hybrid version of labels (labels + additional predictions)
    conf=None,                  # object confidence threshold for detection (default 0.25 prediction, 0.001 val)
    iou=0.7,                    # intersection over union (IoU) threshold for NMS
    max_det=300,                # maximum number of detections per image
    half=False,                 # use half precision (FP16)
    dnn=False,                  # use OpenCV DNN for ONNX inference
    plots=True,                 # save plots and images during train/val
)