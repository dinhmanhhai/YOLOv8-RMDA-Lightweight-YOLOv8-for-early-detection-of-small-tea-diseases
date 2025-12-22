# TDDet

TDDet is an lightweight and efficient tea disease detector for quickly and accurately detecting tea diseases.

## Framework

![](./TDDet.png)

*Figure 3: The framework of TDDet.*

## dataset

download the dataset: [link](https://pan.baidu.com/s/1cACKNPdyohigHbc8gRZ7ng?pwd=4d02) 

## Configs

#### requires

Python 3.9

CUDA 12.2

PyTorch 1.8

#### yolo command

```bash
pip install ultralytics
```

#### training

```bash
# Chạy từ folder TDDet
# Sử dụng wrapper script để đảm bảo TDDet modules được import đúng cách
python yolo_with_tddet.py train model=codes/cfg/models/v8/yolov8-mobilenetv4.yaml data=../dataset/data.yaml device=0 cache=False imgsz=640 epochs=150 batch=16 close_mosaic=10 workers=1 optimizer=SGD patience=30 project=runs/train name=exp

# Hoặc nếu có TDDet.yaml ở root folder
python yolo_with_tddet.py train model=TDDet.yaml data=../dataset/data.yaml device=0 cache=False imgsz=640 epochs=150 batch=16 close_mosaic=10 workers=1 optimizer=SGD patience=30 project=runs/train name=exp
```

#### testing

```bash
# Chạy từ folder TDDet
python yolo_with_tddet.py val model=runs/train/exp/weights/best.pt data=../dataset/data.yaml split=test imgsz=640 batch=16 project=runs/val name=exp
```