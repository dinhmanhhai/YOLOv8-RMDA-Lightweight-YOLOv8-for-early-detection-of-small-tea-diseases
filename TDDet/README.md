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
yolo train model=ultralytics/cfg/models/v8/TDDet.yaml data=ultralytics/dataset/chayev11/data.yaml device=0 cache=False imgsz=640 epochs=150 batch=16 close_mosaic=10 workers=1 optimizer=SGD patience=30 project=runs/train name=exp
```

#### testing

```bash
yolo val model=runs/train/exp/weights/best.pt data=ultralytics/dataset/chayev11/data.yaml split=test imgsz=640 batch=16 project=runs/val name=exp
```
