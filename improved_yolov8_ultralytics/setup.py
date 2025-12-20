# Setup file for Improved YOLOv8s Ultralytics Integration

from setuptools import setup, find_packages

setup(
    name="improved-yolov8-ultralytics",
    version="1.0.0",
    description="Improved YOLOv8s with Ultralytics CLI integration",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy>=1.19.0",
        "einops>=0.3.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "PyYAML>=5.4.0",
        "tqdm>=4.60.0",
        "wandb>=0.15.0",
    ],
    entry_points={
        "console_scripts": [
            "yolo=ultralytics.cfg:entrypoint",
        ],
    },
    python_requires=">=3.8",
)

