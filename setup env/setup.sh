#!/bin/bash

# Update package list
sudo apt update
sudo apt install software-properties-common

# Add deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.9 and required packages
sudo apt install python3.9 python3.9-dev python3.9-venv

# Create virtual environment
python3.9 -m venv yolov8_env

# Activate virtual environment
source yolov8_env/bin/activate

# Install requirements
pip install -r requirements.txt