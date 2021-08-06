#!/bin/bash
#
# Install Python3
sudo apt -y update
sudo apt install -y \
    python3-dev \
    python3-pip \
    libatlas-base-dev \
    libopenjp2-7 \
    libtiff5 \
    libjpeg62-turbo
sudo apt-get install -y git
# Install lobe-python with TensorFlow Lite backend (ONNX backend could also work for Raspberry Pi)
sudo pip3 install setuptools
sudo pip3 install lobe[tflite]