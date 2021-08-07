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
sudo pip3 install setuptools
# Install lobe-python with TensorFlow Lite backend
sudo pip3 install lobe[tflite]