#!/bin/bash
# Add TensorFlow Lite package repo (https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python)
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
# Install Python3 and deps
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
# Install TensorFlow Lite Runtime
sudo apt install -y python3-tflite-runtime
# Install lobe-python which can use the tflite backend
sudo pip3 install lobe