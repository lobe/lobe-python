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
# Install lobe-python
sudo pip3 install setuptools
sudo pip3 install git+https://github.com/lobe/lobe-python --no-cache-dir