# Lobe Python API
Code to run exported Lobe models in Python using the TensorFlow, TensorFlow Lite, or ONNX options.

## Install
### Linux
Before running these commands, make sure that you have [git](https://git-scm.com/download/linux) installed.

```shell script
# Install Python3
sudo apt update
sudo apt install -y python3-dev python3-pip

# Install Pillow dependencies
sudo apt update
sudo apt install -y libatlas-base-dev libopenjp2-7 libtiff5 libjpeg62-dev

# Install lobe-python
pip3 install setuptools
pip3 install git+https://github.com/lobe/lobe-python --no-cache-dir
```

For Raspberry Pi OS (Raspian) run:
```shell script
cd ~
wget https://raw.githubusercontent.com/lobe/lobe-python/master/scripts/lobe-rpi-install.sh
sudo ./lobe-rpi-install.sh
```

### Mac/Windows
Use a virtual environment with Python 3.7 or 3.8
```shell script
python3 -m venv .venv

# Mac:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```
Install the library
```shell script
# make sure pip is up to date
python -m pip install --upgrade pip
# install
pip install git+https://github.com/lobe/lobe-python
```

## Usage
```python
from lobe import ImageModel

model = ImageModel.load('path/to/exported/model')

# OPTION 1: Predict from an image file
result = model.predict_from_file('path/to/file.jpg')

# OPTION 2: Predict from an image url
result = model.predict_from_url('http://url/to/file.jpg')

# OPTION 3: Predict from Pillow image
from PIL import Image
img = Image.open('path/to/file.jpg')
result = model.predict(img)

# Print top prediction
print(result.prediction)

# Print all classes
for label, confidence in result.labels:
    print(f"{label}: {confidence*100}%")

```
Note: model predict functions should be thread-safe. If you find bugs please file an issue.

## Resources

See the [Raspberry Pi Trash Classifier](https://github.com/microsoft/TrashClassifier) example, and its [Adafruit Tutorial](https://learn.adafruit.com/lobe-trash-classifier-machine-learning).
