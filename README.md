# Lobe Python API
Code to run exported Lobe models in Python using the TensorFlow or TensorFlow Lite options.

## Install
### Linux
```shell script
# Install Python3
sudo apt update
sudo apt install -y python3-dev python3-pip

# Install Pillow dependencies
sudo apt update
sudo apt install -y \
    libatlas-base-dev \
    libopenjp2-7 \
    libtiff5 \
    libjpeg62-turbo

# Install lobe-python
pip3 install setuptools git
pip3 install git+https://github.com/lobe/lobe-python
```

### Mac/Windows
Use a virtual environment with Python 3.7
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

If you're running this on a Pi and having issues, and seeing this error:

```
Could not install packages due to an EnvironmentError: 404 Client Error: Not Found for url:  https://pypi.org/simple/tflite-runtime/ 
```

running this may help:

```shell script
pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl
```
