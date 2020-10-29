# lobe-python
Code to run exported Lobe models in Python.

## Install
```
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

## Usage
```
from lobe import ImageModel

model = ImageModel.load('path/to/exported/model')

# OPTION 1: Predict from an image file
result = model.predict_from_file('path/to/file.jpg')

# OPTION 2: Predict from an image url
result = model.predict_from_url('http://path/to/file.jpg')

# OPTION 3: Predict from Pillow image
from PIL import Image
img = Image.open('path/to/file.jpg')
result = model.predict(img)

# Print top prediction
print(result.prediction)

# Print all classes
for label, prop in result.labels:
    print(f"{label}: {prop*100}%")

```

## Resources

If you're running this on a Pi and having issues, and seeing this error:

```bash
Could not install packages due to an EnvironmentError: 404 Client Error: Not Found for url:  https://pypi.org/simple/tflite-runtime/ 
```

running this may help:

```bash
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
```
