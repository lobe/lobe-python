# Lobe Python API
Code to run exported Lobe models in Python using the TensorFlow, TensorFlow Lite, or ONNX options.

Works with Python 3.7, 3.8, 3.9, and 3.10 untested for other versions. [Note: 3.10 only works with the TensorFlow backend]

## Install
### Backend options with pip
You can install each of the backends on an individual basis, or all together through pip like so:
```shell
# For all of the supported backends (TensorFlow, TensorFlow Lite, ONNX)
pip install lobe[all]

# For TensorFlow only
pip install lobe[tf]

# For TensorFlow Lite only (note for Raspberry Pi see our setup script in scripts/lobe-rpi-install.sh)
pip install lobe[tflite]

# For ONNX only
pip install lobe[onnx]
```

Installing lobe-python without any options (`pip install lobe`) will only install the base requirements, no backends will be installed.
If you try to load a model with a backend that hasn't been installed, an error message will
show you the instructions to install the correct backend.

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
# Swap out the 'all' option here for your desired backend from 'backend options with pip' above.
pip3 install lobe[all]
```

For Raspberry Pi OS (Raspian) run:
```shell script
cd ~
wget https://raw.githubusercontent.com/lobe/lobe-python/master/scripts/lobe-rpi-install.sh
chmod 755 lobe-rpi-install.sh
sudo ./lobe-rpi-install.sh
```

### Mac/Windows
We recommend using a virtual environment:
```shell script
python3 -m venv .venv

# Mac:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```
Install the library
```shell script
# Make sure pip is up to date
python -m pip install --upgrade pip
# Swap out the 'all' option here for your desired backend from 'backend options with pip' above.
pip install lobe[all]
```

## Usage
```python
from lobe import ImageModel

model = ImageModel.load('path/to/exported/model/folder')

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

# Visualize the heatmap of the prediction on the image 
# this shows where the model was looking to make its prediction.
heatmap = model.visualize(img)
heatmap.show()
```
Note: model predict functions should be thread-safe. If you find bugs please file an issue.

## Resources

See the [Raspberry Pi Trash Classifier](https://github.com/microsoft/TrashClassifier) example, and its [Adafruit Tutorial](https://learn.adafruit.com/lobe-trash-classifier-machine-learning).
