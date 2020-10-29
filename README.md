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

If you're running this on a Pi, check this out:

https://nam06.safelinks.protection.outlook.com/?url=https%3A%2F%2Fdl.google.com%2Fcoral%2Fpython%2Ftflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl&data=04%7C01%7Cscusack%40microsoft.com%7Ca764dead35ee4625983408d87c393104%7C72f988bf86f141af91ab2d7cd011db47%7C0%7C0%7C637395933563998953%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C1000&sdata=aiTithClUQWvoEusDpt4cNK4tLhLfrtOSnsiGov3GhU%3D&reserved=0
