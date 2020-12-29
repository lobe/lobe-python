#!/usr/bin/env python
from lobe import ImageModel

model = ImageModel.load('path/to/exported/model')

# Predict from an image file
result = model.predict_from_file('path/to/file.jpg')

# Predict from an image url
result = model.predict_from_url('http://url/to/file.jpg')

# Predict from Pillow image
from PIL import Image
img = Image.open('path/to/file.jpg')
result = model.predict(img)

# Print top prediction
print("Top prediction:", result.prediction)

# Print all classes
for label, confidence in result.labels:
    print(f"{label}: {confidence*100:.6f}%")
