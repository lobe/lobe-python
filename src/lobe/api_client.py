#!/usr/bin/env python
import json

import requests
from PIL import Image

from .api_constants import IMAGE_INPUT
from .image_utils import image_to_base64
from .results import ClassificationResult


def send_image_predict_request(image: Image.Image, predict_url: str) -> ClassificationResult:
    payload = {
        IMAGE_INPUT: image_to_base64(image)
    }
    response = requests.post(predict_url, json=payload)
    response.raise_for_status()
    return ClassificationResult(json.loads(response.text))
