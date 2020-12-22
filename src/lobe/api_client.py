#!/usr/bin/env python
import requests
from PIL import Image

from .image_utils import image_to_base64
from .results import ClassificationResult


def send_image_predict_request(image: Image.Image, predict_url: str) -> ClassificationResult:
    payload = {
        "inputs": {"Image": image_to_base64(image)},
    }
    response = requests.post(predict_url, json=payload)
    response.raise_for_status()
    return ClassificationResult.from_json(response.text)
