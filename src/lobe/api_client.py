#!/usr/bin/env python
import requests
from PIL import Image
from image_utils import image_to_base64
from ._results import PredictionResult

def send_image_predict_request(image: Image.Image, predict_url: str, key: str) -> PredictionResult:
    payload = {
        "inputs": {"Image": image_to_base64(image)},
        "key": key
    }
    headers = {'authorization': f'Bearer {key}'}
    response = requests.post(predict_url, json=payload, headers=headers)
    response.raise_for_status()
    return PredictionResult.from_json(response.text)
