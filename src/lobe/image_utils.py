from io import BytesIO
from PIL import Image
import numpy as np
from typing import Tuple
import requests
import base64


def crop_center(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    crop_width, crop_height = size
    width, height = image.size
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    right = min(width, left + crop_width)
    bottom = min(height, top + crop_height)
    return image.crop((left, top, right, bottom))


def crop_center_square(image: Image.Image, size: Tuple[int, int]=None) -> Image.Image:
    if not size:
        width, height = image.size
        size = min(width, height)
    return crop_center(image, (size, size))


def resize_uniform_to_fill(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    width, height = image.size
    min_w, min_h = size

    # Pick the bigger scale factor, to ensure both dimensions are completely filled
    scale = max(min_w / width, min_h / height)
    new_size = (round(scale * width), round(scale * height))
    return image.resize(new_size)


def resize_uniform_to_fit(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    width, height = image.size
    max_w, max_h = size

    # Pick the smaller scale factor, to ensure the entire image fits within the bounds
    scale = min(max_w / width, max_h / height)
    new_size = (round(scale * width), round(scale * height))
    return image.resize(new_size)


def update_orientation(image: Image.Image) -> Image.Image:
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if exif != None and exif_orientation_tag in exif:
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def ensure_rgb_format(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    ensure_rgb_format(image).save(buffer,format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return image


def get_image_from_file(path: str) -> Image.Image:
    return Image.open(path)


def preprocess_image(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    image_processed = update_orientation(image)

    # resize and crop image to the model's required size
    image_processed = ensure_rgb_format(image_processed)
    image_processed = resize_uniform_to_fill(image_processed, size)
    image_processed = crop_center(image_processed, size)
    return image_processed


def image_to_array(image: Image.Image) -> np.ndarray:
    # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
    image = np.asarray(image) / 255.0
    # pad with an extra batch dimension
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image


def array_to_image(image: np.ndarray) -> Image.Image:
    # make the input array 0-255 int
    image = np.uint8(image * 255)
    # squeeze extra batch dimension if it exists
    if image.shape[0] == 1:
        image = np.squeeze(image, axis=0)
    return Image.fromarray(image)
