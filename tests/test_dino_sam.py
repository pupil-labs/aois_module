import base64
import os
from pathlib import Path

import cv2
import numpy as np
import requests

from pupil_labs.aois_module._to_cloud import post_aoi


def load_img():
    img = cv2.imread(
        os.path.join(Path(__file__).parent / "assets" / "reference_image.jpeg")
    )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_img(img: np.array, extension=".png") -> str:
    _, buffer = cv2.imencode(extension, img)
    return base64.b64encode(buffer).decode("utf-8")


def decode_img(img_str: str) -> np.array:
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img


def toCloud(response, url, api_key):
    index = 0
    for index, r in response:
        index += 1
        post_aoi(r.mask, r.label, [], index, url, api_key)


if __name__ == "__main__":
    img = load_img()
    img = encode_img(img)

    data = {"image": img, "text": "grass"}

    response = requests.post("http://127.0.0.1:8000/segment", json=data)
    # pred_names, boxes = AOI.predict_dino(img, "grass")
    # aois = AOI.predict_sam(img, pred_names, boxes)
    # toCloud(response, url="", api_key="4bsfwpCc7fdVXn5uvK6WJdWzcxpaafdMfrbBqEX3KMAR")

    print(response.content)
