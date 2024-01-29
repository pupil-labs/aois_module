import base64
import os
from pathlib import Path

import cv2
import numpy as np
import requests


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


# def toCloud(response, url, api_key):
#     for index, r in enumerate(response):
#         post_aoi(r['mask'], r['label'], [], index + 1, url, api_key)


if __name__ == "__main__":
    img = load_img()
    img = encode_img(img)

    data = {"image": img, "text": "bottle"}

    response = requests.post("http://0.0.0.0:8002/segment", json=data)
    # pred_names, boxes = AOI.predict_dino(img, "bottle")
    # aois = AOI.predict_sam(img, pred_names, boxes)
    # toCloud(
    #     response.content,
    #     url="https://cloud.pupil-labs.com/workspaces/d6bde22c-0c74-4d7d-8ab6-65b665c3cb4e/projects/56d10f4d-2899-4ddb-bef9-388c714dc812/enrichments/d3f52f82-c197-4a1a-ab78-29c96e2262b5",
    #     api_key="4bsfwpCc7fdVXn5uvK6WJdWzcxpaafdMfrbBqEX3KMAR",
    # )

    print(response.content)
