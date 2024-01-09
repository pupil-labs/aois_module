import base64
import logging
import uuid
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import requests

API_KEY = ""
WORKSPACE_ID = ""
PROJECT_ID = ""
ENRICHMENT_ID = ""

API_URL = "https://api.cloud.pupil-labs.dev/v2"
URL = f"{API_URL}/workspaces/{WORKSPACE_ID}/projects/{PROJECT_ID}/enrichments/{ENRICHMENT_ID}/aois"

COLORS = [
    "#39FF14",  # Neon Green
    "#FFFF00",  # Bright Yellow
    "#FFA500",  # Vivid Orange
    "#FF69B4",  # Hot Pink
    "#00BFFF",  # Sky Blue
    "#FF0000",  # Bright Red
    "#00FFFF",  # Cyan
    "#800080",  # Bright Purple
    "#32CD32",  # Lime Green
    "#FF00FF",  # Magenta
]


def encodeMask(img: np.array) -> str:
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")


def api_get(path: str) -> dict:
    url = f"{API_URL}/{path}"
    if (
        requests.get(
            url, headers={"api-key": API_KEY, "workspace_id": WORKSPACE_ID}
        ).json()["status"]
        == "success"
    ):
        return requests.get(
            url, headers={"api-key": API_KEY, "workspace_id": WORKSPACE_ID}
        ).json()["result"]
    else:
        error = requests.get(
            url, headers={"api-key": API_KEY, "workspace_id": WORKSPACE_ID}
        ).json()["message"]
        log.error(error)
        raise (Exception(error))


def post_aoi(
    img: np.array, label: str, bbox, index: int, url: Optional[str] = URL
) -> dict:
    h, w = img.shape
    index = index % COLORS.__len__()
    color = COLORS[index]
    color = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))[::-1]
    coloredMask = np.zeros((*img.shape, 4), dtype=np.uint8)
    coloredMask[img] = color + (255,)
    coloredMask[~img] = (0, 0, 0, 0)

    payload = {
        # "bounding_box": {
        #     "max_x": str(bbox[0] / w),
        #     "max_y": str(bbox[1] / h),
        #     "min_x": str(bbox[2] / w),
        #     "min_y": str(bbox[3] / h),
        # },
        # "centroid_xy": {
        #     "x": str(((bbox[0] / w) + (bbox[2] / w)) / 2),
        #     "y": str(((bbox[1] / h) + (bbox[3] / h)) / 2),
        # },
        "color": COLORS[index],
        "created_at": datetime.utcnow().isoformat() + "Z",
        "description": "string",
        "enrichment_id": ENRICHMENT_ID,
        "id": str(uuid.uuid4()),
        "mask_image_data_url": "data:image/png;base64," + encodeMask(coloredMask),
        "name": label,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    r = requests.post(
        url,
        json=payload,
        headers={"api-key": API_KEY, "workspace_id": WORKSPACE_ID},
    )
    if r.status_code != 201:
        logging.error(r._content)
    else:
        logging.info(f"AOI {label} set in Cloud")


def get_aois(url: Optional[str] = URL) -> dict:
    r = api_get(url)
    if r.status == 200:
        logging.info(r.result)
