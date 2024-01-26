import logging
import uuid
from datetime import datetime

import numpy as np
import requests

API_URL = "https://api.cloud.pupil-labs.com/v2"

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


def post_aoi(
    mask: np.array, label: str, bbox, index: int, url: str, api_key: str
) -> dict:
    (_, workspace_id, project_id, enrichment_id) = (
        f"https://{url.split('/')[2]}",
        url.split('/')[4],
        url.split('/')[6],
        url.split('/')[8],
    )
    # h, w = mask.shape
    # index = index % COLORS.__len__()
    # color = COLORS[index]
    # color = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))[::-1]
    # coloredMask = np.zeros((*mask.shape, 4), dtype=np.uint8)
    # coloredMask[mask] = color + (255,)
    # coloredMask[~mask] = (0, 0, 0, 0)

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
        "enrichment_id": enrichment_id,
        "id": str(uuid.uuid4()),
        "mask_image_data_url": "data:image/png;base64,"
        + mask,  # AOI_Generator.encode_img(coloredMask),
        "name": label,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    r = requests.post(
        f"{API_URL}/workspaces/{workspace_id}/projects/{project_id}/enrichments/{enrichment_id}/aois",
        json=payload,
        headers={"api-key": api_key, "workspace_id": workspace_id},
    )
    if r.status_code != 201:
        logging.error(r._content)
    else:
        logging.info(f"AOI {label} set in Cloud")
