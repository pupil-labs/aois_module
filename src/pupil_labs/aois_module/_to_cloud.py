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
    index = index % COLORS.__len__()
    payload = {
        "color": COLORS[index],
        "created_at": datetime.utcnow().isoformat() + "Z",
        "description": "string",
        "enrichment_id": enrichment_id,
        "id": str(uuid.uuid4()),
        "mask_image_data_url": mask,  # AOI_Generator.encode_img(coloredMask),
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
    return r


def list_aois(url: str, api_key: str) -> dict:
    (_, workspace_id, project_id, enrichment_id) = (
        url.split('/')[2],
        url.split('/')[4],
        url.split('/')[6],
        url.split('/')[8],
    )
    response = requests.get(
        f"{API_URL}/workspaces/{workspace_id}/projects/{project_id}/enrichments/{enrichment_id}/aois",
        headers={"api-key": api_key},
    )
    if response.status_code == 200:
        aois = response.json().get('result', [])
        aoi_ids = [aoi['id'] for aoi in aois]
        return aoi_ids
    else:
        logging.error(response.text)
        return None


def delete_aois(
    url: str,
    aoi_ids: list,
    api_key: str,
) -> dict:
    (_, workspace_id, project_id, enrichment_id) = (
        url.split('/')[2],
        url.split('/')[4],
        url.split('/')[6],
        url.split('/')[8],
    )
    payload = {"aoi_ids": aoi_ids}
    response = requests.delete(
        f"{API_URL}/workspaces/{workspace_id}/projects/{project_id}/enrichments/{enrichment_id}/aois",
        json=payload,
        headers={"api-key": api_key},
    )
    if response.status_code == 200:
        logging.info("AOIs deleted successfully")
        return response.json()
    else:
        logging.error(f"Failed to delete AOIs: {response.text}")
        return response.json()
