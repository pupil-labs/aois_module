import base64
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

CKPT_REPO_ID = "ShilongLiu/GroundingDINO"
CKPT_FILENAME = "groundingdino_swinb_cogcoor.pth"
CKPT_CONFIG_FILENAME = "GroundingDINO_SwinB.cfg.py"

SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_t": "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt",
}
SAM_MODEL = "vit_h"
CACHE_PATH = os.environ.get(
    "TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints")
)


def load_model_hf(repo_id, filename, ckpt_config_filename, device="cpu"):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model.to(device)


class AOI_Generator:
    def __init__(self):
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # else "mps" if torch.has_mps
        self.dino_model = load_model_hf(
            CKPT_REPO_ID, CKPT_FILENAME, CKPT_CONFIG_FILENAME, device=self.device
        )
        self.model_cache_path = os.path.join(
            CACHE_PATH, SAM_MODELS[SAM_MODEL].split("/")[-1]
        )
        if not Path(self.model_cache_path).exists():
            self.sam = sam_model_registry[SAM_MODEL]()
            self.sam.load_state_dict(
                torch.hub.load_state_dict_from_url(SAM_MODELS[SAM_MODEL]), strict=True
            )
        else:
            self.sam = sam_model_registry[SAM_MODEL](checkpoint=self.model_cache_path)
        self.sam.to(device=self.device)
        self.sam_full = SamAutomaticMaskGenerator(
            self.sam,
            pred_iou_thresh=0.88,
            min_mask_region_area=5000,
            crop_overlap_ratio=0.1,
            output_mode="binary_mask",
        )
        self.sam_predictor = SamPredictor(self.sam)
        self.transform = torchvision.transforms.ToTensor()

    def scale_img(self, img: np.array, max_size: Optional[int] = 1024) -> np.array:
        height, width = img.shape[:2]
        if max(height, width) > max_size:
            sf = max_size / height if height > width else max_size / width
            img = cv2.resize(
                img,
                dsize=None,
                fx=sf,
                fy=sf,
            ).astype(np.uint8)
            logging.info(f"Image scaled down by: {1 / sf}")
        return img

    def decode_img(self, img_str: str) -> np.array:
        img_bytes = base64.b64decode(img_str)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return img

    def encode_img(self, img: np.array, extension: Optional[str] = ".png") -> str:
        _, buffer = cv2.imencode(extension, img)
        return base64.b64encode(buffer).decode("utf-8")

    def predict_dino(
        self,
        image: np.array,
        text_input: str,
        box_threshold: Optional[float] = 0.3,
        text_threshold: Optional[float] = 0.25,
    ):
        caption = text_input.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."

        transformed_image = self.transform(image)
        scaled_image = transformed_image.to(self.device)

        boxes, logits, pred_phrases = predict(
            device=self.device,
            model=self.dino_model.to(self.device),
            image=scaled_image,
            caption=caption,
            box_threshold=0.2,
            text_threshold=0.25,
        )
        pred_phrases = [f"{phrase}_{i}" for i, phrase in enumerate(pred_phrases)]

        self.dino_labels = pred_phrases
        self.dino_boxes = boxes
        return pred_phrases, boxes

    def predict_sam(
        self,
        image: np.array,
        pred_phrases: str,
        boxes: np.array,
    ):
        if boxes is None:
            masks = self.sam_full.generate(image)
            logging.info(f"A total of {len(masks)} AOIs generated with SAM")
            if len(masks) == 0:
                raise ValueError("No AOIs were generated with SAM")
            else:
                masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
                aois = pd.DataFrame(masks)
        else:
            H, W = image.shape[:2]
            self.sam_predictor.set_image(image)
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([
                W,
                H,
                W,
                H,
            ])
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
                boxes_xyxy, image.shape[:2]
            ).to(self.device)

            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            masks = masks.detach().cpu().numpy()
            masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3])

            def create_masks(mask, H: int, W: int) -> np.array:
                import random

                def generate_random_color_hex():
                    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

                color = generate_random_color_hex()
                color = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))[::-1]
                coloredMask = np.zeros((*mask.shape, 4), dtype=np.uint8)
                coloredMask[mask] = color + (255,)
                coloredMask[~mask] = (0, 0, 0, 0)
                return coloredMask

            aois = pd.DataFrame([
                {
                    "segmentation": mask,
                    "label": pred_phrases[i],
                    "bbox": boxes_xyxy[i].tolist(),
                    "mask": "data:image/png;base64,"
                    + self.encode_img(create_masks(mask, H, W)),
                }
                for i, mask in enumerate(masks)
            ])
            color_masks = [create_masks(mask, H, W) for i, mask in enumerate(masks)]
        return aois, color_masks

    def paint_image(self, image: np.array, color_masks: np.array) -> np.array:
        if image.shape[2] == 3:
            image = np.concatenate(
                [image, np.full((*image.shape[:2], 1), 255, dtype=np.uint8)], axis=-1
            )

        for mask in color_masks:
            alpha = mask[:, :, 3][:, :, np.newaxis] / 255.0
            image = (1 - alpha) * image + alpha * mask

        return image.astype(np.uint8)
