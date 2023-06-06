import os, cv2
import logging
import torch
import torchvision
from torchvision.ops import box_convert
from pathlib import Path
import pandas as pd

from timeit import default_timer as timer  # For timing the code
from PIL import Image, ImageDraw
from pupil_labs.aois_module._helpers import draw_mask, draw_box

from huggingface_hub import hf_hub_download

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict

from segment_anything import (
    SamAutomaticMaskGenerator,
    sam_model_registry,
    build_sam,
    SamPredictor,
)

SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

CACHE_PATH = os.environ.get(
    "TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints")
)


def build_dino(self):
    if self.dino_text_input is not None and self.is_sam:
        logging.info(f"Looking for {self.dino_text_input}")
    elif self.dino_text_input is None and self.is_sam:
        logging.info("No dino text input, segmenting the whole image")
        return
    else:
        logging.info("Dino text input is only available for SAM AOIs")
        return

    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    def load_model_hf(repo_id, filename, ckpt_config_filename, device="cpu"):
        cache_config_file = hf_hub_download(
            repo_id=repo_id, filename=ckpt_config_filename
        )

        args = SLConfig.fromfile(cache_config_file)
        model = build_model(args)
        args.device = device

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location=device)
        log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        return model.to(device)

    self.dino_model = load_model_hf(
        ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device=self.device
    )
    return self


def build_sam_model(self):
    checkpoint_url = SAM_MODELS[self.sam_model]
    filename = checkpoint_url.split("/")[-1]
    self.model_path = os.path.join(CACHE_PATH, filename)

    if not Path(self.model_path).exists():
        self.sam = sam_model_registry[self.sam_model]()
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
        self.sam.load_state_dict(state_dict, strict=True)
    else:
        self.sam = sam_model_registry[self.sam_model](checkpoint=self.model_path)

    self.sam.to(device=self.device)
    if self.dino_text_input is None:
        self.mg = SamAutomaticMaskGenerator(
            self.sam,
            pred_iou_thresh=0.88,
            min_mask_region_area=5000,
            crop_overlap_ratio=0.1,
            output_mode=self.mask_output,
        )
    else:
        self.mg = SamPredictor(self.sam)
    return self


def predict_dino(self, box_threshold=0.3, text_threshold=0.25, debug=False):
    caption = self.dino_text_input.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    dino_model = self.dino_model.to(self.device)
    transform = torchvision.transforms.ToTensor()
    transformed_image = transform(self.scaled_image)
    scaled_image = transformed_image.to(self.device)

    boxes, logits, pred_phrases = predict(
        device=self.device,
        model=dino_model,
        image=scaled_image,
        caption=self.dino_text_input,
        box_threshold=0.2,
        text_threshold=0.25,
    )
    pred_phrases = [f"{phrase}_{i}" for i, phrase in enumerate(pred_phrases)]

    if debug:
        annotated_frame = annotate(
            image_source=self.scaled_image,
            boxes=boxes,
            logits=logits,
            phrases=pred_phrases,
        )
        annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
        cv2.imshow("DINO", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    self.dino_labels = pred_phrases
    self.dino_boxes = boxes
    return self


def predict_sam(self, debug=False):
    if self.dino_text_input is None:
        start_masking = timer()
        self.masks = self.mg.generate(self.scaled_image)
        logging.info(
            f"Time to generate AOIs on the image with SAM: {timer()-start_masking} s"
        )
        # remove masks that are too small
        logging.info(f"A total of {len(self.masks)} AOIs generated with SAM")
        if len(self.masks) == 0:
            raise ValueError("No AOIs were generated with SAM")
        else:
            self.masks = sorted(self.masks, key=(lambda x: x["area"]), reverse=True)
            self.aois = pd.DataFrame(self.masks)
    else:
        H, W = self.scaled_image.shape[:2]
        self.mg.set_image(self.scaled_image)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(self.dino_boxes) * torch.Tensor(
            [W, H, W, H]
        )
        transformed_boxes = self.mg.transform.apply_boxes_torch(
            boxes_xyxy, self.scaled_image.shape[:2]
        )

        masks, _, _ = self.mg.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks = masks.detach().cpu().numpy()
        self.masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3])
        self.aois = pd.DataFrame(
            [
                {
                    "segmentation": mask,
                    "label": self.dino_labels[i],
                    "bbox": boxes_xyxy[i].tolist(),
                }
                for i, mask in enumerate(self.masks)
            ]
        )
        if debug:
            img = Image.fromarray(self.scaled_image)
            draw = ImageDraw.Draw(img)
            colors = lambda x: (x * 0.001, x * 0.002, x * 0.003)
            draw_mask(self.masks, draw, colors)
            draw_box(boxes_xyxy, self.dino_labels, draw, colors)
            img = img.show()

    return self
