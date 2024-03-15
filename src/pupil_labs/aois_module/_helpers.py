import logging
import os

import cv2
import numpy as np
from PIL import ImageFont
from pycocotools import mask as mask_utils


def scale_img(self, save=False):
    if self.scaling_factor is None:
        max_size = 1024
        height, width = self.reference_image.shape[:2]
        if height > width:
            self.scaling_factor = max_size / height
        else:
            self.scaling_factor = max_size / width
    self.scaled_image = cv2.resize(
        self.reference_image_bgr.copy(),
        dsize=None,
        fx=self.scaling_factor,
        fy=self.scaling_factor,
    )
    self.scaled_image = self.scaled_image.astype(np.uint8)
    logging.info(f"Image scaled down by: {1/self.scaling_factor}")
    if save:
        cv2.imwrite(
            os.path.join(self.output_path, "scaled_image.jpeg"), self.scaled_image
        )
    return self


def draw_mask(self, masks, draw, colors):
    for mask_id, mask in enumerate(masks):
        if self.mask_output == "coco_rle":
            mask = mask_utils.decode(mask)
        color = colors(mask_id)
        color = tuple([int(255 * c) for c in color])
        nonzero_coords = np.transpose(np.nonzero(mask))

        for coord in nonzero_coords:
            draw.point(coord[::-1], fill=color)


def draw_box(boxes, labels, draw, colors):
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        color = colors(idx)
        color = tuple([int(255 * c) for c in color])
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=2)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((box[0], box[1]), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (box[0], box[1], w + box[0], box[1] + h)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1]), str(label), fill="white")
        draw.text((box[0], box[1]), label)
