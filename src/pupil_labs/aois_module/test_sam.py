from pathlib import Path

import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def show_masks(masks, img):
    if len(masks) == 0:
        return

    sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    max_height, max_width = sorted_masks[0]["segmentation"].shape[:2]

    img_mask = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    img_mask_base = img_mask.copy()
    for ann in sorted_masks:
        m = ann["segmentation"]
        color_mask = np.random.rand(3) * 255
        img_mask[m] = color_mask

    cv2.addWeighted(img_mask_base, 0.7, img, 0.3, 0, img)
    cv2.imshow("Annotations", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


model_path = str(Path(__file__).parent / "assets" / "sam_vit_b_01ec64.pth")
sam = sam_model_registry["vit_b"](checkpoint=model_path)
sam.to(device="cpu")
mask_generator = SamAutomaticMaskGenerator(sam)

img = cv2.imread(str(Path(__file__).parent / "assets" / "reference_image.jpeg"))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

masks = mask_generator.generate(img)
print(masks)
show_masks(masks, img)
