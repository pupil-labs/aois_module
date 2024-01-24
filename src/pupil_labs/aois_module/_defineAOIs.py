import io
import logging
import os
import struct
from timeit import default_timer as timer  # For timing the code
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
import cv2
import matplotlib as mpl  # For plotting
import matplotlib.pyplot as plt
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import torch

from matplotlib import patches
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from rich.logging import RichHandler  # For logging

# For parsing the arguments
from pupil_labs.aois_module._helpers import draw_box, draw_mask, scale_img
from pupil_labs.aois_module._models import (build_dino, build_sam_model, predict_dino,
                                            predict_sam)
from pupil_labs.aois_module._parser import DataClasses, MetricClasses, init_parser
from pupil_labs.aois_module._to_cloud import post_aoi

verbit = struct.calcsize("P") * 8
if verbit != 64:
    error = "Sorry, this script only works on 64 bit systems!"
    raise Exception(error)


logging.getLogger("defineAOIs")
logging.basicConfig(
    format="%(message)s",
    datefmt="[%X]",
    level=logging.INFO,
    handlers=[RichHandler()],
)

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(
        "[white bold on #0d122a]◎ Define AIOs in RIM by Pupil Labs[/]",
        extra={"markup": True},
    )
    logging.info(
        "[white bold on #4267B2]∞ Powered by Meta AI open source libraries[/]",
        extra={"markup": True},
    )
    yield
    logging.info("Sam went to sleep!")

class defineAOIs:
    def __init__(self):
        

    def run(self):
        args = init_parser()
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))
        logging.info(self.__dict__)
        self.sanity_checks()

        self.load_aois()

        self.reference_image_bgr = cv2.imread(
            os.path.join(self.input_path, "reference_image.jpeg")
        )
        self.reference_image = cv2.cvtColor(self.reference_image_bgr, cv2.COLOR_BGR2RGB)
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # else "mps" if torch.has_mps

        self = scale_img(self)
        if self.aois is None and self.is_sam:
            start = timer()
            self.get_sam_aois()
            logging.info(
                f"Time to load the model/generate AOIs with SAM: {timer()-start} s"
            )
        elif self.aois is None and not self.is_sam:
            self.get_sq_aois()

        if self.upload2cloud:
            self.toCloud()

        # self.load_files()
        self.report_aois()
        self.figure = {}
        ax = self.plot_color_patches(
            values=pd.Series(self.aois.index), ax=plt.gca(), save=True
        )

        # self.in_aoi()

        # if self.metric in [MetricClasses.hit_rate, MetricClasses.all]:
        #     self.compute_hit_rate()
        # if self.metric in [MetricClasses.first_contact, MetricClasses.all]:
        #     self.compute_first_contact()
        # if self.metric in [MetricClasses.dwell_time, MetricClasses.all]:
        #     self.compute_dwell_time()

        # self.save_aois()
        # self.create_report()
        logging.info("Done")

    def sanity_checks(self):
        if isinstance(self.metric, str):
            self.metric = MetricClasses[self.metric]
            logging.info("metric: %s", self.metric)
        if isinstance(self.type, str):
            self.type = DataClasses[self.type]
            logging.info("type: %s", self.type)

        if self.input_path is None or not os.path.exists(self.input_path):
            self.input_path = get_path()
        if self.output_path is None or not os.path.exists(self.output_path):
            from datetime import datetime

            self.output_path = os.path.join(
                self.input_path,
                "output_aois",
                datetime.today().strftime("%Y-%m-%d"),
            )
            counter = 0
            new_output_path = self.output_path
            while os.path.exists(new_output_path):
                counter += 1
                new_output_path = f"{new_output_path}_{counter}"
            self.output_path = new_output_path
            os.makedirs(self.output_path, exist_ok=True)
        logging.info("Input path: %s", self.input_path)

    def load_models(self):
        if self.is_sam:
            self = build_sam_model(self)
        if self.dino_text_input is not None:
            self = build_dino(self)

    def get_sam_aois(self):
        self.load_models()
        if self.dino_text_input is not None:
            self = predict_dino(self)
        self = predict_sam(self)

    def load_aois(self):
        if self.aois_path is not None:
            self.aois = pd.read_pickle(self.aois_path)
            self.is_sam = True if "segmentation" in self.aois.columns else False
        else:
            self.aois = None
            filenames = ["aoi_ids.pkl", "aoi_ids_sam.pkl"]
            for filename in filenames:
                if os.path.exists(os.path.join(self.output_path, filename)):
                    logging.info("AOIs already defined on that folder, reusing them")
                    self.aois_path = os.path.join(self.output_path, filename)
                    self.aois = pd.read_pickle(self.aois_path)
                    self.is_sam = True if "segmentation" in self.aois.columns else False
        if isinstance(self.aois, pd.DataFrame):
            self.scaling_factor = self.aois.scaling_factor[0]
            logging.info(self.aois.columns)

    def save_aois(self):
        self.aois["scaling_factor"] = self.scaling_factor
        if self.is_sam:
            self.aois.to_pickle(self.output_path + "/aoi_ids_sam.pkl")
        else:
            self.aois.to_pickle(self.output_path + "/aoi_ids.pkl")

    def report_aois(self):
        logging.info("Areas of interest:")
        logging.info(self.aois)

    def plot_color_patches(
        self,
        values,
        ax,
        alpha=0.3,
        colorbar=False,
        data=None,
        unit_label="",
        save=False,
    ):
        ref_image = self.scaled_image.copy()
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGBA)
        ax.imshow(ref_image)
        # normalize patch values
        values_normed = values.astype(np.float32)
        values_normed -= values_normed.min()
        values_normed /= values_normed.max()

        colors = mpl.colormaps["gnuplot"]

        if not self.is_sam:
            for aoi_id, aoi_val in values_normed.items():
                aoi_id = int(aoi_id)
                aoi = [
                    self.aois.x[aoi_id],
                    self.aois.y[aoi_id],
                    self.aois.width[aoi_id],
                    self.aois.height[aoi_id],
                ]
                ax.add_patch(
                    patches.Rectangle(
                        aoi,
                        *aoi[2:],
                        alpha=alpha,
                        facecolor=colors(aoi_val),
                        edgecolor=colors(aoi_val),
                        linewidth=5,
                    )
                )
                ax.text(aoi[0] + 20, aoi[1] + 120, f"{aoi_id}", color="black")
        else:
            ref_image = Image.fromarray(ref_image)
            draw = ImageDraw.Draw(ref_image)
            draw_mask(self, self.aois.segmentation, draw, colors)
            if self.dino_text_input is not None:
                draw_box(self.aois.bbox, self.aois.label, draw, colors)

        ax.imshow(ref_image)
        ax.axis("off")

        if colorbar:
            norm = mpl.colors.Normalize(vmin=0, vmax=values.max())
            cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colors), ax=ax)
            cb.set_label(unit_label)

        if data is not None and self.scatter:
            if self.type == DataClasses.fixations:
                field0 = "fixation detected in reference image"
                field1 = "fixation x [px]"
                field2 = "fixation y [px]"
            elif self.type == DataClasses.gaze:
                field0 = "gaze detected in reference image"
                field1 = "gaze position in reference image x [px]"
                field2 = "gaze position in reference image y [px]"
            data_in = data[data[field0] is True]
            ax.scatter(data_in[field1], data_in[field2], s=20, color="red", alpha=0.8)

        plt.draw()
        if save:
            ax.title.set_text("Areas of Interest")
            buf = io.BytesIO()
            plt.savefig(f"{self.output_path}/AOIs.png")
            plt.savefig(buf, format="png")
            buf.seek(0)
            self.figure["AOIs"] = buf.getvalue()
            plt.close()
        return ax

    def toCloud(self):
        index = 0
        for index, aoi in self.aois.iterrows():
            index += 1
            post_aoi(aoi.segmentation, aoi.label, aoi.bbox, index)


def main():
    nest_asyncio.apply()
    uvicorn.run(
        app,
        port=8001,
    )


if __name__ == "__main__":
    main()
