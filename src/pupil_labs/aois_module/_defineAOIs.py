import os, logging, cv2, io, struct, torch
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation

import matplotlib as mpl  # For plotting
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
from PIL import Image, ImageDraw

from timeit import default_timer as timer  # For timing the code

from pupil_labs.aois_module._parser import (
    init_parser,
    MetricClasses,
    DataClasses,
)  # For parsing the arguments
from pupil_labs.aois_module._helpers import (
    get_path,
    confirm_read,
    scale_img,
    draw_box,
    draw_mask,
)  # For selecting the input folder

from pupil_labs.aois_module._report import generate_report
from pupil_labs.aois_module._models import (
    build_dino,
    build_sam_model,
    predict_sam,
    predict_dino,
)

from rich.logging import RichHandler  # For logging

from pycocotools import mask as mask_utils

verbit = struct.calcsize("P") * 8
if verbit != 64:
    error = "Sorry, this script only works on 64 bit systems!"
    raise Exception(error)

sns.set_context("paper")
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.3)


logging.getLogger("defineAOIs")
logging.basicConfig(
    format="%(message)s",
    datefmt="[%X]",
    level=logging.INFO,
    handlers=[RichHandler()],
)


class defineAOIs:
    def __init__(self):
        logging.info(
            "[white bold on #0d122a]◎ Define AIOs in RIM by Pupil Labs[/]",
            extra={"markup": True},
        )
        logging.info(
            "[white bold on #4267B2]∞ Powered by Meta AI open source libraries[/]",
            extra={"markup": True},
        )

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

        self.load_files()
        self.report_aois()
        self.figure = {}
        ax = self.plot_color_patches(
            values=pd.Series(self.aois.index), ax=plt.gca(), save=True
        )

        self.in_aoi()

        if self.metric in [MetricClasses.hit_rate, MetricClasses.all]:
            self.compute_hit_rate()
        if self.metric in [MetricClasses.first_contact, MetricClasses.all]:
            self.compute_first_contact()
        if self.metric in [MetricClasses.dwell_time, MetricClasses.all]:
            self.compute_dwell_time()

        self.save_aois()
        self.create_report()
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
            self.output_path = os.path.join(self.input_path, "output")
            os.makedirs(self.output_path, exist_ok=True)
        logging.info("Input path: %s", self.input_path)

    def get_sq_aois(self):
        self.scaled_aois = cv2.selectROIs("AOI Annotation", self.scaled_image)
        cv2.destroyAllWindows()
        aois = self.scaled_aois
        self.aois = pd.DataFrame(aois, columns=["x", "y", "width", "height"])

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
        if self.aois_path != None:
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

    def compute_hit_rate(self):
        # AOIs that have never been gazed at do not show up in the fixations data
        # so we need to set them to 0 manually
        hits = self.data.groupby(["recording id", "AOI"]).size() > 0
        self.hit_rate = (
            hits.groupby("AOI").sum() / self.data["recording id"].nunique() * 100
        )
        for aoi_id in range(len(self.aois)):
            if not aoi_id in self.hit_rate.index:
                self.hit_rate.loc[aoi_id] = 0
        self.hit_rate.sort_index(inplace=True)
        if len(self.hit_rate) > 10:
            self.hit_rate = self.hit_rate[self.hit_rate > 0]
        self.current = {
            "var": self.hit_rate,
            "title": "Hit Rate",
            "ylabel": "Hit Rate [%]",
            "name": "hit_rate",
        }
        self.plot_results()
        logging.info("Hit rate per AOI:")
        logging.info(self.hit_rate.head())

    def compute_first_contact(self):
        # Compute the time difference for the respective section
        self.sections_df.set_index("section id", inplace=True)
        for section_id, start_time in self.sections_df[
            "section start time [ns]"
        ].items():
            data_indices = self.data.loc[self.data["section id"] == section_id].index
            logging.info(
                "The section {} starts at {} and has {} points".format(
                    section_id,
                    start_time,
                    len(data_indices),
                )
            )
            field_ts = (
                "start timestamp [ns]"
                if self.type == DataClasses.fixations
                else "timestamp [ns]"
            )
            self.data.loc[data_indices, "aligned timestamp [s]"] = (
                self.data.loc[data_indices, field_ts] - start_time
            ) / 1e9
        self.first_contact = self.data.groupby(["section id", "AOI"])[
            "aligned timestamp [s]"
        ].min()
        self.first_contact = self.first_contact.groupby("AOI").mean()
        self.current = {
            "var": self.first_contact,
            "title": "First Contact",
            "ylabel": "Time to first contact [s]",
            "name": "first_contact",
        }
        self.plot_results()
        logging.info(self.first_contact)

    def compute_dwell_time(self):
        # Compute the dwell time for the respective AOI
        if self.type == DataClasses.fixations:
            self.dwell_time = self.data.groupby(["recording id", "AOI"])[
                "duration [ms]"
            ].sum()
            self.dwell_time = self.dwell_time.groupby("AOI").mean()
            self.dwell_time /= 1000
            logging.info(self.dwell_time.head())
            self.current = {
                "var": self.dwell_time,
                "title": "Dwell Time",
                "ylabel": "Dwell Time [s]",
                "name": "dwell_time",
            }
            self.plot_results()
        else:
            logging.info("Dwell time is only available for fixations data")

    def load_files(self):
        # Load the sections file and the fixations file onto pandas DataFrames
        logging.info("Loading files ...")
        self.sections_df = pd.read_csv(self.input_path + "/sections.csv")
        logging.info(self.sections_df["start event name"].unique())
        logging.info(self.sections_df["end event name"].unique())

        self.fixations_df = pd.read_csv(self.input_path + "/fixations.csv")
        logging.info("A total of %d fixations were found", len(self.fixations_df))

        self.gaze_df = pd.read_csv(self.input_path + "/gaze.csv")
        logging.info("A total of %d gaze points were found", len(self.gaze_df))

        # Make data fixations or gaze, depending on the selected type
        self.data_df = (
            self.fixations_df if self.type == DataClasses.fixations else self.gaze_df
        )
        field_detected = (
            "fixation detected in reference image"
            if self.type == DataClasses.fixations
            else "gaze detected in reference image"
        )
        # filter for fixations that are in the reference image and check which AOI they are in
        self.data = self.data_df[self.data_df[field_detected]]

    def in_aoi(self):
        def check_in_rect(self, rect):
            rect_x, rect_y, rect_width, rect_height = rect
            if self.type == DataClasses.fixations:
                fieldx = "fixation x [px]"
                fieldy = "fixation y [px]"
            elif self.type == DataClasses.gaze:
                fieldx = "gaze position in reference image x [px]"
                fieldy = "gaze position in reference image y [px]"
            x_hit = (self.data[fieldx] * self.scaling_factor).between(
                rect_x, rect_x + rect_width
            )
            y_hit = (self.data[fieldy] * self.scaling_factor).between(
                rect_y, rect_y + rect_height
            )
            return x_hit & y_hit

        def check_in_mask(self, mask):
            if self.mask_output == "coco_rle":
                mask = mask_utils.decode(mask)
            if self.type == DataClasses.fixations:
                fieldx = "fixation x [px]"
                fieldy = "fixation y [px]"
            elif self.type == DataClasses.gaze:
                fieldx = "gaze position in reference image x [px]"
                fieldy = "gaze position in reference image y [px]"
            fieldx_values = (self.data[fieldx].values * self.scaling_factor).astype(int)
            fieldy_values = (self.data[fieldy].values * self.scaling_factor).astype(int)
            hit = mask[fieldy_values, fieldx_values]  # check  order
            return hit

        for row in self.aois.itertuples():
            if not self.is_sam:
                data_in_aoi_index = check_in_rect(
                    self, [row.x, row.y, row.width, row.height]
                )
            else:
                data_in_aoi_index = check_in_mask(self, row.segmentation)
            self.data.loc[data_in_aoi_index, "AOI"] = row.Index

        logging.info(
            f"A total of %d {self.type} points were detected in AOIs", len(self.data)
        )

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
            data_in = data[data[field0] == True]
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

    def plot_results(
        self,
    ):
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        kwargs = {
            "ax": ax[1],
            "colorbar": True,
            "alpha": 0.6,
        }
        sns.barplot(
            x=self.current["var"].index.to_numpy(dtype=np.int64),
            y=self.current["var"],
            ax=ax[0],
        )
        ax[0].set_xlabel("AOI ID")
        ax[0].set_ylabel(self.current["ylabel"])
        if "label" in self.aois.columns:
            aoi_id = self.current["var"].index.to_numpy(dtype=np.int64)
            ax[0].set_xticklabels(self.aois.label[aoi_id], rotation=90)
        ax[0].set_title(self.current["title"])
        self.plot_color_patches(
            self.current["var"], unit_label=self.current["ylabel"], **kwargs
        )
        fig.suptitle(f"{self.current['title']} - {self.type}")
        plt.tight_layout()
        figname = f"{self.output_path}/{self.current['title']}_{self.type}.png"
        plt.draw()
        plt.savefig(figname)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        self.figure[self.current["name"]] = buf.getvalue()
        plt.close(fig)

    def create_report(self):
        generate_report(self)


def main():
    aois = defineAOIs()
    aois.run()


if __name__ == "__main__":
    main()
