import argparse
from enum import Enum

from pupil_labs.aois_module._models import SAM_MODELS


class MetricClasses(Enum):
    all = 0
    hit_rate = 1
    first_contact = 2
    dwell_time = 3

    def __str__(self):
        return self.name


class DataClasses(Enum):
    fixations = 0
    gaze = 1

    def __str__(self):
        return self.name


def group_decorator(title):
    def inner_group_decorator(func):
        def wrapper(parser):
            group = parser.add_argument_group(title)
            func(group)
            return group

        return wrapper

    return inner_group_decorator


def init_parser():
    parser = argparse.ArgumentParser(description="Pupil Labs - AOI Annotation")

    @group_decorator("General")
    def general_group(group):
        group.add_argument(
            "--metric",
            default=MetricClasses.all,
            type=MetricClasses,
            choices=list(MetricClasses),
            help="Metric to calculate",
        )
        group.add_argument(
            "--type",
            default=DataClasses.fixations,
            type=str,
            choices=list(DataClasses),
            help="Whether to use fixations or gaze points",
        )

    @group_decorator("Input/Output")
    def input_output_group(group):
        group.add_argument(
            "-i",
            "--input_path",
            default=None,
            type=str,
            help="Path to the reference image mapper download folder from Pupil Cloud",
        )
        group.add_argument(
            "-o",
            "--output_path",
            default=None,
            type=str,
            help="Path to the output folder, where to save the results",
        )
        group.add_argument(
            "-cloud",
            "--upload2cloud",
            default=False,
            type=bool,
            help="Whether to upload to Cloud or not",
        )

    @group_decorator("Events")
    def events_group(group):
        group.add_argument(
            "--start",
            default="recording.begin",
            type=str,
            help="Name of the start event",
        )
        group.add_argument(
            "--end", default="recording.end", type=str, help="Name of the end event"
        )

    @group_decorator("AOI")
    def aoi_group(group):
        group.add_argument(
            "-sam",
            "--is_sam",
            default=True,
            type=bool,
            help="Whether to use segment anything to automatically segment the AOIs or not",
        )
        group.add_argument(
            "--sam_model",
            "-sm",
            default="vit_h",
            type=str,
            choices=list(SAM_MODELS.keys()),
            help="Path to the SAM model",
        )
        group.add_argument(
            "--mask_output",
            default="binary_mask",
            choices=["coco_rle", "binary_mask"],
            type=str,
        )
        group.add_argument(
            "-aois",
            "--aois_path",
            default=None,
            type=str,
            help="Path to already defined AOIs",
        )
        group.add_argument(
            "-dino",
            "--dino-text-input",
            default=None,
            type=str,
            help="What should I look for? eg. player, tv, ...",
        )
        group.add_argument(
            "--scaling_factor",
            default=None,
            type=float,
            help="Scaling factor for the reference image",
        )

    @group_decorator("Viz")
    def viz_group(group):
        group.add_argument(
            "-s",
            "--scatter",
            action="store_true",
            help="Whether to add a scatter plot or not",
        )
        parser.set_defaults(scatter=False)

    for group in [
        general_group,
        input_output_group,
        events_group,
        aoi_group,
        viz_group,
    ]:
        group(parser)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = init_parser()
