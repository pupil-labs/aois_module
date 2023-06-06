import os, platform
import logging
import cv2
import numpy as np

import tkinter as tk  # For GUI
from tkinter import filedialog
from pupil_labs.aois_module._parser import MetricClasses, DataClasses


def get_path():
    root = tk.Tk()
    root.withdraw()
    msg = "Select the directory"
    arguments = {"title": msg}
    if platform.system() == "Darwin":
        arguments["message"] = msg
    path = filedialog.askdirectory(**arguments)
    # check if the folder contains the required files
    if (
        not os.path.exists(os.path.join(path, "fixations.csv"))
        or not os.path.exists(os.path.join(path, "gaze.csv"))
        or not os.path.exists(os.path.join(path, "sections.csv"))
        or not os.path.exists(os.path.join(path, "reference_image.jpeg"))
    ):
        error = f"The selected folder does not contain a reference_image.jpeg, fixations.csv, gaze.csv or sections.csv files"
        logging.error(error)
        raise SystemExit(error)
    root.destroy()
    return path


def confirm_read(filename):
    logging.info("AOIs already defined on that folder")
    from tkinter.messagebox import askyesno

    yn = askyesno(
        f"AOIs already defined on that folder {filename}",
        "Do you want to use them?",
    )
    return yn

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
        cv2.imwrite(os.path.join(self.output_path, "scaled_image.jpeg"), self.scaled_image)
    return self
