import os
import progressbar
import warnings
import cv2 as cv
import numpy as np


def get_captions(captions):
    """Inverts a dictionary with list values.

    Args:
        d: The dictionary to invert.

    Returns:
        The inverted dictionary.
    """

    inverted_dict = {}
    for key, value in captions.items():
        for item in value:
            inverted_dict.setdefault(item, key)  # Set default value for repeated values
    return inverted_dict


pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def get_image_files(path):
    images = []
    # recursively search for images in the folder
    for folder, _, files in os.walk(path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(folder, file))

    return images


def show(img: np.ndarray):
    try:
        cv.imshow("results", img)
    except KeyboardInterrupt:
        exit(0)
    key = cv.waitKey(0) & 0xFF
    if key == 27 or key == ord("q") or key == 3:
        exit(0)


def filter_warnings():
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", FutureWarning)
    warnings.filterwarnings(
        "ignore",
        message="The `device` argument is deprecated and will be removed in v5 of Transformers.",
    )
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint")
    warnings.filterwarnings(
        "ignore", message="None of the inputs have requires_grad=True"
    )
    warnings.filterwarnings("ignore", message="To copy construct from a tensor")

    warnings.filterwarnings("ignore", message="torch.meshgrid")
    warnings.filterwarnings("ignore", message="torch.broadcast_tensors")
    # warnings.filterwarnings("ignore", message="Warning: CUDA not available. GroundingDINO will run very slowly.")
