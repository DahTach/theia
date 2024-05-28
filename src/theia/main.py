import argparse
import os
import cv2 as cv
import warnings
import pathlib
from tqdm import tqdm
import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model", type=str, default="GroundingDINO", help="Model Name"
)
parser.add_argument(
    "-i",
    "--images",
    type=str,
    default="~/Developer/datasets/pallets_sorted/vertical/",
    help="Images of Images Folder Path",
)
parser.add_argument("--mode", type=int, default=0, help="0: batch, 1: nms, 2: Base")

parser.add_argument("--device", type=str, help="Device to use")
parser.add_argument(
    "-s", "--save", action="store_true", default=False, help="Save Drawn Images"
)


def get_image_files(path):
    images = []
    # recursively search for images in the folder
    for folder, _, files in os.walk(path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(folder, file))

    return images


# Generalized main to choose the model
# def main():
#     from model import Model
#     from caption import captions
#
#     args = parser.parse_args()
#     model = Model(args.model, captions)
#
#     for image in get_image_files(args.images):
#         results = model.predict(image)
#         model.show(image, results)


def predict(img_path, model, method=0, save=False):
    image = cv.imread(img_path)
    if method == 0:
        results = model.batch_predict(image)
        image = model.draw_batch(image, results)
    elif method == 1:
        results = model.nms_predict(image)
        image = model.draw_nms(image, results)
    elif method == 2:
        results = model.base_predict(img_path)
        image = model.draw_base(image, results)
    if save:
        utils.save_image(image, img_path)
    else:
        utils.show(image)


def main():
    from dino import Dino
    from autodistill.detection import CaptionOntology
    from caption import captions

    args = parser.parse_args()

    utils.filter_warnings()

    # TODO: Implement multithreading

    try:
        model = Dino(
            ontology=CaptionOntology(utils.get_captions(captions)), device=args.device
        )
    except Exception as e:
        raise e

    # check if the path is a file or a folder
    if os.path.isfile(args.images):
        if args.save:
            path = pathlib.Path(args.images)
            results_dir = path.parent.parent / f"{path.parent.name}_results"
            results_dir.mkdir(exist_ok=True)
        predict(args.images, model, method=args.mode, save=args.save)
    if os.path.isdir(args.images):
        if args.save:
            path = pathlib.Path(args.images)
            results_dir = path.parent / f"{path.name}_results"
            results_dir.mkdir(exist_ok=True)
        for image_path in tqdm(get_image_files(args.images)):
            predict(image_path, model, method=args.mode, save=args.save)


if __name__ == "__main__":
    main()
