import argparse
import os
import cv2 as cv
import warnings

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model", type=str, default="GroundingDINO", help="Model Name"
)
parser.add_argument("-i", "--images", type=str, help="Image Folder Path")
parser.add_argument("-m", "--model", type=str, help="Model Name")
parser.add_argument(
    "-i",
    "--images",
    type=str,
    default="~/Developer/datasets/pallets_sorted/vertical/",
    help="Image Folder Path",
)
parser.add_argument(
    "--image",
    type=str,
    help="Single Image",
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


def main():
    from dino import Dino
    from autodistill.detection import CaptionOntology
    from caption import captions
    import utils

    args = parser.parse_args()

    utils.filter_warnings()

    try:
        model = Dino(ontology=CaptionOntology(utils.get_captions(captions)))
    except Exception as e:
        raise e

    if args.image:
        image = cv.imread(args.image)
        results = model.dino_predict(image)
        image = model.draw(image, results)
        utils.show(image)
    else:
        for image in get_image_files(args.images):
            image = cv.imread(image)
            results = model.dino_predict(image)
            image = model.draw(image, results)
            utils.show(image)


if __name__ == "__main__":
    main()
