import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, help="Model Name")
parser.add_argument("-i", "--images", type=str, help="Image Folder Path")


def get_image_files(path):
    images = []
    # recursively search for images in the folder
    for folder, _, files in os.walk(path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(folder, file))

    return images


def main():
    from model import Model
    from caption import captions

    args = parser.parse_args()
    model = Model(args.model, captions)

    for image in get_image_files(args.images):
        results = model.predict(image)
        model.show(image, results)


if __name__ == "__main__":
    main()
