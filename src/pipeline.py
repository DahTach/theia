"""
Steps:
- Dataset
- Inference and ground truths on the dataset
- Dataset metrics
"""

import json
from collections import defaultdict

json_sample = {
    "images": [
        {
            "name": "image1",
            "frame": "horizontal",
            "gr_truths": [("class_id", [0.1, 0.2, 0.3, 0.4])],
        },
    ],
    "aliases": [
        {
            "class_name": "class_id",
            "frame": "vertical",
            "alias": ["class_name"],
            "metrics": {
                "precision": 0.1,
                "recall": 0.2,
                "f1": 0.3,  # f1 score
            },
        }
    ],
}


class Image:
    def __init__(self, path, gr_truths, framing):
        self.path = path
        self.gr_truths = gr_truths
        self.frame = framing


class Alias:
    def __init__(self, class_id, class_name, alias, metrics, path, framing):
        self.class_id = class_id
        self.class_name = class_name
        self.alias = alias
        self.metrics: Metrics = metrics
        self.path = path
        self.frame = framing


class Metrics:
    def __init__(self, results, gr_truths):
        self.precision = 0
        self.recall = 0
        self.f1 = 0


class Dataset:
    def __init__(self, json, clss_path):
        self.path = json
        self.clss_path = clss_path
        self.images = []
        self.aliases = []
        self.classnames = self.classes

    def add_image(self, image: Image):
        if isinstance(image, Image):
            self.images.append(image)
        else:
            raise ValueError("Image object is expected")

    def add_alias(self, alias: Alias):
        if isinstance(alias, Alias):
            self.aliases.append(alias)
        else:
            raise ValueError("Alias object is expected")

    @property
    def classes(self):
        # open classes.txt and return all classes in a list
        with open(self.clss_path, "r") as f:
            return f.readlines()

    @property
    def history(self):
        # get all aliases from the dataset
        history = defaultdict(dict)
        for alias in self.aliases:
            history[alias.class_name][alias.alias] = alias.metrics
        return history

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def auto_alias(self, class_id):
        vision = Vision("dino")
        lang = LLM(self.history)
        alias = ""
        for image in self.images:
            results = vision.predict(image.path)
            metrics = Metrics(results, image.gr_truths)
            self.add_alias(
                Alias(
                    class_id=class_id,
                    class_name=self.classnames[class_id],
                    alias=alias,
                    metrics=metrics,
                    path=image.path,
                    framing=image.frame,
                )
            )
        self.save()


class LLM:
    def __init__(self, history):
        self.history = history
        self.model = None

    def generate(self):
        pass


class Vision:
    def __init__(self, model):
        self.model = model

    def predict(self, image):
        pass


class Pipeline:
    def __init__(self, dataset, inference, ground_truths):
        self.dataset = dataset
        self.inference = inference
        self.ground_truths = ground_truths

    def dataset_metrics(self):
        pass

    def run(self):
        self.dataset_metrics()


def pipeline():
    images = [""]
    for image in images:
        image = Image(image, gr_truths, framing)
        dataset.add_image(image)
    dataset.auto_alias()
    dataset.save("dataset.json")
