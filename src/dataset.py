import os
import json

from torchvision.io.video import re
import utils
from typing import List, Dict, Tuple
from collections import defaultdict

FILE_PATH = "/Users/francescotacinelli/Developer/theia/data/captions.json"


class Captions:
    def __init__(self, file_path):
        if not file_path:
            raise FileNotFoundError("File path not provided.")
        with open(file_path, "r") as f:
            data = json.load(f)
        self.data = defaultdict(None, data)

    @property
    def class_ids(self):
        return self.data.keys()

    @property
    def aliases_name(self):
        aliases = []
        for _, value in self.data.items():
            for alias in value["aliases"]:
                aliases.append(alias["name"] or "")

        return aliases

    @property
    def aliases(self):
        aliases = []
        for value in self.data.values():
            for alias in value["aliases"]:
                aliases.append(alias)

        return aliases

    @property
    def aliases_by_class(self):
        clss_aliases = []
        for clss, value in self.data.items():
            aliases = []
            for alias in value["aliases"]:
                aliases.append(alias["name"] or "")
            clss_aliases.append(aliases)

        return clss_aliases

    @property
    def captions_ontology(self):
        caption_ontology = {}
        for value in self.data.values():
            for alias in value["aliases"]:
                caption_ontology.setdefault(alias.get("name"), value["name"])

        return caption_ontology

    def __getitem__(self, id: int | str):
        id = str(id) if isinstance(id, int) else id
        return self.data[id]

    def get_alias(self, alias_name):
        for _, clss in self.data.items():
            for alias in clss["aliases"]:
                if alias.get("name") == alias_name:
                    return alias

    def get_class_from_alias(self, alias_name):
        for idx, clss in self.data.items():
            for alias in clss["aliases"]:
                if alias.get("name") == alias_name:
                    return int(idx)

    def aliases_from_class(self, id: int):
        aliases = []
        for id, value in self.data.items():
            if id == id:
                for alias in value["aliases"]:
                    aliases.append(alias.get("name"))
        return aliases

    def add_metric(self, alias_name, metric, value):
        for _, clss in self.data.items():
            for alias in clss["aliases"].items():
                if alias.get("name") == alias_name:
                    alias["metrics"][str(metric)] = value

    def add_alias(self, id, alias, metrics={}):
        for clss in self.data.values():
            for alias in clss["aliases"]:
                if alias.get("name") == alias:
                    return

        self.data[id]["aliases"].append({"name": alias, "metrics": {**metrics}})

    def get_AP(self, alias_name):
        for _, clss in self.data.items():
            for alias in clss["aliases"]:
                if alias.get("name") == alias_name:
                    return alias.get("metrics").get("ap")

    def get_metrics(self, alias_name):
        for _, clss in self.data.items():
            for alias in clss["aliases"]:
                if alias.get("name") == alias_name:
                    return alias.get("metrics")

    def get_description(self, id, alias_name=None):
        if not alias_name:
            return self.data[id].get("description")
        else:
            id = self.get_class_from_alias(alias_name)
            return self.data[id].get("description")

    def save(self, file_path=FILE_PATH):
        """Saves the model to a file."""
        with open(file_path, "w") as f:
            json.dump(self.data, f)


class Dataset:
    def __init__(
        self, path="/Users/francescotacinelli/Developer/datasets/pallets_sorted/test/"
    ):
        self.path = path
        self.captions_path = (
            "/Users/francescotacinelli/Developer/theia/data/captions.json"
        )
        self.images: List[str]
        self.ground_truths: Dict[int, List[Tuple]]
        self.classes: Dict[int, str]
        self.images, self.ground_truths, self.classes = self.init()
        self.captions = self.init_captions(self.captions_path)

    def init(self):
        # iterate over the dataset and return the images and the ground truths
        images = []
        ground_truths = {}
        classes = {}
        if not self.path:
            raise FileNotFoundError("Dataset path not provided.")
        elif not os.path.exists(self.path):
            raise FileNotFoundError("Dataset path does not exist.")
        else:
            print("Loading dataset...")
        for folder, _, files in os.walk(self.path):
            for file in files:
                if os.path.basename(file) == "classes.txt":
                    try:
                        with open(os.path.join(folder, file)) as f:
                            class_names = f.read().splitlines()
                            for i, name in enumerate(class_names):
                                classes.setdefault(i, name)
                    except FileNotFoundError:
                        print(f"No classes for {file}")
                if file.endswith((".jpg", ".jpeg", ".png")):
                    images.append(os.path.join(folder, file))
                    try:
                        with open(
                            os.path.join(
                                folder, os.path.basename(file).split(".")[0] + ".txt"
                            )
                        ) as f:
                            grounds = f.read().splitlines()
                            for line in grounds:
                                class_id, *bbox = line.split(" ")
                                ground_truths.setdefault(
                                    os.path.basename(file).split(".")[0], []
                                ).append(
                                    (int(class_id), [float(coord) for coord in bbox])
                                )
                    except FileNotFoundError:
                        print(f"No ground truth for {file}")
                        ground_truths.setdefault(
                            os.path.basename(file).split(".")[0], []
                        )

        return images, ground_truths, classes

    def init_captions(self, captions_path):
        return Captions(file_path=captions_path)
