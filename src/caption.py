from autodistill.detection import CaptionOntology
from typing import Dict, List


class Captions:
    def __init__(self, captions: Dict[str, List[str]]):
        self.captions = captions

    def classes(self):
        return set(self.captions.keys())

    def __getitem__(self, item: str):
        return self.captions.get(item)

    def __iter__(self):
        return iter(self.captions)

    def __repr__(self):
        return f"Captions({self.captions})"

    def class_prompts(self, class_name: str) -> List[str]:
        return self.captions[class_name]

    def prompt_class(self, prompt: str) -> List[str] | str:
        key_list = [key for key, val in self.captions.items() if val == prompt]
        if len(key_list) > 0:
            return key_list[0]
        else:
            raise KeyError(f"Prompt {prompt} not found in captions")


captions = {
    "water": [
        "water container",
        "water crate",
        "water case",
        "bottle crate",
        "bottle case",
        "bottle pack",
        "bottle carton",
    ],
    "cans": ["can pack", "carton cans", "can paket", "can case", "can crate"],
    "box": [
        "cardboard box",
        "cardboard",
        "box",
        "package",
        "parcel",
        "carton",
        "pack",
        "packet",
        "case",
        "crate",
        "chest",
        "trunk",
        "coffer",
        "casket",
        "hamper",
        "canteen",
        "bin",
        "container",
    ],
    "keg": [
        "keg",
        "beer keg",
        "beer barrel",
        "barrel",
        "cask",
        "vat",
        "tun",
        "drum",
        "hogshead",
        "firkin",
        "tub",
        "tank",
        "container",
        "vessel",
    ],
}
