import json
from typing import List, Dict

FILE_PATH = "/Users/francescotacinelli/Developer/theia/data/captions.json"


class Captions:
    def __init__(self, file_path):
        if not file_path:
            raise FileNotFoundError("File path not provided.")
        with open(file_path, "r") as f:
            data = json.load(f)
        self.dict: Dict[str, List[str]] = data

    @property
    def categories(self):
        return list(self.dict.keys())

    @property
    def aliases(self):
        return [alias for aliases in self.dict.values() for alias in aliases]

    def __getitem__(self, item):
        return self.dict[item]

    def get(self, alias, default=0):
        for idx, (category, aliases) in enumerate(self.dict.items()):
            if alias in aliases:
                return idx
        print(f"Alias {alias} not found in categories.")
        return default

    def save(self, file_path=FILE_PATH):
        """Saves the model to a file."""
        with open(file_path, "w") as f:
            json.dump(self.dict, f)

    def edit(self, category: str, alias: str, index: int | None = None):
        """Adds or replaces an alias to a category."""
        # check if category exists
        if not self.dict.get(category, False):
            self.dict.setdefault(category, [alias])
        elif index:
            # check if index is valid
            if index < len(self.dict[category]):
                self.dict[category][index] = alias
            else:
                self.dict[category].append(alias)
        # if alias already exists in category replace it
        elif alias in self.dict[category]:
            for i, a in enumerate(self.dict[category]):
                if a == alias:
                    self.dict[category][i] = alias
        # check if alias already exists
        elif alias not in self.aliases:
            self.dict[category].append(alias)


#
# def main():
#     captions = Captions(FILE_PATH)
#     print(captions.categories)
#     print(captions.aliases)
#
#
# if __name__ == "__main__":
#     main()
