import importlib

import cv2 as cv
import numpy.typing as npt

from autodistill.detection import CaptionOntology
from autodistill.utils import plot


models = {
    # SUPPORTED MODELS
    "GroundedSAM": "autodistill_grounded_sam",  # Amazing results
    "GroundingDINO": "autodistill_grounding_dino",  # Good results
    # TODO: implement new samclip api: https://github.com/autodistill/autodistill-sam-clip
    "SAMCLIP": "autodistill_sam_clip",
    # TODO: implement OWLv2 score thresholding or check nms
    "OWLv2": "autodistill_owlv2",  # Might have good results
    "OWLViT": "autodistill_owl_vit",  # Few detections
    "Kosmos2": "autodistill_kosmos_2",  # Seems to not be working
    "FastSAM": "autodistill_fastsam",  # Gets stuck in inference
    # UNSUPPORTED MODELS / missing dependencies
    "LLaVA": "autodistill_llava",  # Only available for CUDA
    "DETIC": "autodistill_detic",  # detectron2
    "CoDet": "autodistill_codet",  # detectron2
    "VLPart": "autodistill_vlpart",  # detectron2
}


class Model:
    def __init__(self, name, captions):
        self.name = name
        self.captions = self.get_captions(captions)
        self.available_models = models
        self.model = self.__getitem__(name)

    def __repr__(self):
        return f"{self.name}"

    def _get_model(self, name):
        import_name = self.available_models[name]
        module = importlib.import_module(import_name)
        imported_class = getattr(module, self.name)
        return imported_class

    def get_captions(self, captions):
        """Inverts a dictionary with list values.

        Args:
            d: The dictionary to invert.

        Returns:
            The inverted dictionary.
        """

        inverted_dict = {}
        for key, value in captions.items():
            for item in value:
                inverted_dict.setdefault(
                    item, key
                )  # Set default value for repeated values
        return inverted_dict

    def __getitem__(self, name):
        model = self._get_model(name)
        return model(ontology=CaptionOntology(self.captions))

    def predict(self, image: str):
        return self.model.predict(image)

    def label(self, folder, extension):
        return self.model.label(folder, extension)

    def draw(self, image, results) -> npt.NDArray:
        print(f"RESULTS: {results}")
        return plot(
            image=image,
            classes=self.model.ontology.classes(),
            # slice the last result from the list to fix autodistill bug
            detections=results,
            raw=True,
        )

    def show(self, image, results):
        img = cv.imread(str(image))
        img = self.draw(img, results)

        try:
            cv.imshow("results", img)
        except KeyboardInterrupt:
            exit(0)

        key = cv.waitKey(0) & 0xFF
        if key == 27 or key == ord("q") or key == 3:
            exit(0)

    def show_cv(self, img, results):
        # img = cv.imread(str(image))
        img = self.draw(img, results)

        # try:
        #     cv.imshow("results", img)
        # except KeyboardInterrupt:
        #     exit(0)
        #
        # key = cv.waitKey(0) & 0xFF
        # if key == 27 or key == ord("q") or key == 3:
        #     exit(0)
        return img
