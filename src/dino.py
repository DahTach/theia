import os
import urllib.request
from typing import List, Tuple

import cv2 as cv
import groundingdino.datasets.transforms as T
import numpy as np
import torch
import torchvision
from groundingdino.models import build_model
from groundingdino.util.inference import Model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap
from PIL import Image
from torchvision.ops import box_convert
from tqdm import tqdm
import utils
import supervision as sv
from autodistill.helpers import load_image
from autodistill_grounding_dino.helpers import combine_detections, load_grounding_dino
from caption import Captions

TEST_FILE_PATH = "/Users/francescotacinelli/Developer/theia/data/test_captions.json"
captions = Captions(TEST_FILE_PATH)

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "-i",
#     "--images",
#     type=str,
#     default="~/Developer/datasets/pallets_sorted/vertical/",
#     help="Image Folder Path",
# )
# parser.add_argument(
#     "--image",
#     type=str,
#     help="Single Image",
# )


class Dino:
    def __init__(self, ontology, device=None, box_threshold=0.35, text_threshold=0.25):
        self.device = self.get_device() if not device else device
        self.ontology = ontology
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.distill_cache_dir = os.path.expanduser("~/.cache/autodistill")
        self.cache = os.path.join(self.distill_cache_dir, "groundingdino")
        self.config = os.path.join(self.cache, "GroundingDINO_SwinT_OGC.py")
        self.checkpoint = os.path.join(self.cache, "groundingdino_swint_ogc.pth")
        self.load()

    def get_device(self):
        if torch.cuda.is_available():
            print("using cuda")
            return "cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("using cpu because mps is not fully supported yet")
            return "cpu"
        else:
            print("using cpu")
            return "cpu"

    def load(self):
        try:
            print("trying to load grounding dino directly")
            self.model = DinoModel(
                model_config_path=self.config,
                model_checkpoint_path=self.checkpoint,
                device=self.device,
            )
        except Exception as e:
            print(f"Occured error: {e}")
            print("downloading dino model weights")
            if not os.path.exists(self.cache):
                os.makedirs(self.cache)

            if not os.path.exists(self.checkpoint):
                url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
                urllib.request.urlretrieve(url, self.checkpoint, utils.show_progress)

            if not os.path.exists(self.config):
                url = "https://raw.githubusercontent.com/roboflow/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
                urllib.request.urlretrieve(url, self.config, utils.show_progress)

            self.model = DinoModel(
                model_config_path=self.config,
                model_checkpoint_path=self.checkpoint,
                device=self.device,
            )

    def preprocess_image(self, image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    def phrases2classes(self, phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            try:
                class_ids.append(classes.index(phrase))
            except ValueError:
                class_ids.append(None)
        return np.array(class_ids)

    def evaluate(self, image, captions, box_threshold=0.1, text_threshold=0.25)-> List[Tuple]:
        image = cv.imread(image)
        predictions: List[Tuple] = []
        for prompt in tqdm(captions):
            boxes, class_ids = self.model.fast_predict_with_prompt(
                image=image,
                prompt=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            for class_id, box in zip(class_ids, boxes):
                predictions.append((class_id, box))

        return predictions

    def base_predict(self, input: str, progress=None, prompts=[]) -> sv.Detections:
        image = load_image(input, return_format="cv2")

        detections_list = []

        # progress = tqdm(self.ontology.prompts())
        progress = tqdm(prompts)
        for prompt in progress:
            detections = self.model.predict_with_classes(
                image=image,
                classes=[prompt],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )

            detections_list.append(detections)

            # progress.update(1)

        detections = combine_detections(
            detections_list, overwrite_class_ids=range(len(detections_list))
        )

        return detections

    def nms_predict(
        self,
        image: np.ndarray,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        progress=None,
    ):
        # TODO: draw the results without nms
        detections = [torch.tensor([]), torch.tensor([]), []]
        progress = tqdm(
            [prompt for prompts in captions.dict.values() for prompt in prompts]
        )
        for prompt in progress:
            boxes, confidences, class_ids = self.model.predict_with_prompt(
                image=image,
                prompt=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            detections[0] = torch.cat((detections[0], boxes), dim=0)
            detections[1] = torch.cat((detections[1], confidences), dim=0)

            # extend class_ids with the same class for each box
            detections[2].extend([prompt] * len(class_ids))

            progress.update(1)

        filtered_detections = self.fast_nms(detections)

        return filtered_detections

    def batch_predict(
        self,
        image: np.ndarray,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        progress=None,
    ):
        detections = [torch.tensor([]), torch.tensor([]), []]
        progress = tqdm(captions.dict.items())
        for cls, prompts in progress:
            boxes, confidences, class_ids = self.model.predict_with_prompts(
                image=image,
                prompts=prompts,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            detections[0] = torch.cat((detections[0], boxes), dim=0)
            detections[1] = torch.cat((detections[1], confidences), dim=0)

            # extend class_ids with the same class for each box
            detections[2].extend([cls] * len(class_ids))

            progress.update(1)

        filtered_detections = self.fast_nms(detections)

        return filtered_detections

    def fast_nms(
        self,
        detections,
        iou_threshold=0.5,
        containment_threshold=0.8,
        size_deviation_threshold=1.5,
    ):
        all_boxes = detections[0]
        all_scores = detections[1]
        all_class_ids = detections[2]

        # Perform NMS
        keep_indices = torchvision.ops.nms(all_boxes, all_scores, iou_threshold)

        # TODO: roi align the boxes with the pallet
        # torchvision.ops.roi_align

        # Remove boxes that are bigger than average by size deviation threshold
        areas = torchvision.ops.box_area(all_boxes)
        avg_area = torch.mean(areas)
        for i, area in enumerate(areas):
            if area > avg_area * size_deviation_threshold:
                # Remove this index from keep_indices
                keep_indices = keep_indices[keep_indices != i]

        # Remove boxes with high containment in others
        remove_indices = []
        for i in keep_indices:
            box_i = all_boxes[i]
            for j in keep_indices:
                if i == j:  # Skip self-comparison
                    continue
                box_j = all_boxes[j]

                # Calculate intersection area
                inter_width = torch.max(
                    torch.tensor(0),
                    torch.min(box_i[2], box_j[2]) - torch.max(box_i[0], box_j[0]),
                )
                inter_height = torch.max(
                    torch.tensor(0),
                    torch.min(box_i[3], box_j[3]) - torch.max(box_i[1], box_j[1]),
                )
                inter_area = inter_width * inter_height
                box_i_area = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])

                # Check for high containment
                containment_ratio = inter_area / box_i_area
                if containment_ratio > containment_threshold:
                    remove_indices.append(i)
                    break

        # Update keep_indices, removing those with high containment
        keep_indices = [idx for idx in keep_indices if idx not in remove_indices]

        # Reconstruct the final list of detections with original class IDs
        filtered_detections = [[], [], []]
        for index in keep_indices:
            filtered_detections[0].append(all_boxes[index].tolist())
            filtered_detections[1].append(all_scores[index].item())
            filtered_detections[2].append(all_class_ids[index])

        return filtered_detections

    def draw_base(self, image: np.ndarray, detections):
        return self.plot(
            image=image,
            classes=self.ontology.classes(),
            # slice the last result from the list to fix autodistill bug
            detections=detections,
            raw=True,
        )

    def plot(self, image: np.ndarray, detections, classes: List[str], raw=False):
        """
        Plot bounding boxes or segmentation masks on an image.

        Args:
            image: The image to plot on
            detections: The detections to plot
            classes: The classes to plot
            raw: Whether to return the raw image or plot it interactively

        Returns:
            The raw image (np.ndarray) if raw=True, otherwise None (image is plotted interactively
        """
        # TODO: When we have a classification annotator
        # in supervision, we can add it here
        if detections.mask is not None:
            annotator = sv.MaskAnnotator()
        else:
            annotator = sv.BoxAnnotator()

        label_annotator = sv.LabelAnnotator()

        labels = [
            f"{classes[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _ in detections
        ]

        annotated_frame = annotator.annotate(scene=image.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, labels=labels, detections=detections
        )

        if raw:
            return annotated_frame

        sv.plot_image(annotated_frame, size=(8, 8))

    def draw_nms(self, image: np.ndarray, detections):
        # Draw boxes on the image

        # change color based on class
        colors = {
            "water": (0, 255, 0),
            "cans": (0, 0, 255),
            "box": (255, 0, 0),
            "keg": (255, 255, 0),
            "bottle": (0, 255, 255),
        }
        boxes = detections[0]
        confidences = detections[1]
        prompt_ids = detections[2]

        class_ids = []
        for prompt_id in prompt_ids:
            for cls, prompts in captions.dict.items():
                if prompt_id in prompts:
                    class_ids.append(cls)

        for i, (box, confidence, prompt_id) in enumerate(
            zip(boxes, confidences, prompt_ids)
        ):
            class_id = class_ids[i]
            color = colors.get(class_id, (0, 0, 0))
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv.putText(
                image,
                f"{prompt_id}: {confidence:.2f}",
                (x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

        return image

    def draw_batch(self, image: np.ndarray, detections):
        # Draw boxes on the image

        # change color based on class
        colors = {
            "water": (0, 255, 0),
            "cans": (0, 0, 255),
            "box": (255, 0, 0),
            "keg": (255, 255, 0),
            "bottle": (0, 255, 255),
        }
        boxes = detections[0]
        confidences = detections[1]
        class_ids = detections[2]

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            color = colors.get(class_id, (0, 0, 0))
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv.putText(
                image,
                f"{class_id}: {confidence:.2f}",
                (x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

        return image


class DinoModel(Model):
    def __init__(
        self, model_config_path: str, model_checkpoint_path: str, device: str = "cuda"
    ):
        self.model = self.load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device,
        ).to(device)
        self.device = device

    def load_model(
        self, model_config_path: str, model_checkpoint_path: str, device: str = "cpu"
    ):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model

    def predict_with_prompts(
        self,
        image: np.ndarray,
        prompts: List[str],
        box_threshold: float,
        text_threshold: float,
    ):
        caption = ". ".join(prompts)
        processed_image = self.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = self.predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        source_h, source_w, _ = image.shape
        class_id = self.phrases2classes(phrases=phrases, classes=prompts)
        confidence = torch.tensor(logits)

        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        return boxes, confidence, class_id

    def fast_predict_with_prompt(
        self,
        image: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Tuple[List[torch.Tensor], List]:
        processed_image = self.preprocess_image(image_bgr=image).to(self.device)
        boxes = self.fast_predict(
            model=self.model,
            image=processed_image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        class_id = captions.get(prompt)
        class_ids = [class_id for _ in range(len(boxes))]

        return boxes, class_ids

    def predict_with_prompt(
        self,
        image: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ):
        # caption = ". ".join(classes)
        processed_image = self.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = self.predict(
            model=self.model,
            image=processed_image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        source_h, source_w, _ = image.shape
        class_id = self.phrases2classes(phrases=phrases, classes=[prompt])
        confidence = torch.tensor(logits)

        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        return boxes, confidence, class_id

    def preprocess_caption(self, caption: str) -> str:
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."

    def fast_predict(
        self,
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cpu",
    ) -> List[torch.Tensor]:
        caption = self.preprocess_caption(caption=caption)

        model = model.to(device)
        image = image.to(device)

        with torch.no_grad():
            outputs = model(image[None], captions=[caption])

        mask = outputs["pred_logits"].sigmoid()[0].max(dim=1)[0] > box_threshold
        boxes = outputs["pred_boxes"][0][mask]

        return boxes

    def predict(
        self,
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        caption = self.preprocess_caption(caption=caption)

        model = model.to(device)
        image = image.to(device)

        with torch.no_grad():
            outputs = model(image[None], captions=[caption])

        prediction_logits = (
            outputs["pred_logits"].cpu().sigmoid()[0]
        )  # prediction_logits.shape = (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[
            0
        ]  # prediction_boxes.shape = (nq, 4)

        # TODO: refactor to a new predict method that returns the original boxes tensors

        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)

        phrases = [
            get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenizer
            ).replace(".", "")
            for logit in logits
        ]

        return boxes, logits.max(dim=1)[0], phrases


# def main():
#     args = parser.parse_args()
#       import utils
#
#     utils.filter_warnings()
#
#     try:
#         model = Dino(ontology=CaptionOntology(get_captions(captions)))
#     except Exception as e:
#         raise e
#
#     if args.image:
#         image = cv.imread(args.image)
#         results = model.dino_predict(image)
#         image = model.draw(image, results)
#         show(image)
#     else:
#         for image in utils.get_image_files(args.images):
#             image = cv.imread(image)
#             results = model.dino_predict(image)
#             image = model.draw(image, results)
#           utils.show(image)
#
#
# if __name__ == "__main__":
#     main()
