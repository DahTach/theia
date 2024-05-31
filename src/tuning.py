# Automated search of the best aliases for the grounded model
import os
import itertools
from typing import List, Tuple, Dict
import torch
from collections import defaultdict
import torchvision.ops as ops
import torchvision.utils as torchutils
import torchvision.io as io
import cv2 as cv

"""
Steps:
- benchmark the grounded model with the default aliases
- changing aliases one category at a time
- use the benchmarking results to determine the best aliases
- keep doing this until we find the best aliases
"""


class Dataset:
    def __init__(self, path):
        self.path = path
        self.images: List[str]
        self.ground_truths: Dict[int, List[Tuple]]
        self.classes: Dict[int, str]
        self.images, self.ground_truths, self.classes = self.init()

    def init(self):
        # iterate over the dataset and return the images and the ground truths
        images = []
        ground_truths = {}
        classes = {}
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


class Tuning:
    def __init__(self, model, dataset, captions):
        self.model = model
        self.dataset = dataset
        self.captions = captions

    def benchmark(self, dataset, captions, draw=False):
        """
        Benchmark the grounded model with the default aliases
        """
        benchmark = {}
        for image, ground_truth in zip(
            self.dataset.images, self.dataset.ground_truths.values()
        ):
            predictions = self.model.evaluate(image, captions)
            metrics, matched_boxes = self.compare(predictions, ground_truth)
            benchmark.setdefault(image, metrics)
            if draw and matched_boxes:
                image = self.draw(matched_boxes, image, metrics)
                cv.imshow("results", image)
                key = cv.waitKey(0) & 0xFF
                if key == 27 or key == ord("q") or key == 3:
                    exit(0)

        return benchmark

    def draw(self, matched_boxes, image_path, metrics):
        # FIXME: boxes are traslated and predictions are not drawn
        """
        Draw the matched boxes on the images
        """
        # image = torch.tensor(cv.imread(image_path))
        image = io.read_image(image_path)
        c, w, h = image.shape
        # make a different color for each class, except for the ground truth that is always black
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
        ]

        def denormalize_boxes(boxes, original_image_size):
            """
            Converts normalized xyxy bounding boxes to pixel coordinates based on the original image size.

            Args:
                boxes (torch.Tensor): Normalized bounding boxes in xyxy format (values between 0 and 1).
                original_image_size (tuple): Original image size (height, width).

            Returns:
                torch.Tensor: Bounding boxes in xyxy format with pixel coordinates.
            """

            height, width = original_image_size
            scale_factor = torch.tensor(
                [width, height, width, height], device=boxes.device
            )
            denormalized_boxes = boxes * scale_factor
            return denormalized_boxes.int()

        for class_id, (predictions, ground_truths) in matched_boxes.items():
            # preds = predictions * torch.Tensor([w, h, w, h])
            # grounds = ground_truths * torch.Tensor([w, h, w, h])
            preds = denormalize_boxes(predictions, (h, w))
            grounds = denormalize_boxes(ground_truths, (h, w))
            torchutils.draw_bounding_boxes(
                image,
                preds,
                labels=[self.dataset.classes[class_id] for _ in range(len(preds))],
                colors=colors[class_id],
            )
            image = torchutils.draw_bounding_boxes(
                image, grounds, labels=None, colors=(0, 0, 0)
            )

        # image = image[0].permute(1, 2, 0).numpy()
        image = image.permute(1, 2, 0).numpy()
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        # draw the metrics on the bottom left corner of the image
        for idx, metric in metrics.items():
            class_name = self.dataset.classes[idx]
            tp = metric.get("true_positives", 0)
            fp = metric.get("false_positives", 0)
            fn = metric.get("false_negatives", 0)
            cv.putText(
                image,
                f"{class_name}: TP={tp}, FP={fp}, FN={fn}",
                (0, h - 20 * idx),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        return image

    def inverse_normalize(self, tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    def compare(self, predictions: List[Tuple], ground_truths: List[Tuple]):
        """
        Compare the ground truth and the predictions

        Args:
            predictions: List of Tuples (bbox, class_id) with the predictions
            ground_truths: List of tuples (bbox, class_id) with the ground truths

        Returns:
            results: Dictionary with the results of the comparison metrics (keys are the classes)
            metrics: Dictionary with the metrics of the comparison
        """

        metrics = {}
        matched_boxes = {}

        # TODO: Implement better logic for this

        # Organize predictions and ground truths by class for efficiency
        predictions_by_class = defaultdict(list)
        ground_truths_by_class = defaultdict(list)

        for class_id, bbox in predictions:
            predictions_by_class[class_id].append(bbox)
        for class_id, bbox in ground_truths:
            ground_truths_by_class[class_id].append(bbox)

        # Convert to tensors once per class for faster calculations
        for idx in self.dataset.classes.keys():
            # Check for empty lists

            class_predictions = predictions_by_class.get(idx, [])
            class_ground_truths = ground_truths_by_class.get(idx, [])

            if not class_ground_truths:
                metrics[idx] = {"false_positives": len(class_predictions)}
                continue
            if not class_predictions:
                metrics[idx] = {"false_negatives": len(class_ground_truths)}
                continue

            # Convert lists of boxes to tensors
            gr_bbxs = ops.box_convert(
                boxes=torch.tensor(class_ground_truths),
                in_fmt="cxcywh",
                out_fmt="xyxy",
            )
            pr_bbxs = ops.box_convert(
                boxes=torch.stack(class_predictions, dim=0),
                in_fmt="cxcywh",
                out_fmt="xyxy",
            )

            # Find matching pairs (if any) based on IoU
            iou_matrix = ops.box_iou(gr_bbxs, pr_bbxs)  # Efficient IoU calculation
            matched_indices = torch.where(
                iou_matrix >= 0.5
            )  # Assuming IoU threshold of 0.5

            # total number of boxes that are matched
            trues = matched_indices[0].numel()
            # Count True Positives (TP) or the number of boxes that are matched and not overlapped
            tp_count = matched_indices[0].unique().numel()
            # number of boxes that are matched but overlapped
            overlap_count = trues - tp_count
            # number of boxes that are not matched
            unmatched = len(iou_matrix[0]) - trues
            # Count False Positives (FP)
            fp_count = len(class_predictions) - tp_count
            # Count False Negatives (FN)
            fn_count = len(class_ground_truths) - tp_count

            metrics[idx] = {
                "matched": trues,
                "overlap": overlap_count,
                "true_positives": tp_count,
                "unmatched": unmatched,
                "false_positives": fp_count,
                "false_negatives": fn_count,
            }

            # Get the indices of the matching boxes
            if matched_indices[0].numel() > 0:
                truth_indices, pred_indices = matched_indices
                matched_boxes.setdefault(
                    idx, (pr_bbxs[pred_indices], gr_bbxs[truth_indices])
                )

        return metrics, matched_boxes

    def tune(self):
        """
        Automated search of the best aliases for the grounded model
        """
        pass


def main():
    import utils
    import json
    from dino import Dino
    from autodistill.detection import CaptionOntology
    from caption import Captions

    TEST_FILE_PATH = "/Users/francescotacinelli/Developer/theia/data/test_captions.json"
    captions = Captions(file_path=TEST_FILE_PATH).dict

    utils.filter_warnings()

    try:
        model = Dino(ontology=CaptionOntology(utils.get_captions(captions)))
    except Exception as e:
        raise e

    # dataset = Dataset(
    #     "/Users/francescotacinelli/Developer/datasets/pallets_sorted/labeled_vertical/"
    # )

    dataset = Dataset(
        "/Users/francescotacinelli/Developer/datasets/pallets_sorted/test/"
    )
    print(
        f"Benchmarking {len(dataset.images)} images from {dataset.path.split('/')[-1]} dataset"
    )

    tuning = Tuning(model, dataset, captions)
    captions_benchmark = tuning.benchmark(dataset, captions, draw=True)

    print(captions_benchmark)

    # save captions_benchmark dict to json file
    benchmark_path = (
        "/Users/francescotacinelli/Developer/theia/benchmarks/captions_benchmark.json"
    )
    benchmark = {
        "captions": captions,
        "benchmark": captions_benchmark,
    }
    with open(benchmark_path, "w") as f:
        json.dump(benchmark, f)


if __name__ == "__main__":
    main()
