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
import imagesize
from guessing import Guesser
import numpy as np

"""
Steps:
- benchmark the grounded model with the default aliases
- changing aliases one category at a time
- use the benchmarking results to determine the best aliases
- keep doing this until we find the best aliases
"""


class Tuning:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.guesser = Guesser()
        self.benchmarks = []

    def tune(self, clss: int = 0):
        """
        Automated search of the best aliases for the grounded model
        """

        previous_aliases = self.dataset.captions.aliases
        previous_aliases_names = self.dataset.captions.aliases_name

        description = self.dataset.captions.get_description(clss)
        for i in range(100):
            prev_alias = previous_aliases_names[i - 1] if i > 0 else "Box"
            prev_AP = self.dataset.captions.get_AP(prev_alias) or 0.1
            aliasAP = self.benchmark(alias)
            # self.dataset.captions.add_AP(alias, aliasAP)
            # prev_AP = self.dataset.captions.get_AP(aliases[i - 1]) or 0.1
            # prev_alias = aliases[i - 1] if i > 0 else "Box"
            guess = self.guesser.guess(
                prev_alias=prev_alias,
                prev_AP=prev_AP,
                description=description,
            )
            print(f"Guess for alias {alias}: {guess}")
            self.add_alias(alias, aliasAP)
        break

    def add_alias(self, alias_name, metrics):
        """
        Save the results of the guess
        """
        self.dataset.captions.add_alias(alias_name, metrics)
        self.dataset.caption.save()

    def benchmark(self, alias, draw=False) -> float:
        """
        Benchmark the grounded model with the default aliases, one alias at a time
        """

        class_id = self.dataset.captions.get_class_from_alias(alias)
        metric_dict = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }
        threshold_dict = {i: metric_dict for i in range(0, 10)}
        dataset_metrics = defaultdict(dict, threshold_dict)

        for img_path, ground_truth in zip(
            self.dataset.images, self.dataset.ground_truths.values()
        ):
            predictions = self.model.evalias(img_path, alias, class_id)
            metrics = self.evalias(predictions, ground_truth, class_id)

            for thr, mets in metrics.items():
                for name, value in mets.items():
                    dataset_metrics[thr][name] += value

        return self.aliasAP(dataset_metrics)

    def aliasAP(
        self,
        thr_metrics: Dict,
        iou_thresholds: List[float] = [i for i in range(0, 10)],
    ) -> float:
        precisions = []
        recalls = []
        for iou_thr in iou_thresholds:  # Iterate over thresholds
            tp = thr_metrics[iou_thr]["true_positives"]
            fp = thr_metrics[iou_thr]["false_positives"]
            fn = thr_metrics[iou_thr]["false_negatives"]

            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0

            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0

            precisions.append(precision)
            recalls.append(recall)

        # Calculate AP for this alias
        AP = self.calc_ap(precisions, recalls)

        return AP

    # def classAP(
    #     self,
    #     confusion_matrix: Dict,
    #     iou_thresholds: List[float] = [i for i in range(0, 10)],
    # ):
    #     """Calculates mean Average Precision (mAP) from confusion matrix.
    #
    #     Args:
    #         confusion_matrix: Dictionary with the confusion matrix data.
    #         iou_thresholds: List of IoU thresholds used.
    #
    #     Returns:
    #         float: The calculated mAP.
    #         dict: A dictionary containing AP for each class.
    #     """
    #
    #     alias_APs = {}  # Store AP for each class
    #
    #     for alias_id, thr_metrics in confusion_matrix.items():
    #         precisions = []
    #         recalls = []
    #         for iou_thr in iou_thresholds:  # Iterate over thresholds
    #             tp = thr_metrics[iou_thr]["true_positives"]
    #             fp = thr_metrics[iou_thr]["false_positives"]
    #             fn = thr_metrics[iou_thr]["false_negatives"]
    #
    #             if tp + fp > 0:
    #                 precision = tp / (tp + fp)
    #             else:
    #                 precision = 0
    #
    #             if tp + fn > 0:
    #                 recall = tp / (tp + fn)
    #             else:
    #                 recall = 0
    #
    #             precisions.append(precision)
    #             recalls.append(recall)
    #
    #         # Calculate AP for this alias
    #         ap = self.calc_ap(precisions, recalls)
    #
    #         alias_APs.setdefault(class_id, ap)
    #
    #     return alias_APs

    def calc_ap(self, precisions, recalls):
        """Calculates Average Precision (AP) using 11-point interpolation.

        Args:
            precisions (np.ndarray): Array of precision values at different confidence thresholds.
            recalls (np.ndarray): Array of recall values at different confidence thresholds.
            iou_thresholds (List[float]): List of IoU thresholds to evaluate.

        Returns:
            float: The calculated Average Precision (AP).
        """

        precisions.reverse()

        ap = 0
        for i in range(10):
            area = precisions[i] * recalls[i]
            ap += area

        return ap

    def evalias(
        self, predictions: List[Tuple], ground_truths: List[Tuple], class_id: int
    ):
        metrics = defaultdict(dict)

        # Organize predictions and ground truths by class for efficiency
        ground_truths_by_class = defaultdict(list)

        class_predictions = [bbox for _, bbox in predictions]
        for class_id, bbox in ground_truths:
            ground_truths_by_class[class_id].append(bbox)

        class_ground_truths = ground_truths_by_class.get(class_id, [])

        if not class_ground_truths:
            for i in range(0, 10):
                metrics[i]["false_positives"] = len(class_predictions)
        if not class_predictions:
            for i in range(0, 10):
                metrics[i]["false_negatives"] = len(class_ground_truths)

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
        for i in range(0, 10):
            matched_indices = torch.where(
                iou_matrix >= 0.1 * i
            )  # Assuming IoU threshold of 0.5

            # Count True Positives (TP) or the number of boxes that are matched and not overlapped
            tp_count = matched_indices[0].unique().numel()
            # Count False Positives (FP)
            fp_count = len(class_predictions) - tp_count
            # Count False Negatives (FN)
            fn_count = len(class_ground_truths) - tp_count

            metric = {
                "true_positives": tp_count,
                "false_positives": fp_count,
                "false_negatives": fn_count,
            }

            metrics[i] = metric

        return metrics

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
        """ example:
        metrics = {
        0 (class name): {
            1(iou threshold): {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
            },
            2(iou threshold): {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
            },
            3(iou threshold): {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
            },
        },
        1 (class name): {
            1(iou threshold): {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
            },
            2(iou threshold): {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
            },
            3(iou threshold): {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
            },
        },
        }
        """
        matched_boxes = {}

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
            for i in range(0, 10):
                matched_indices = torch.where(
                    iou_matrix >= 0.1 * i
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

                metric = {
                    # "matched": trues,
                    # "overlap": overlap_count,
                    "true_positives": tp_count,
                    # "unmatched": unmatched,
                    "false_positives": fp_count,
                    "false_negatives": fn_count,
                }
                metrics[idx].setdefault(i, metric)

                # Get the indices of the matching boxes
                if matched_indices[0].numel() > 0:
                    truth_indices, pred_indices = matched_indices
                    m_bbxs = (pr_bbxs[pred_indices], gr_bbxs[truth_indices])
                    matched_boxes.setdefault(idx, [])
                    matched_boxes[idx].append(m_bbxs)

        return metrics, matched_boxes

    def draw(self, matched_boxes, image_path, metrics):
        """
        Draw the matched boxes on the images
        """
        image = io.read_image(image_path)
        w, h = imagesize.get(image_path)

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
            preds = denormalize_boxes(predictions, (h, w))
            grounds = denormalize_boxes(ground_truths, (h, w))
            image = torchutils.draw_bounding_boxes(
                image, grounds, labels=None, colors=(0, 0, 0), width=1, fill=True
            )
            image = torchutils.draw_bounding_boxes(
                image,
                preds,
                labels=[self.dataset.classes[class_id] for _ in range(len(preds))],
                colors=colors[class_id],
                width=3,
                font_size=3,
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


def main():
    import utils
    import json
    from dino import Dino
    from autodistill.detection import CaptionOntology
    from dataset import Dataset

    utils.filter_warnings()

    # 1. Load the dataset
    dataset = Dataset(
        "/Users/francescotacinelli/Developer/datasets/pallets_sorted/test/"
    )
    cap_ont = dataset.captions.captions_ontology

    # 2. Load the model
    try:
        model = Dino(ontology=CaptionOntology(cap_ont))
    except Exception as e:
        raise e

    # 3. Benchmark the grounded model with the default aliases
    tuning = Tuning(model, dataset)
    tuning.tune()

    # # 4. Save the results
    # benchmark_path = (
    #     "/Users/francescotacinelli/Developer/theia/benchmarks/captions_benchmark.json"
    # )
    # results = {
    #     "captions": captions,
    #     "benchmark": captions_benchmark,
    # }
    # with open(benchmark_path, "w") as f:
    #     json.dump(results, f)

    # 6. Tune the model


if __name__ == "__main__":
    main()
