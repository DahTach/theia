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
from transformers.configuration_utils import re
from guessing import Guesser
import numpy as np

"""
Steps:
- benchmark the grounded model with the default aliases
- changing aliases one category at a time
- use the benchmarking results to determine the best aliases
- keep doing this until we find the best aliases
"""


def get_confMatr(predictions: List[Tuple], ground_truths: List[Tuple], class_id: int):
    class_prs = [bbox for _, bbox in predictions]
    class_grs = [bbox for idx, bbox in ground_truths if idx == class_id]

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    print("class_prs", class_prs, "class_grs", class_grs)

    if class_grs and class_prs:
        # Convert lists of boxes to tensors
        gr_bbxs = ops.box_convert(
            boxes=torch.tensor(class_grs),
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )

        pr_bbxs = torch.stack(class_prs, dim=0)

        # Find matching pairs (if any) based on IoU
        iou_matrix = ops.box_iou(gr_bbxs, pr_bbxs)  # Efficient IoU calculation
        matched_indices = torch.where(
            iou_matrix >= 0.5
        )  # Assuming IoU threshold of 0.5

        true_positives = matched_indices[0].unique().numel()

        false_positives = len(class_prs) - true_positives

        false_negatives = len(class_grs) - true_positives
    elif not class_prs:
        false_negatives = 1
    elif not class_grs:
        false_positives = 1

    return true_positives, false_positives, false_negatives


class Tuning:
    def __init__(self):
        self.benchmarks = []

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

    def compare(
        self, predictions: List[Tuple], ground_truths: List[Tuple], class_ids: List[int]
    ):
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

        # Organize predictions and ground truths by class for efficiency
        predictions_by_class = defaultdict(list)
        ground_truths_by_class = defaultdict(list)

        for class_id, bbox in predictions:
            predictions_by_class[class_id].append(bbox)
        for class_id, bbox in ground_truths:
            ground_truths_by_class[class_id].append(bbox)

        # Convert to tensors once per class for faster calculations
        for idx in class_ids:
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

    def draw(self, matched_boxes, image_path, metrics, classes: List[str]):
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
                labels=[classes[class_id] for _ in range(len(preds))],
                colors=colors[class_id],
                width=3,
                font_size=3,
            )

        # image = image[0].permute(1, 2, 0).numpy()
        image = image.permute(1, 2, 0).numpy()
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        # draw the metrics on the bottom left corner of the image
        for idx, metric in metrics.items():
            class_name = classes[idx]
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


if __name__ == "__main__":
    main()
