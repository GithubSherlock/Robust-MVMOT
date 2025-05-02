"""
COCO Evaluation Module
Implements evaluation metrics for object detection, segmentation and keypoint tasks
Author: Shiqi Jiang
Date: 2025-04-25
"""


import copy
import io
from contextlib import redirect_stdout
import logging
import numpy as np
import pycocotools.mask as mask_util
import torch

from utils import utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class CocoEvaluator:
    """
    COCO evaluation wrapper for multiple tasks (detection, segmentation, keypoints)
    
    Handles evaluation of model predictions against ground truth annotations using
    COCO metrics (AP, AR) across different IoU thresholds.
    
    Attributes:
        coco_gt: Ground truth COCO annotations
        writer: Optional tensorboard writer for logging metrics
        iou_types: Types of IoU calculations to perform
        coco_eval: Dictionary of COCO evaluators for each IoU type
    """
    def __init__(self, coco_gt, iou_types, writer=None):
        """
        Initialize COCO evaluator
        
        Args:
            coco_gt: COCO ground truth annotations
            iou_types: List of IoU types to evaluate
            writer: Optional tensorboard writer
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(f"This constructor expects iou_types of type list or tuple, instead  got {type(iou_types)}")
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.writer = writer
        self.iou_types = iou_types
        self.coco_eval = {}
        # Initialize evaluators for each IoU type
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        """
        Update evaluator with new predictions
        
        Args:
            predictions: Dictionary of predictions by image ID
            
        Time Complexity: O(N) where N = number of predictions
        Space Complexity: O(N)
        """
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)
        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            # Load predictions into COCO format
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)
            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        """
        Synchronize evaluation results between different processes in distributed training
        
        Concatenates evaluation images from all processes and creates a common
        evaluation environment to ensure consistent results across processes.
        
        Time Complexity: O(P * N) where:
            P = number of processes
            N = number of images per process
        Space Complexity: O(P * N)
        """
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        """
        Accumulate evaluation results for final metrics calculation
        
        Calls accumulate() on each COCO evaluator to compute final
        statistics across all evaluation results.
        
        Time Complexity: O(K * N) where:
            K = number of IoU types
            N = number of images
        Space Complexity: O(1)
        """
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    # def summarize(self):
    #     for iou_type, coco_eval in self.coco_eval.items():
    #         # print(f"IoU metric: {iou_type}")
    #         logging.info(f"IoU metric: {iou_type}")
    #         coco_eval.summarize()

    def summarize(self, global_step=None):
        """
        Summarize evaluation results and log metrics
        
        Args:
            global_step: Optional global step for tensorboard logging
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        for iou_type, coco_eval in self.coco_eval.items():
            logging.info(f"IoU metric: {iou_type}")
            coco_eval.summarize() # capture of all evaluation metrics
            stats = coco_eval.stats
            if stats is None or len(stats) == 0:
                logging.error("Stats array is empty after summarize()")
                continue
            if self.writer is not None:
                self.writer.add_scalar(f'{iou_type}/AP_all', stats[0], global_step)
                self.writer.add_scalar(f'{iou_type}/AP_50', stats[1], global_step)
                self.writer.add_scalar(f'{iou_type}/AP_75', stats[2], global_step)
                self.writer.add_scalar(f'{iou_type}/AP_small', stats[3], global_step)
                self.writer.add_scalar(f'{iou_type}/AP_medium', stats[4], global_step)
                self.writer.add_scalar(f'{iou_type}/AP_large', stats[5], global_step)
                self.writer.add_scalar(f'{iou_type}/AR_max1', stats[6], global_step)
                self.writer.add_scalar(f'{iou_type}/AR_max10', stats[7], global_step)
                self.writer.add_scalar(f'{iou_type}/AR_max100', stats[8], global_step)
                self.writer.add_scalar(f'{iou_type}/AR_small', stats[9], global_step)
                self.writer.add_scalar(f'{iou_type}/AR_medium', stats[10], global_step)
                self.writer.add_scalar(f'{iou_type}/AR_large', stats[11], global_step)
            
            logging.info(f"\n{'='*20} Evaluation Results ({iou_type}) {'='*20}")
            if global_step is not None:
                logging.info(f"Global Step: {global_step}")
            logging.info("\nAverage Precision:") # AP 指标记录
            logging.info(f"    Average Precision @[ IoU=0.50:0.95 ] = {stats[0]:.3f}")
            logging.info(f"    Average Precision @[ IoU=0.50      ] = {stats[1]:.3f}")
            logging.info(f"    Average Precision @[ IoU=0.75      ] = {stats[2]:.3f}")
            logging.info(f"    Average Precision @[ small  objects] = {stats[3]:.3f}")
            logging.info(f"    Average Precision @[ medium objects] = {stats[4]:.3f}")
            logging.info(f"    Average Precision @[ large  objects] = {stats[5]:.3f}")
            logging.info("\nAverage Recall:") # AR 指标记录
            logging.info(f"    Average Recall @[ max dets=1    ] = {stats[6]:.3f}")
            logging.info(f"    Average Recall @[ max dets=10   ] = {stats[7]:.3f}")
            logging.info(f"    Average Recall @[ max dets=100  ] = {stats[8]:.3f}")
            logging.info(f"    Average Recall @[ small  objects] = {stats[9]:.3f}")
            logging.info(f"    Average Recall @[ medium objects] = {stats[10]:.3f}")
            logging.info(f"    Average Recall @[ large  objects] = {stats[11]:.3f}")
            logging.info(f"{'='*60}\n")

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    """
    Merge evaluation results from multiple processes
    
    Args:
        img_ids: List of image IDs from current process
        eval_imgs: Evaluation results from current process
        
    Returns:
        tuple: (merged_img_ids, merged_eval_imgs)
        
    Time Complexity: O(P * N * log(P * N)) where:
        P = number of processes
        N = number of images per process
    Space Complexity: O(P * N)
    """
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    """
    Create a common COCO evaluation environment
    
    Args:
        coco_eval: COCO evaluator instance
        img_ids: List of image IDs
        eval_imgs: Evaluation results
        
    Time Complexity: O(N) where N = number of images
    Space Complexity: O(N)
    """
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(imgs):
    """
    Perform COCO evaluation
    
    Args:
        imgs: COCO evaluator instance
        
    Returns:
        tuple: (image_ids, evaluation_results)
        
    Time Complexity: O(N) where N = number of images
    Space Complexity: O(N)
    """
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))
