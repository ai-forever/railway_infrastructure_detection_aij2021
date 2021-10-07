from typing import Dict, Any
import time
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as pctmask
import numpy as np
import torch
from collections import defaultdict



PATH_TO_GT_ANN = "input/detection_gt.json"
PATH_TO_DT_ANN = "output/detection_predictions.json"

PATH_TO_GT_JSON = "input/segmentation_gt.json"
PATH_TO_PRED_JSON = "output/segmentation_predictions.json"


def calculation_map_050(path_to_gt_ann, path_to_dt_ann):
    annType = "bbox"
    cocoGt = COCO(path_to_gt_ann)
    cocoDt = cocoGt.loadRes(path_to_dt_ann)
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    map_050 = cocoEval.stats[1]
    return map_050


def get_mask_from_name(name: str, submit: Dict[str, Any]) -> np.ndarray:

    image_annots = [x for x in submit["images"] if x["file_name"] == name][0][
        "annotations"
    ]
    assert len(image_annots) == 3, "Invalid number of predicted classes"

    mask = []
    for class_id in (0, 1, 2):
        ann = [x for x in image_annots if x["class_id"] == class_id]
        mask.append(pctmask.decode(ann))

    result = np.zeros(shape=mask[0].shape)
    for cls_mask, cls_value in zip(mask, [1, 2, 3]):
        result[cls_mask != 0] = cls_value

    return result


def calculate_meanIOU(path_to_gt_json, path_to_predict_json):
    with open(path_to_gt_json, "r") as f:
        gt = json.load(f)

    with open(path_to_predict_json, "r") as f:
        predictions = json.load(f)

    eps = 1e-8
    inersection_per_class = defaultdict(int)
    union_per_class = defaultdict(int)

    for image_dict in tqdm(gt["images"]):

        gt_mask = get_mask_from_name(image_dict["file_name"], gt)
        pred_mask = get_mask_from_name(
            image_dict["file_name"], predictions)
        gt_mask = torch.tensor(gt_mask)
        pred_mask = torch.tensor(pred_mask)

        for class_id in (0, 1, 2, 3):
            y_pred_i = (pred_mask == class_id).float()
            y_true_i = (gt_mask == class_id).float()

            intersection = torch.sum(y_pred_i * y_true_i).item()
            cardinality = (torch.sum(y_pred_i) + torch.sum(y_true_i)).item()
            union = cardinality - intersection

            inersection_per_class[class_id] += intersection
            union_per_class[class_id] += union

    ious = []
    for class_id in (0, 1, 2, 3):
        score = (inersection_per_class[class_id]) / (union_per_class[class_id] + eps)
        ious.append(score)
    return np.mean(ious)


def competition_metric(
    path_to_gt_ann,
    path_to_dt_ann,
    path_to_gt_json,
    path_to_predict_json,
):
    map_050 = calculation_map_050(path_to_gt_ann, path_to_dt_ann)
    mean_iou = calculate_meanIOU(path_to_gt_json, path_to_predict_json)
    print(f"Detection: {map_050}")
    print(f"Segmentation: {mean_iou}")
    return round(float(0.7 * map_050 + 0.3 * mean_iou), 5)


if __name__ == "__main__":
    time1 = time.time()
    result = competition_metric(
        PATH_TO_GT_ANN,
        PATH_TO_DT_ANN,
        PATH_TO_GT_JSON,
        PATH_TO_PRED_JSON,
    )
    print("Metric result: ", result)
    time2 = time.time()
    print(time2 - time1, "ok")
