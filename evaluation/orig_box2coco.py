import os
import uuid
import json


PATH_TO_GT_BOXES = "input/bboxes/"
PATH_TO_DET_ANN_TEMPLATE = "evaluation/det_ann_template.json"
PATH_TO_GT_ANN = "input/detection_gt.json"


def make_coco_detection_ann(path_to_bboxes, ann_template):

    class2id = {el["name"]: el["id"] for el in ann_template["categories"]}

    for file_name in os.listdir(path_to_bboxes):
        with open(os.path.join(path_to_bboxes, file_name), "r") as f:
            ann_json = json.load(f)
            ann_template["images"].append(
                {
                    "file_name": file_name[:-5],
                    "height": ann_json["img_size"]["height"],
                    "width": ann_json["img_size"]["width"],
                    "id": int(file_name.split(".")[1]),
                }
            )
            for box in ann_json["bb_objects"]:
                bbox = [
                    box["x1"],
                    box["y1"],
                    box["x2"] - box["x1"],
                    box["y2"] - box["y1"],
                ]
                ann_template["annotations"].append(
                    {
                        "file_name": file_name[:-5],
                        "segmentation": [
                            [
                                box["x1"],
                                box["y1"],
                                box["x1"],
                                box["y2"],
                                box["x2"],
                                box["y2"],
                                box["x2"],
                                box["y1"],
                            ]
                        ],
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "image_id": int(file_name.split(".")[1]),
                        "bbox": bbox,
                        "category_id": class2id[box["class"]],
                        "id": uuid.uuid1().fields[0],
                    }
                )
    return ann_template


def save_to_coco_format(path_to_det_ann_template, path_to_gt_boxes, path_to_gt_ann):
    with open(path_to_det_ann_template, "r") as f:
        ann_template = json.load(f)
    cocoGt = make_coco_detection_ann(path_to_gt_boxes, ann_template)
    with open(path_to_gt_ann, "w") as f:
        json.dump(cocoGt, f)


if __name__ == "__main__":
    save_to_coco_format(PATH_TO_DET_ANN_TEMPLATE, PATH_TO_GT_BOXES, PATH_TO_GT_ANN)
