import os
import uuid
import json


def make_coco_detection_ann(path_to_bboxes, ann_template):
    
    class2id = {el['name']: el['id'] for el in ann_template['categories']}
    
    for file_name in os.listdir(path_to_bboxes):
        with open(os.path.join(path_to_bboxes, file_name), 'r') as f:
            ann_json = json.load(f)
            ann_template['images'].append({
                "file_name": file_name[:-5],
                "height": ann_json['img_size']['height'],
                "width":ann_json['img_size']['width'],
                "id": int(file_name.split('.')[1]),
            })
            for box in ann_json['bb_objects']:
                bbox = [box['x1'], box['y1'],
                        box['x2'] - box['x1'], box['y2'] - box['y1']]
                ann_template['annotations'].append({
                    "file_name": file_name[:-5],
                    "segmentation":[[box['x1'], box['y1'],
                                     box['x1'], box['y2'],
                                     box['x2'], box['y2'],
                                     box['x2'], box['y1'],

                                     ]],
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "image_id": int(file_name.split('.')[1]),
                    "bbox": bbox,
                    "category_id": class2id[box['class']],
                    "id": uuid.uuid1().fields[0],
                })
    return ann_template
