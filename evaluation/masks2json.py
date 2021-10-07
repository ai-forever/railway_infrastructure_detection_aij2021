from typing import Dict, Any
import json
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
import pycocotools.mask as pctmask

from segmentation.src.utils import read_mask

def mask2rle(path_to_mask: Path) -> Dict[str, Any]:
    name = path_to_mask.name
    
    raw_mask = read_mask(path_to_mask)
    
    result = {
        "file_name": name,
        "annotations": []
    }
    for class_id, class_value in enumerate((1, 2, 3)):
        _single_mask = raw_mask == class_value
        encoded = pctmask.encode(np.asfortranarray(_single_mask).astype(np.uint8))
        encoded["class_id"] = class_id
        encoded['counts'] = encoded['counts'].decode('ascii')
        result['annotations'].append(encoded)
    return result

def convert_masks2rle_json(path_to_masks: str, path_to_save: str) -> None:
    
    result = {
        "images": [],
        "categories": [
            {"supercategory": "railway_object","id": 0,"name": "MainRailPolygon"},
            {"supercategory": "railway_object","id": 1,"name": "AlternativeRailPolygon"},
            {"supercategory": "railway_object","id": 2,"name": "Train"},
        ]
    }
    
    mask_paths = sorted(list(Path(path_to_masks).glob('*.png')))
    
    for mask_pth in tqdm(mask_paths):
        result["images"].append(mask2rle(mask_pth))
    with open(path_to_save, 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_masks', default='./input/masks', type=str)
    parser.add_argument('--path_to_save', default='./input/segmentation_gt.json', type=str)
    args = parser.parse_args()

    convert_masks2rle_json(args.path_to_masks, 
                           args.path_to_save)