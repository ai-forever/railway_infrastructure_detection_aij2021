from pathlib import Path
from typing import List, Union, Mapping, Sequence
import torch
from tqdm import tqdm
import numpy as np
import cv2
import json
import pycocotools.mask as pctmask

def image_to_tensor(image: np.ndarray, dummy_channels_dim=True) -> torch.Tensor:
    """
    Reference: https://github.com/BloodAxe/pytorch-toolbelt/blob/398e8c833bf4af0f345fef92366f435c56d83737/pytorch_toolbelt/utils/torch_utils.py#L146
    Convert numpy image (RGB, BGR, Grayscale, SAR, Mask image, etc.) to tensor
    Args:
        image: A numpy array of [H,W,C] shape
        dummy_channels_dim: If True, and image has [H,W] shape adds dummy channel, so that
            output tensor has shape [1, H, W]
    See also:
        rgb_image_from_tensor - To convert tensor image back to RGB with denormalization
        mask_from_tensor
    Returns:
        Torch tensor of [C,H,W] or [H,W] shape (dummy_channels_dim=False).
    """
    if len(image.shape) not in {2, 3}:
        raise ValueError(
            f"Image must have shape [H,W] or [H,W,C]. Got image with shape {image.shape}")

    if len(image.shape) == 2:
        if dummy_channels_dim:
            image = np.expand_dims(image, 0)
    else:
        # HWC -> CHW
        image = np.moveaxis(image, -1, 0)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    return image


def read_image(path: Path):
    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_mask(path: Path) -> np.ndarray:
    
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    for idx, v in enumerate((0, 6, 7, 10)):
        mask[mask == v] = idx
    return mask


def prepare_segmentation_submit(image_paths: List[Path], 
                                predicted_masks: np.ndarray, 
                                path_to_save: str) -> None:
    """Function to prepare json with encoded masks, ready for evaluation. 
       !Note!: Each N_i mask of predicted_masks have to correspond to N_i image name in image_filenames list!  

    Args:
        image_paths (List[Path]): N paths to images used for an inference.
        predicted_masks (np.ndarray): numpy array of (N, height, width, 3 (corresponds to number of classes)). Proper width and height will be taken from original images.
        path_to_save (str): where to store json with predictions
    """

    # Pay attention, your model may output in another order, but class index have to match with this structure
    submit = {
        "images": [],
        "categories": [
            {"supercategory": "railway_object","id": 0,"name": "MainRailPolygon"},
            {"supercategory": "railway_object","id": 1,"name": "AlternativeRailPolygon"},
            {"supercategory": "railway_object","id": 2,"name": "Train"},
        ]
    }
    image_filenames = []
    masks_list = []
    for idx, path in enumerate(image_paths):
        image_filenames.append(path.name)
        im_height, im_width = read_image(path).shape[:2]
        image_annots = []
        for cls_mask in (0,1,2):
            _single_mask = predicted_masks[idx, ..., cls_mask]
            _single_mask = cv2.resize(_single_mask, (im_width, im_height), interpolation=cv2.INTER_NEAREST)
            encoded = pctmask.encode(np.asfortranarray(_single_mask).astype(np.uint8))
            # Save class_id here
            encoded["class_id"] = cls_mask 
            image_annots.append(encoded)
        masks_list.append(image_annots)

    path_to_save = Path(path_to_save)

    for filename, mask in tqdm(zip(image_filenames, masks_list), total=len(image_filenames), desc='segmentation submit preparation: '):

        assert len(mask) == 3, 'Mask have to contain predicts for all 3 classes.'

        image_annots = []
        for class_id in (0, 1, 2):
            encoded = mask[class_id]
            image_annots.append({
                "counts": encoded['counts'].decode('ascii'),
                "size": encoded['size'],
                'class_id': encoded['class_id']

            })
           
        submit['images'].append({
            "file_name": filename,
            "annotations": image_annots
            }
        )
        with open(path_to_save, "w") as json_file:
            json.dump(submit, json_file)