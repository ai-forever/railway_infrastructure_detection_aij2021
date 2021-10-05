from typing import Callable, List, Optional, Union
from torch.utils.data import Dataset
import cv2

import albumentations as A
from pathlib import Path
from torch.utils.data import Dataset
from pytorch_toolbelt.datasets.common import read_image_rgb
from segmentation.src.utils import image_to_tensor


class RZDDataset(Dataset):
    """Base dataset for image segmentation"""

    def __init__(
        self,
        image_filenames: List[Union[str, Path]],
        transform: A.Compose,
        mask_filenames: Optional[List[Union[str, Path]]] = None,
        read_image_fn: Callable = read_image_rgb,
        read_mask_fn: Callable = cv2.imread,
        need_weight_mask=False,
        make_mask_target_fn: Callable = image_to_tensor,
    ):
        if mask_filenames is not None and len(image_filenames) != len(mask_filenames):
            raise ValueError(
                "Number of images does not corresponds to number of targets")

        self.need_weight_mask = need_weight_mask

        self.images = image_filenames
        self.masks = mask_filenames
        self.read_image = read_image_fn
        self.read_mask = read_mask_fn
        self.make_target = make_mask_target_fn
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.read_image(self.images[index])
        data = {"image": image}
        if self.masks is not None:
            data["mask"] = self.read_mask(self.masks[index])

        data = self.transform(**data)

        image = data["image"]
        sample = {
            'idx': index,
            'image': image_to_tensor(image),
        }

        if self.masks is not None:
            mask = data["mask"]
            sample['mask'] = self.make_target(mask)

        return sample
