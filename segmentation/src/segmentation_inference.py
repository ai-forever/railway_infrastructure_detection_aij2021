import json
from typing import List

import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from catalyst import dl
import pycocotools.mask as pctmask 

# custom modules
from segmentation.src.runner import CustomRunner
from segmentation.src.dataset import RZDDataset
from segmentation.src.transforms import get_infer_augs
from segmentation.src.utils import read_image, prepare_segmentation_submit

import os
BASE_DIR = "/home/jovyan"


def get_segmentation_solution(path_to_images: str, 
                              path_to_save: str, 
                              batch_size: int = 32,
                              num_workers: int = 15,
                              device: str = 'cpu'):

    print(device)
    image_paths = sorted(list(Path(path_to_images).glob('*.png')))

    dataset = RZDDataset(image_filenames=image_paths, 
                         transform=get_infer_augs(), 
                         read_image_fn=read_image)

    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=num_workers, 
                        prefetch_factor=10)

    # load model
    model = smp.Unet(encoder_name="resnet18",
                        encoder_weights=None,
                        in_channels=3,
                        classes=4,
                        activation=None)
    model.eval()
    model.to(device)

    checkpoint = torch.load(os.path.join(BASE_DIR, "models/segmentation_model.pth"), map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    runner = CustomRunner(
            input_key="image",
            output_key="scores",
            target_key="mask",
            loss_key="loss"
        )

    predicted_masks = [] 
    for prediction in tqdm(runner.predict_loader(loader=loader, model=model), desc='segmentation inference: '):
        predicted_masks.append(prediction["scores"].detach().cpu().numpy())


    predicted_masks = np.concatenate(predicted_masks, axis=0)
    predicted_masks = softmax(predicted_masks, axis=1)
    predicted_masks = np.moveaxis(predicted_masks, 1, -1)[...,1:] # 0 class is background, no need
    
    threshold = 0.9
    predicted_masks[predicted_masks > threshold] = 1
    predicted_masks[predicted_masks < threshold] = 0


    prepare_segmentation_submit(image_paths,
                                predicted_masks,
                                path_to_save)
