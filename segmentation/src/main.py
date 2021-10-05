from typing import Any, Mapping
import numpy as np
from pathlib import Path
import argparse

from collections import OrderedDict
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from catalyst import dl
from sklearn.model_selection import train_test_split


import segmentation_models_pytorch as smp

from segmentation.src.runner import CustomRunner
from segmentation.src.dataset import RZDDataset
from segmentation.src.transforms import get_hard_augs, get_infer_augs
from segmentation.src.utils import read_image, read_mask
from segmentation.src.losses import FocalLoss, JaccardLoss


def main(args):
    train_batch_size = args.train_batch_size
    valid_batch_size = args.valid_batch_size

    data_path = Path(args.path_to_data)
    image_paths = sorted(
        list((data_path / 'images').glob('*.png')))
    mask_paths = sorted(list((data_path / 'masks').glob('*.png')))

    train_images, valid_images, train_masks, valid_masks = train_test_split(
        image_paths, mask_paths, test_size=0.33, random_state=42)

    train_dataset = RZDDataset(
        image_filenames=train_images,
        mask_filenames=train_masks,
        transform=get_hard_augs(),
        read_image_fn=read_image,
        read_mask_fn=read_mask,
    )

    valid_dataset = RZDDataset(
        image_filenames=valid_images,
        mask_filenames=valid_masks,
        transform=get_infer_augs(),
        read_image_fn=read_image,
        read_mask_fn=read_mask,
    )

    loaders = OrderedDict()

    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=10,
        prefetch_factor=15,
    )

    loaders["valid"] = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=10,
        prefetch_factor=15,
    )

    model = smp.Unet(encoder_name="resnet18",
                     encoder_weights='imagenet',
                     in_channels=3,
                     classes=4,
                     activation=None)

    criterion = {'focal': FocalLoss(mode='multiclass'),
                 'dice': JaccardLoss(mode='multiclass')
                 }

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1)

    runner = CustomRunner(
        input_key="image",
        output_key="scores",
        target_key="mask",
        loss_key="loss"
    )

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        num_epochs=args.num_epochs,
        callbacks=[
            dl.CriterionCallback(metric_key="focal_loss",
                                 input_key="scores",
                                 target_key="mask",
                                 criterion_key='focal'),
            dl.CriterionCallback(metric_key="dice_loss",
                                 input_key="scores",
                                 target_key="mask",
                                 criterion_key='dice'),
            dl.MetricAggregationCallback(metric_key="loss",
                                         metrics=["focal_loss", "dice_loss"],
                                         mode="sum"),
            dl.OptimizerCallback(metric_key="loss"),

            dl.IOUCallback(input_key="softm_preds",
                           target_key="one_hot_targets"),
        ],
        logdir=args.logdir,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AIJ baseline training')
    parser.add_argument('--path_to_data', default='./data', type=str, help='Path to directory with images and masks folders.')
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--valid_batch_size', default=16, type=int)
    parser.add_argument('--num_epochs', default=55, type=int)
    parser.add_argument('--logdir', default="./logdir/focal-dice_loss_baseline_rn18_4cl", type=str)
    args = parser.parse_args()
    main(args)
