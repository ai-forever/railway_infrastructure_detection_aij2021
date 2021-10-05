import os
import json
from tqdm import tqdm
from PIL import Image
from shutil import copyfile


def makedirs(path_to_save_labels, path_to_images, dataset_type2image):
    for split_type in dataset_type2image.keys():
        if not os.path.exists(os.path.join(path_to_save_labels, split_type)):
            os.makedirs(os.path.join(path_to_save_labels, split_type))
        if not os.path.exists(os.path.join(path_to_images, split_type)):
            os.makedirs(os.path.join(path_to_images, split_type))


def get_yolo_labels(
    path_to_bboxes, path_to_save_labels, path_to_images, image2dataset_type, class_names
):
    for file_name in os.listdir(path_to_bboxes):
        with open(os.path.join(path_to_bboxes, file_name), "r") as f:
            img_boxes = json.load(f)
        img_name = file_name[:-5]
        im = Image.open(os.path.join(path_to_images, img_name))
        im_width, im_height = im.size
        split_type = image2dataset_type[img_name]
        if len(img_boxes["bb_objects"]) > 0:
            for box in img_boxes["bb_objects"]:
                x_center = ((box["x1"] + box["x2"]) / 2) / im_width
                y_center = ((box["y1"] + box["y2"]) / 2) / im_height
                width = (box["x2"] - box["x1"]) / im_width
                height = (box["y2"] - box["y1"]) / im_height
                label_class = class_names.index(box["class"])
                with open(
                    os.path.join(
                        path_to_save_labels, split_type, img_name[:-4] + ".txt"
                    ),
                    "a+",
                ) as f:
                    f.write(f"{label_class} {x_center} {y_center} {width} {height}\n")

        # Если не хотите пропускать семплы без разметки
        else:
            open(
                os.path.join(path_to_save_labels, split_type, img_name[:-4] + ".txt"),
                "a",
            ).close()


def copy_images(images_name, image2dataset_type, path_to_images):
    for img_path in tqdm(images_name):
        split_type = image2dataset_type[img_path]
        try:
            copyfile(
                os.path.join(path_to_images, img_path),
                os.path.join(path_to_images, split_type, img_path),
            )
        except IsADirectoryError:
            continue