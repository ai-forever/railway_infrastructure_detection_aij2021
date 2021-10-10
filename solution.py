import os
import time
import torch

from detection.detection_predict import get_detection_solution
from segmentation.src.segmentation_inference import get_segmentation_solution

BASE_DIR = "/home/jovyan"

PATH_TO_DET_WEIGHTS = os.path.join(BASE_DIR, "models/detection_model.pt")
PATH_TO_SEGMENTATION_WEIGHTS = os.path.join(BASE_DIR, "models/segmentation_model.pt")
PATH_TO_TEST_IMAGES = os.path.join(BASE_DIR, "input/images/")

PATH_TO_PRED = os.path.join(BASE_DIR, "output/")
PATH_TO_DT_ANN = os.path.join(BASE_DIR, "output/detection_predictions.json")
PATH_TO_SEGMENTATION_ANN = os.path.join(BASE_DIR, "output/segmentation_predictions.json")


if __name__ == "__main__":
    t1 = time.time()
    if not os.path.exists(PATH_TO_PRED):
        os.mkdir(PATH_TO_PRED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Detection model inference
    get_detection_solution(
        PATH_TO_DET_WEIGHTS, PATH_TO_TEST_IMAGES, PATH_TO_DT_ANN, device, img_size=1280
    )
    # Segmentation model inference 
    get_segmentation_solution(PATH_TO_TEST_IMAGES, PATH_TO_SEGMENTATION_ANN, device=device)
    t2 = time.time()
    print(t2 - t1, "ok")
