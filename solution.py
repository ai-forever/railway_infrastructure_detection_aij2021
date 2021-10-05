import os
import time
import torch

from detection.detection_predict import get_detection_solution
from segmentation.src.segmentation_inference import get_segmentation_solution

PATH_TO_DET_WEIGHTS = "models/detection_model.pt"
PATH_TO_SEGMENTATION_WEIGHTS = "models/segmentation_model.pt"
PATH_TO_TEST_IMAGES = "test/images/"

PATH_TO_PRED = "output/"
PATH_TO_DT_ANN = "output/detection_predictions.json"
PATH_TO_SEGMENTATION_ANN = "output/segmentation_predictions.json"


if __name__ == "__main__":
    t1 = time.time()
    if not os.path.exists(PATH_TO_PRED):
        os.mkdir(PATH_TO_PRED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Detection model inference
    get_detection_solution(
        PATH_TO_DET_WEIGHTS, PATH_TO_TEST_IMAGES, PATH_TO_DT_ANN, device, img_size=640
    )
    # Segmentation model inference 
    get_segmentation_solution(PATH_TO_TEST_IMAGES, PATH_TO_SEGMENTATION_ANN, device=device)
    t2 = time.time()
    print(t2 - t1, "ok")
