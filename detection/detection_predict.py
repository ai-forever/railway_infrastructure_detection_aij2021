import os
import json
import torch
import cv2
from tqdm import tqdm


from detection.yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from detection.yolov5.utils.augmentations import letterbox
from detection.yolov5.models.experimental import attempt_load


def preprocess(path_to_img, img_size, device):
    orig_img = cv2.imread(path_to_img)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img = letterbox(orig_img, img_size, auto=False)[0]
    img = img.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img).float().to(device)
    img_tensor /= 255
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, list(orig_img.shape), list(img_tensor.shape)


def postprocess(
    out,
    tensor_shape,
    orig_shape,
    path_to_img,
    conf_thres=0.001,
    iou_thres=0.6,
):
    out = non_max_suppression(out, conf_thres, iou_thres)
    img_predict = []
    for pred in out:
        pred[:, :4] = scale_coords(tensor_shape[2:], pred[:, :4], orig_shape).round()
        box = xyxy2xywh(pred[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2
        for p, b in zip(pred.tolist(), box.tolist()):
            img_predict.append(
                {
                    "image_id": int(path_to_img.split(".")[1]),
                    "category_id": int(p[5]) + 1,
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )
    return img_predict


def get_img_predict(model, path_to_img, device, img_size):
    img_tensor, orig_shape, tensor_shape = preprocess(path_to_img, img_size, device)
    with torch.no_grad():
        out, _ = model(img_tensor, augment=False)
    img_predict = postprocess(out, tensor_shape, orig_shape, path_to_img)
    return img_predict


def get_detection_solution(
    path_to_weights, path_to_test_images, path_to_dt_ann, device, img_size
):
    detect_model = attempt_load(path_to_weights, map_location=device)
    detect_model.eval()
    result = []
    for file_name in tqdm(os.listdir(path_to_test_images)):
        img_predict = get_img_predict(
            detect_model, os.path.join(path_to_test_images, file_name), device, img_size
        )
        result.extend(img_predict)
    with open(path_to_dt_ann, "w") as f:
        json.dump(result, f)
