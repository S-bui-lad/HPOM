import yaml
import cv2
from PIL import Image
import numpy as np
import os

from models.yolo_wrap import YOLOWrapper


def load_cfg(cfg_path="configs/pipeline.yaml"):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def detect_objects(image_path, cfg_path="configs/pipeline.yaml"):
    cfg = load_cfg(cfg_path)
    ycfg = cfg.get('yolo', {})

    # read image (BGR cv2)
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # create yolo wrapper
    yolo = YOLOWrapper(model_path=ycfg.get('model', 'yolov8n.pt'),
                       conf=ycfg.get('conf', 0.2),
                       iou=ycfg.get('iou', 0.6),
                       max_det=ycfg.get('max_det', 300))

    # YOLO wrapper expects RGB array in many implementations; our wrapper will handle
    boxes, scores, labels, names = yolo.detect(image_cv)

    # convert to PIL RGB
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    return {
        'boxes': np.array(boxes, dtype=np.float32),
        'scores': np.array(scores, dtype=float),
        'labels': np.array(labels, dtype=int),
        'names': names,
        'image_cv': image_cv,
        'image_pil': image_pil
    }