import base64
import urllib.request
from pathlib import Path
from PIL import Image
import io

import numpy as np
import torch
from cv2 import cv2

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device


class PythonPredictor:

    def __init__(self, config):
        urllib.request.urlretrieve("https://github.com/ultralytics/yolov5/releases/download/v3.1/yolov5s.pt", "yolov5s.pt")

    def predict(self, payload):
        """ Model Run function """

        im = Image.open(io.BytesIO(base64.b64decode(payload["base64"])))
        im.save('image.png', 'PNG')

        # Initialize
        device = select_device()

        # Load model
        model = attempt_load("./yolov5s.pt", map_location=device)
        imgsz = check_img_size(640, s=model.stride.max())  # check img_size

        dataset = LoadImages('./image.png', img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device).float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

            det = pred[0]

            p, s, im0, frame = Path(path), '', im0s, getattr(dataset, 'frame', 0)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            img = Image.fromarray(im0)

            im_file = io.BytesIO()
            img.save(im_file, format="PNG")
            im_bytes = base64.b64encode(im_file.getvalue()).decode("utf-8") 

            return im_bytes
