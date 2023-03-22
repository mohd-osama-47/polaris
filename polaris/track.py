#!/usr/bin/env python3

import os
import cv2
from ultralytics import YOLO

# Get the path of this file and use it to find the model to load
cur_path = os.path.dirname(__file__)

# Load a model
model = YOLO(os.path.join(cur_path, 'model/model_weights.pt'))  # load the trained model

model.track(source="night.avi", show=True, verbose=False)