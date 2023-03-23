#!/usr/bin/env python3

import os
import cv2
from ultralytics import YOLO

# Get the path of this file and use it to find the model to load
cur_path = os.path.dirname(__file__)

# Load a model
model = YOLO(os.path.join(cur_path, 'model/model_weights.pt'))  # load the trained model

def get_preds(image: str, out)->list:
    '''
    Runs YOLOv8 predictions on the supplied image directory and saves to where the user desires the output
    '''
    res = model.predict(image, verbose=False, stream=True)
    for r in res:
        cv2.imwrite(os.path.join(out,r.path.split("/")[-1]), r.plot())

