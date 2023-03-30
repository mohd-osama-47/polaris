#!/usr/bin/env python3

import os
import sys
import cv2
import json
from tqdm import tqdm
from datetime import datetime, date
from ultralytics import YOLO


json_out = {
    "info": {
        "contributor": "Polaris",
        "date_created": str(datetime.now()),
        "description": "",
        "url": "",
        "version": "",
        "year": ""
    },
    "objects_tracked": [],
    "images": [],
    "predictions": [],
}

# #! Template of json detection out:
# temp_pred = {
#     "id": None,
#     "image_id": None,
#     "predicted_object_id": None,
#     "bbox": [
#         None,   #Top Left X
#         None,   #Top Left Y
#         None,   #Width
#         None    #Height
#     ],
#     "confidence": 0.87,
#     "extra_dict": {}
# }

# Get the path of this file and use it to find the model to load
cur_path = os.path.dirname(__file__)

# Load a model
model = YOLO(os.path.join(cur_path, 'polaris/model/model_weights.pt'))  # load the trained model


def get_preds(image: str, out)->list:
    '''
    Runs YOLOv8 predictions on the supplied image directory and saves to where the user desires the output
    '''
    res = model.predict(image, verbose=False, stream=True)
    for r in res:
        cv2.imwrite(os.path.join(out,r.path.split("/")[-1]), r.plot())
        os.mkdir(os.path.join(out,r.path.split("/")[-1]))
    
res = model.predict("images/1.png", verbose=False)
for i, r in enumerate(res):
    # print(r)
    for box in r.boxes:
        temp_pred = {
        "id": i,
        "image_id": 1,
        "predicted_object_id": CLASSES_DICT[box.cls.item()],
        "bbox": box.xywh.tolist()[0] ,
        "confidence": box.conf.item(),
        "extra_dict": {}
        }
        json_out["predictions"].append(temp_pred)

print(json.dumps(json_out, indent = 4))
