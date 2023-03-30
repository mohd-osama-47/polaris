#!/usr/bin/env python3

import os
import re
import cv2
import json
from tqdm import tqdm
from ultralytics import YOLO
from datetime import datetime

CLASSES_DICT = {
    0:"Person",
    1:"offroad_vehicle",
    2:"Motorcyclist",
    3:"ATV driver",
    4:"None",
    5:"Car",
    6:"Bus",
    7:"Truck"
}

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

'''
objects_tracked:
{
    "id": number,
    "supercategory": name,
    "extra_dict": {}
},
images format:
{
    "id": ImageID,
    "file_name": "IMAGE NAME",
    "extra_dict": {}
},

prediction format:
{
    "id": count,
    "image_id": image,
    "predicted_object_id": TrackedId,
    "bbox": [
        "top_left_x",
        "top_left_y",
        "width",
        "height"
    ],
    "confidence": 0,
    "extra_dict": {}
},
'''

VALID_IMAGES = ["jpg","png"]

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_images(folder:str):
    """Get all images that can be manipulated in the passed folder ONLY"""
    for item in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, item)):
            try:
                if item.split(".")[1].lower() not in VALID_IMAGES:
                    continue
            except IndexError:
                continue
            yield os.path.abspath(os.path.join(folder, item))

# Get the path of this file and use it to find the model to load
cur_path = os.path.dirname(__file__)

# Load a model
model = YOLO(os.path.join(cur_path, 'model/model_weights.pt'))  # load the trained model

img_list = []
for filepath in tqdm(get_images("images/"), unit="image", colour='green'):
    img_list.append(filepath)

img_list = natural_sort(img_list)

res = model.track(source=img_list, verbose=False, tracker="bytetrack.yaml", stream=True, show=True)
count=1
for i, r in enumerate(res,1):
    # print("ID",i,":",r)
    temp_image = {
        "id": i,
        "file_name": str(r.path),
        "extra_dict": {}
    }
    json_out["images"].append(temp_image)
    for box in r.boxes:
        bbox=box.xyxy.tolist()[0]
        temp_pred = {
            "id": count,
            "image_id": i,
            "predicted_object_id": int(box.id.item()),
            "bbox": [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],
            "confidence": box.conf.item(),
            "extra_dict": {
                "category_id": int(box.cls.item())+1,
        }
        }
        count+=1
        json_out["predictions"].append(temp_pred)

with open(os.path.join(".","outputVID.json"), "w") as outfile:
        outfile.write(json.dumps(json_out, indent = 4))

cv2.waitKey(0)