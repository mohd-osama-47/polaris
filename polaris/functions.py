#!/usr/bin/env python3

import os
import sys
import cv2
import json
from datetime import datetime
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.yolo.engine import results


# Get the path of this file and use it to find the model to load
cur_path = os.path.dirname(__file__)

# Load a model
model = YOLO(os.path.join(cur_path, 'model/model_weights.pt'))  # load the trained model

# For tracking:
# json_out = {
#     "info": {
#         "contributor": "Polaris",
#         "date_created": str(datetime.now()),
#         "description": "",
#         "url": "",
#         "version": "",
#         "year": ""
#     },
#     "objects_tracked": [],
#     "images": [],
#     "predictions": [],
# }


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
# For predictions:
json_out = {
    "info": {
        "contributor": "Polaris",
        "date_created": str(datetime.now()),
        "description": "",
        "url": "",
        "version": "",
        "year": ""
    },
    "categories": [
        {
            "id": 1,
            "name": "Person",
            "supercategory": ""
        },
        {
            "id": 2,
            "name": "offroad_vehicle",
            "supercategory": ""
        },
        {
            "id": 3,
            "name": "Motorcyclist",
            "supercategory": ""
        },
        {
            "id": 4,
            "name": "ATV driver",
            "supercategory": ""
        },
        {
            "id": 5,
            "name": "None",
            "supercategory": ""
        },
        {
            "id": 6,
            "name": "Car",
            "supercategory": ""
        },
        {
            "id": 7,
            "name": "Bus",
            "supercategory": ""
        },
        {
            "id": 8,
            "name": "Truck",
            "supercategory": ""
        },
    ],
    "images": [],
    "annotations": [],
}

'''
"images": [
    {
        "id": 1,
        "width": 640,
        "height": 512,
        "file_name": "image0001.png",
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": 0
    },
]
'''
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

# #! Template of json image out:
# temp_image = {
#         "id": 0,
#         "width": 0,
#         "height": 0,
#         "file_name": "",
#         "license": 0,
#         "flickr_url": "",
#         "coco_url": "",
#         "date_captured": 0
#     }
def get_preds(images: list, out:str, save_files:bool=False)->list:
    '''
    Runs YOLOv8 predictions on the supplied image directory and saves to where the user desires the output
    '''
    count = 1
    for image_num, image in tqdm(enumerate(images, start=1), total=len(images), unit="images", colour='green'):
        res = model.predict(image, verbose=False, stream=True, device="0")
        temp_image = {
            "id": image_num,
            "width": 640,
            "height": 512,
            "file_name": image.split("/")[-1],
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        }
        json_out["images"].append(temp_image)
        
        for i, r in enumerate(res,1):
            if (save_files):
                cv2.imwrite(os.path.join(out,r.path.split("/")[-1]), r.plot())
            
            for box in r.boxes:
                bbox=box.xyxy.tolist()[0]
                temp_pred = {
                "id": count,
                "image_id": image_num,
                "category_id": int(box.cls.item())+1,
                "bbox": [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],
                # "confidence": box.conf.item(),
                "extra_dict": {
                    "confidence": box.conf.item(),
                }
                }
                json_out["annotations"].append(temp_pred)
                count+=1
        
    with open(os.path.join(out,"output.json"), "w") as outfile:
        outfile.write(json.dumps(json_out, indent = 4))
        
        


# The following functionality is taken from the GitHub repo: 

'''
MIT License

Copyright (c) 2020 Mohamad Mansour

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
    same = set(o for o in shared_keys if d1[o] == d2[o])
    return added, removed, modified, same

def Repeat(x): 
    _size = len(x) 
    repeated = [] 
    for i in range(_size): 
        k = i + 1
        for j in range(k, _size): 
            if x[i] == x[j] and x[i] not in repeated: 
                repeated.append(x[i]) 
    return repeated
def testt(A):
    aa=[]
    for i in A:
      aa.append(i['id'])
    print("MAX {}".format(max(aa)))
    print("MIN {}".format(min(aa)))
    return Repeat(aa)


def combine(tt1,tt2,output_file):
    """ Combine two COCO annoatated files and save them into new file
    :param tt1: 1st COCO file path
    :param tt2: 2nd COCO file path
    :param output_file: output file path
    """
    with open(tt1) as json_file:
        d1 = json.load(json_file)
    with open(tt2) as json_file:
        d2 = json.load(json_file)
    b1={}
    for i,j in enumerate(d1['images']):
        b1[d1['images'][i]['id']]=i

    temp=[cc['file_name'] for cc in d1['images']]
    temp2=[cc['file_name'] for cc in d2['images']]
    for i in temp:
        assert not(i in temp2), "Duplicate filenames detected between the two files! @" + i
    

    # Check if both files have the categories dict using only the value and id to compare
    d1_categories_names = {c['name']: c['id'] for c in d1['categories']}
    d2_categories_names = {c['name']: c['id'] for c in d2['categories']}
    
    for c in d1_categories_names:
        # Check if the category name exists in the second file
        if c in d2_categories_names:
            # Check if the category id is the same
            if d1_categories_names[c] != d2_categories_names[c]:
                assert False, 'Category name: {}, id: {} in file 1 and {} in file 2'.format(c, d1_categories_names[c], d2_categories_names[c])
        else:
            assert False, 'Category name: {} in file 1 does not exist in file 2'.format(c)
    
    for c in d2_categories_names:
        if c in d1_categories_names:
            if d1_categories_names[c] != d2_categories_names[c]:
                assert False, 'Category name: {}, id: {} in file 1 and {} in file 2'.format(c, d1_categories_names[c], d2_categories_names[c])
        else:
            assert False, 'Category name: {} in file 2 does not exist in file 1'.format(c)



    files_check_classes={}
    for i,j in enumerate(d1['images']):
        for ii,jj in enumerate(d1['annotations']):
            if jj['image_id']==j['id']:
                try:
                    files_check_classes[j['file_name']].append(jj['category_id'])
                except:
                    files_check_classes[j['file_name']]=[jj['category_id']]

    for i,j in enumerate(d2['images']):
        for ii,jj in enumerate(d2['annotations']):
            if jj['image_id']==j['id']:
                try:
                    files_check_classes[j['file_name']].append(jj['category_id'])
                except:
                    files_check_classes[j['file_name']]=[jj['category_id']]

    b2={}
    for i,j in enumerate(d2['images']):
        b2[d2['images'][i]['id']]=i+max(b1)+1
        
    #Reset File 1 and 2 images ids
    for i,j in enumerate(d1['images']):
        d1['images'][i]['id']= b1[d1['images'][i]['id']]
    for i,j in enumerate(d2['images']):
        d2['images'][i]['id']= b2[d2['images'][i]['id']]
        
    #Reset File 1 and 2 annotations ids
    b3={}
    for i,j in enumerate(d1['annotations']):
        b3[d1['annotations'][i]['id']]=i
    b4={}
    for i,j in enumerate(d2['annotations']):
        b4[d2['annotations'][i]['id']]=max(b3)+i+1




    for i,j in enumerate(d1['annotations']):
        d1['annotations'][i]['id']= b3[d1['annotations'][i]['id']]
        d1['annotations'][i]['image_id']=b1[d1['annotations'][i]['image_id']]
    for i,j in enumerate(d2['annotations']):
        d2['annotations'][i]['id']= b4[d2['annotations'][i]['id']]
        d2['annotations'][i]['image_id']=b2[d2['annotations'][i]['image_id']]

    files_check_classes_temp={}
    pbar = tqdm(total=len(d1['images'])+len(d2['images']), colour='green')
    for i,j in enumerate(d1['images']):
        for ii,jj in enumerate(d1['annotations']):
            if jj['image_id']==j['id']:
                try:
                    files_check_classes_temp[j['file_name']].append(jj['category_id'])
                except:
                    pbar.update(1)
                    files_check_classes_temp[j['file_name']]=[jj['category_id']]


    for i,j in enumerate(d2['images']):
        for ii,jj in enumerate(d2['annotations']):
            if jj['image_id']==j['id']:
                try:
                    files_check_classes_temp[j['file_name']].append(jj['category_id'])
                except:
                    pbar.update(1)
                    files_check_classes_temp[j['file_name']]=[jj['category_id']]
    pbar.close()
    added, removed, modified, same = dict_compare(files_check_classes, files_check_classes_temp)
    assert (len(added)==0 and len(removed)==0 and len(modified)==0),"filenames detected before merging error: "+len(added)+" filenames added "+ len(removed)+" filenames removed "+len(modified)+" filenames' classes modified "+ len(same)+ " filenames entries reserved"

    test=d1.copy()
    for i in d2['images']:
        test['images'].append(i)
    for i in d2['annotations']:
        test['annotations'].append(i)
    test['categories']=d2['categories']
    files_check_classes_temp={}
    pbar = tqdm(total=len(test['images']), colour='green')
    for i,j in enumerate(test['images']):
        for ii,jj in enumerate(test['annotations']):
            if jj['image_id']==j['id']:
                try:
                    files_check_classes_temp[j['file_name']].append(jj['category_id'])
                except:
                    pbar.update(1)
                    files_check_classes_temp[j['file_name']]=[jj['category_id']]

    pbar.close()
    added, removed, modified, same = dict_compare(files_check_classes, files_check_classes_temp)
    assert (len(added)==0 and len(removed)==0 and len(modified)==0),"filenames detected after merging error: "+len(added)+" filenames added "+ len(removed)+" filenames removed "+len(modified)+" filenames' classes modified "+ len(same)+ " filenames entries reserved"

    with open(output_file, 'w') as f:
        json.dump(test,f)