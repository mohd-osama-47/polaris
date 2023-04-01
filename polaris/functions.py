#!/usr/bin/env python3



import os
import sys
import cv2
import json
import torch
import platform
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8_tracking'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH


from ultralytics.yolo.utils import LOGGER
from yolov8_tracking.trackers.multi_tracker_zoo import create_tracker
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.ops import Profile, scale_boxes, non_max_suppression




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
        
def run_tracker(images: list, out:str, save_files:bool=False)->list:
    '''
    Runs YOLOv8 tracker on the supplied image directory and saves to where the user desires the output
    '''
    stride, names, pt = model.model.stride, model.model.names, False
    imgsz = check_imgsz([640, 512], stride=stride)  # check image size
    count = 1
    bs = 1
    dataset = LoadImages(
        images,
        imgsz=imgsz,
        stride=stride,
        auto=pt,
        transforms=getattr(model.model, 'transforms', None),
        vid_stride=1
    )
    txt_path = [None] * bs
    # model.model.warmup(imgsz=(1 if pt or model.model.triton else bs, 3, *imgsz))  # warmup

    tracking_method='strongsort'
    tracking_config="strongsort.yaml"
    # reid_weights='model/osnet_x0_25_msmt17.pt'
    reid_weights = os.path.join(ROOT,'model/osnet_x0_25_msmt17.pt')
    device='0'
    half=False

    conf_thres=0.25  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    classes=None     # filter 
    agnostic_nms=False,  # class-agnostic NMS
    max_det=1000,  # maximum detections per image
    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * bs

    # Run tracking
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs
    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch
        visualize = False
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        
        # Inference
        with dt[1]:
            preds = model(im, augment=False, visualize=visualize)
        
        # Apply NMS
        with dt[2]:
            p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        # Process detections
        for i, det in enumerate(p):  # detections per image
            seen += 1
            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            # folder with imgs
            txt_file_name = p.parent.name  # get folder name containing current img
            save_path = str(out / p.parent.name)  # im.jpg, vid.mp4, ...
        
        curr_frames[i] = im0
        txt_path = str(out / 'tracks' / txt_file_name)  # im.txt
        s += '%gx%g ' % im.shape[2:]  # print string
        imc = im0  # for save_crop
        annotator = Annotator(im0, line_width=2, example=str(names))

        if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
            if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
        
        if det is not None and len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            
            # pass detections to strongsort
            with dt[3]:
                outputs[i] = tracker_list[i].update(det.cpu(), im0)

            # draw boxes for visualization
            if len(outputs[i]) > 0:
                for j, (output) in enumerate(outputs[i]):
                    bbox = output[0:4]
                    id = output[4]
                    cls = output[5]
                    conf = output[6]

                    #! TEMP
                    if False:
                        # to MOT format
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        # Write MOT compliant results to file
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                    if True:
                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        label = f'{id} {names[c]} {conf:.2f}'
                        color = colors(c, True)
                        annotator.box_label(bbox, label, color=color)
                        #! TEMP SAVE save_trajectories=False, save_crop=False
                        if False and tracking_method == 'strongsort':
                            q = output[7]
                            tracker_list[i].trajectory(im0, q, color=color)
                        if False:
                            txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                            save_one_box(np.array(bbox, dtype=np.int16), imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
        else:
            pass
            
        # Stream results
        im0 = annotator.result()
        if True:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                exit()
        
        prev_frames[i] = curr_frames[i]

        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")
    
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    
    s = f"\n{len(list((out / 'tracks').glob('*.txt')))} tracks saved to {out / 'tracks'}" 
    LOGGER.info(f"Results saved to {out}{s}")

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