#!/usr/bin/env python3



import os
import sys
import cv2
import time
import json
import torch
import signal
import platform
import numpy as np
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

import torch

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.utils import LOGGER, colorstr
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes

from polaris.yolov8_tracking.trackers.multi_tracker_zoo import create_tracker


json_file, vid_writer = None, None
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes


# Get the path of this file and use it to find the model to load
cur_path = os.path.dirname(__file__)

# Load a model
model = YOLO(os.path.join(cur_path, 'model/model_weights.pt'))  # load the trained model

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

# For tracking purposes
json_track_out = {
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
    "predictions": []    
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

@torch.no_grad()
def run_tracker(
        source='./resources/night.avi',
        # source='./images',
        yolo_weights=Path('polaris/model/model_weights.pt'),  # model.pt path(s),
        reid_weights=Path('polaris/model/osnet_x0_25_msmt17.pt'),  # model.pt path,
        outfolder=Path('out'),  # output folder path,
        tracking_method='strongsort',
        tracking_config=Path('strongsort.yaml'),
        imgsz=(640, 512),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=True,  # show results
        save_JSON=True, # save results to *.json
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        is_verbose=False
):
    global json_file, vid_writer
    source = str(source)

    # Load model
    try:
        device = select_device(device)
    except:
        device = select_device('')
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half, verbose=is_verbose)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size
    # Dataloader
    bs = 1
    dataset = LoadImages(
        source,
        imgsz=imgsz,
        stride=stride,
        auto=pt,
        transforms=getattr(model.model, 'transforms', None),
        vid_stride=vid_stride
    )
    vid_writer = None
    # if save_JSON:
        
    #     json_file_path = str(outfolder / "track_output.json")
    #     with open(json_file_path, 'w') as file:
    #         file.write(json.dumps(json_track_out, indent = 4))
    #     json_file = open(json_file_path, 'r+')
    #     ujson_data = ujson.load(json_file)

    vid_path = str(outfolder / "track_vid")  # im.jpg, vid.mp4, ...
    vid_path = str(Path(vid_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos

    model.warmup(imgsz=(1, 3, *imgsz))  # warmup

    tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
    if hasattr(tracker, 'model'):
        if hasattr(tracker.model, 'warmup'):
            tracker.model.warmup()
    

    # Run tracking
    #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, outputs, dt = 0, [None], (Profile(), Profile(), Profile(), Profile())
    current_frame, prev_frame = None, None
    prediction_number = 0
    final_tracked_object_id = 0
    for frame_idx, batch in enumerate(dataset):
        start_loop = time.time()
        path, im, current_frame, vid_cap, s = batch
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        
        # Inference
        with dt[1]:
            predictions = model(im, augment=False, visualize=False)

        # Apply NMS
        with dt[2]:
            predictionsNMS = non_max_suppression(predictions, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            detections= predictionsNMS[0]
        # Process detections
        seen += 1

        s += '%gx%g ' % im.shape[2:]  # print string
        imc = current_frame.copy() if save_crop else current_frame  # for save_crop

        annotator = Annotator(current_frame, line_width=line_thickness, example=str(names))
        
        if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'camera_update'):
            if prev_frame is not None and current_frame is not None:  # camera motion compensation
                tracker.tracker.camera_update(prev_frame, current_frame)

        if detections is not None and len(detections):
            detections[:, :4] = scale_boxes(im.shape[2:], detections[:, :4], current_frame.shape).round()  # rescale boxes to current_frame size

            # Print results
            for c in detections[:, 5].unique():
                n = (detections[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # pass detections to strongsort
            with dt[3]:
                outputs = tracker.update(detections.cpu(), current_frame)
            
            # draw boxes for visualization and output file to json
            if len(outputs) > 0:
                if(save_JSON):
                    #? Write the image to JSON
                    temp_image = {
                        "id": frame_idx,
                        "file_name": "image"+str(frame_idx),
                        "extra_dict": {}
                    }
                    json_track_out["images"].append(temp_image)
                    # #! TODO: Optimize the json read/write
                    # #? Read predictions last id from JSON
                    # try:
                    #     with open(json_file_path, 'r') as outfile:
                    #         count = json.load(outfile)["predictions"][-1]["id"]+1
                    # except:
                    #     count = 1
                
                for j, (output) in enumerate(outputs):
                    
                    bbox = output[0:4]
                    id = output[4]
                    cls = output[5]
                    conf = output[6]

                    if save_JSON:
                        o_id = int(output[4])
                        o_cls = int(output[5])
                        o_conf = float(output[6])

                        # try:
                        #     with open(json_file_path, 'r') as outfile:
                        #         object_count = json.load(outfile)["objects_tracked"][-1]["id"]
                        # except:
                        #     object_count = 0
                        if(o_id > final_tracked_object_id):
                            temp_object = {
                                "id": o_id,
                                "supercategory": CLASSES_DICT[o_cls],
                                "extra_dict": {}
                            }
                            json_track_out["objects_tracked"].append(temp_object)
                            final_tracked_object_id = o_id
                        # to MOT format
                        bbox_left = int(output[0])
                        bbox_top = int(output[1])
                        bbox_w = int(output[2] - output[0])
                        bbox_h = int(output[3] - output[1])
                        # Write JSON compliant results to file
                        prediction_number+=1
                        temp_prediction = {
                            "id": prediction_number,
                            "image_id": frame_idx,
                            "predicted_object_id": o_id,
                            "bbox": [bbox_left,bbox_top,bbox_w,bbox_h],
                            "track_id": o_id,
                            "confidence": o_conf,
                            "extra_dict": {}
                        }
                        json_track_out["predictions"].append(temp_prediction)
                        

                    if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                            (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                        color = colors(c, True)
                        annotator.box_label(bbox, label, color=color)
                        
                        if save_trajectories and tracking_method == 'strongsort':
                            q = output[7]
                            tracker.trajectory(current_frame, q, color=color)
                        if save_crop:
                            save_one_box(np.array(bbox, dtype=np.int16), imc, file=outfolder / 'crops' / "image" / names[c] / f'{id}.jpg', BGR=True)
                
                # with open(json_file_path, "w") as outfile:
                #     outfile.write(json.dumps(json_track_out, indent = 4))
                # if save_JSON:
                #     json_file.seek(0)
                #     json_file.truncate()
                #     ujson.dump(ujson_data, json_file)

        else:
            pass
            #tracker.tracker.pred_n_update_all_tracks()
        
        if save_JSON:
            
            json_file_path = str(outfolder / "track_output.json")
            with open(json_file_path, 'w') as file:
                file.write(json.dumps(json_track_out, indent = 4))
            # json_file = open(json_file_path, 'r+')
            # ujson_data = ujson.load(json_file)
            # ujson.dump(ujson_data, json_file)
        # Stream results
        current_frame = annotator.result()
        if show_vid:
            if platform.system() == 'Linux':
                cv2.namedWindow("showing video", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow("showing video", current_frame.shape[1], current_frame.shape[0])
            cv2.imshow("showing video", current_frame)
            if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                exit()

        # Save results (image with detections)
        if save_vid:
            if vid_writer == None:
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, current_frame.shape[1], current_frame.shape[0]
                vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                # vid_writer.release()  # release previous video writer
            vid_writer.write(current_frame)

        prev_frame = current_frame
        if is_verbose:
            # Print total time (preprocessing + inference + NMS + tracking)
            LOGGER.info(f"{s}{'' if len(detections) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms, loop time is: {int((time.time()-start_loop)*1000)}")
        
    # if save_JSON:
    #     json_file.close()
    #     json_file = None
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((outfolder / 'tracks').glob('*.txt')))} tracks saved to {outfolder / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', outfolder)}{s}")
    if save_vid:
        vid_writer.release()

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