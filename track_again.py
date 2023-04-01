
import os
import cv2
import platform
import numpy as np
import datetime
import json
import ujson
import signal
import torch
import torch.backends.cudnn as cudnn
import time
from pathlib import Path

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.utils import LOGGER, colorstr
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes

from polaris.yolov8_tracking.trackers.multi_tracker_zoo import create_tracker
json_file, vid_writer = None, None
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes
json_track_out = {
    "info": {
        "contributor": "Polaris",
        "date_created": str(datetime.datetime.now()),
        "description": "",
        "url": "",
        "version": "",
        "year": ""
    },
    "objects_tracked": [],
    "images": [],
    "predictions": []    
}
json_file = None
@torch.no_grad()
def run(
        # source='./resources/night.avi',
        source='./images',
        yolo_weights=Path('polaris/model/model_weights.pt'),  # model.pt path(s),
        reid_weights=Path('polaris/model/osnet_x0_25_msmt17.pt'),  # model.pt path,
        outfolder=Path('out'),  # output folder path,
        tracking_method='strongsort',
        tracking_config=Path('strongsort.yaml'),
        imgsz=(416, 512),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=True,  # show results
        save_JSON=True, # save results to *.json
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=True,  # save confidences in --save-txt labels
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    global json_file, vid_writer
    source = str(source)

    # Load model
    device = select_device(device)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size
    print(imgsz)
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
    vid_writer = [None]
    
    json_file_path = str(outfolder / "track_output.json")
    with open(json_file_path, 'w') as file:
        file.write(json.dumps(json_track_out, indent = 4))
    json_file = open(json_file_path, 'r+')
    ujson_data = ujson.load(json_file)

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
                    ujson_data["images"].append(temp_image)
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

                    # if save_txt:
                    #     # to MOT format
                    #     bbox_left = output[0]
                    #     bbox_top = output[1]
                    #     bbox_w = output[2] - output[0]
                    #     bbox_h = output[3] - output[1]
                    #     # Write MOT compliant results to file
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                    #                                        bbox_top, bbox_w, bbox_h, -1, -1, -1, i))
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
                                "supercategory": o_cls,
                                "extra_dict": {}
                            }
                            ujson_data["objects_tracked"].append(temp_object)
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
                            "confidence": o_conf,
                            "extra_dict": {}
                        }
                        ujson_data["predictions"].append(temp_prediction)
                        

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
                json_file.seek(0)
                json_file.truncate()
                ujson.dump(ujson_data, json_file)

        else:
            pass
            #tracker.tracker.pred_n_update_all_tracks()
            
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
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()  # release previous video writer
            if vid_cap:  # video
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:  # stream
                fps, w, h = 30, current_frame.shape[1], current_frame.shape[0]
            vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(current_frame)

        prev_frame = current_frame
        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(f"{s}{'' if len(detections) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms, loop time is: {int((time.time()-start_loop)*1000)}")
        
    
    json_file.close()
    json_file = None
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((outfolder / 'tracks').glob('*.txt')))} tracks saved to {outfolder / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', outfolder)}{s}")
    # if update:
    #     strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)
def shutdown_function(signal, frame):
    global json_file, vid_writer
    print("Shutting down")
    if(json_file is not None):
        print("Closing JSON file")
        json_file.close()
        json_file = None
    if(vid_writer is not None):
        print("Closing video writer")
        vid_writer.release()
        vid_writer = None
    exit()

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, shutdown_function)
try:
    run()
except KeyboardInterrupt:
    pass
# Load the shutdown function and exit
shutdown_function(None, None)