import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from utils.datasets import LoadStreams, LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from backend import xyxy_to_xywh, draw_boxes, load_classes

def detect():
    with torch.no_grad():
        # Obtain parent folder to start app from any folder
        source_path = Path(__file__).resolve()
        source_dir = source_path.parent

        # Get arguments from command line
        opt_device = '0'
        opt_trailslen = 32
        source = "D:\Videos_Drones\DJI_0829.MOV"
        source_type = 'video'
        out = "D:\Videos_Drones\output"
        cfg = f'{source_dir}/cfg/yolor_p6.cfg'
        weights = f'{source_dir}/best_drones.pt'
        imgsz = 1280
        names_file = f'{source_dir}/data/drones.names'
        view_img = True
        save_img = True
        frame_save = 300

        # Define source as "webcam" if source is not video file
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # initialize deepsort
        cfg_deep = get_config()
        cfg_deep.merge_from_file(f"{source_dir}/deep_sort_pytorch/configs/deep_sort.yaml")
        reid_ckpt_path = f'{source_dir}/{cfg_deep.DEEPSORT.REID_CKPT}'
        deepsort = DeepSort(reid_ckpt_path, # cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # Initialize device
        device = select_device(opt_device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = Darknet(cfg, imgsz).cuda()
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
        model.to(device).eval()
        if half:
            model.half()  # to FP16

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            # view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            # save_img = True
            dataset = LoadImages(source, img_size=imgsz, auto_size=64)

        # Get names
        names = load_classes(names_file)
        
        # Run inference
        frame_num = 0
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        prevTime = 0
        save_path = ''
        csv_path = ''
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]

            # Apply Non-Maximum Supression (NMS)
            conf_thres = 0.5
            iou_thres = 0.5
            classes = [0,1,2,3,5,7] # Filtro de clases
            pred = non_max_suppression(pred, conf_thres, iou_thres, False, classes, False)
            t2 = time_synchronized()

            # Output files depending on source type
            if source_type == 'video':
                save_path = f'{Path(out)}.avi'
                csv_path = f'{Path(out)}.csv'
            elif source_type == 'camera':
                if frame_num == 0 or frame_num % frame_save == 0:
                    l_time = time.localtime(time.time())

                    l_year = f'{l_time[0]}'
                    l_month = f'{l_time[1]}' if int(l_time[1]) > 9 else f'0{l_time[1]}'
                    l_day = f'{l_time[2]}' if int(l_time[2]) > 9 else f'0{l_time[2]}'
                    l_hour = f'{l_time[3]}' if int(l_time[3]) > 9 else f'0{l_time[3]}'
                    l_min = f'{l_time[4]}' if int(l_time[4]) > 9 else f'0{l_time[4]}'
                    l_sec = f'{l_time[5]}' if int(l_time[5]) > 9 else f'0{l_time[5]}'

                    time_str = f'{l_year}_{l_month}_{l_day}_{l_hour}_{l_min}_{l_sec}'
                    save_path = f'{Path(out)}_{time_str}.avi'
                    csv_path = f'{Path(out)}_{time_str}.csv'

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    xywh_bboxs = []
                    confs = []
                    oids = []
                    for *xyxy, conf, cls in det:
                        # to deep sort format
                        x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                        xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                        xywh_bboxs.append(xywh_obj)
                        confs.append([conf.item()])
                        oids.append(int(cls))

                    xywhs = torch.Tensor(xywh_bboxs)
                    confss = torch.Tensor(confs)
                    
                    outputs = deepsort.update(xywhs, confss, oids, im0)
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -2]
                        object_id = outputs[:, -1]

                        draw_boxes(im0, bbox_xyxy, object_id, identities, csv_path, frame_num, names, opt_trailslen)
                        
                # Print time (inference + NMS)
                print(f' Done. ({(t2 - t1):.3f} s)')

                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime
                
                # View live results
                if view_img:
                    cv2.imshow(p, im0)
                    
                # Save image results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)
            
            if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
                break
            frame_num += 1

        print('Done. (%.3fs)' % (time.time() - t0))
