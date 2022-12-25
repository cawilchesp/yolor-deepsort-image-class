import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn

from yolor.utils.datasets import letterbox
from yolor.utils.general import non_max_suppression, scale_coords
from yolor.utils.torch_utils import select_device, time_synchronized
from yolor.models.models import Darknet

from deep_sort_pytorch.deep_sort import DeepSort

from collections import deque
from icecream import ic

class YOLOR_DEEPSORT:
    def __init__(self, yolor_options: dict, video_options: dict) -> None:
        with torch.no_grad():
            # Get parent folder
            yolor_folder = Path(__file__).resolve()
            yolor_path = yolor_folder.parent

            self.data_deque = {}

            # -----------------
            # Detector Settings
            # -----------------
            # YOLOR Options
            model_cfg = f'{yolor_path}/yolor/cfg/{yolor_options["cfg"]}'
            weights = f'{yolor_path}/weights/{yolor_options["weights"]}'
            inference_size = yolor_options['inference_size']
            class_names_file = f'{yolor_path}/yolor/data/{yolor_options["names_file"]}'
            use_gpu = yolor_options['use_gpu']

            # Initialize device
            if use_gpu:  opt_device = '0'
            else : opt_device = 'cpu'
            device = select_device(opt_device)
            half = device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = Darknet(model_cfg, inference_size).cuda()
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
            model.to(device).eval()
            if half:
                model.half()  # to FP16

            # Get class names
            class_names = self.load_classes(class_names_file)

            # --------------------------------
            # Tracker Deep-SORT Initialization
            # --------------------------------
            deepsort = DeepSort(model_path = f'{yolor_path}/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
                                max_dist = 0.2, min_confidence = 0.3, nms_max_overlap = 0.5, max_iou_distance = 0.7,
                                max_age = 10, n_init = 3, nn_budget = 100, use_cuda = True)

            # --------------------------
            # Video Input/Output Options
            # --------------------------
            source = video_options['source']
            output_path = video_options['output']
            frame_save = video_options['frame_save']
            trail_length = video_options['trail']
            class_filter = video_options['class_filter']
            show_boxes = video_options['show_boxes']
            show_trajectories = video_options['show_trajectories']
            view_image = video_options['view_image']
            save_text = video_options['save_text']
            save_video = video_options['save_video']
            
            # ----------------------------
            # Video Capture Initialization
            # ----------------------------
            # Define source as "webcam" if source is not video file
            webcam = source == 0 or source.startswith('rtsp') or source.startswith('http')

            # Set Dataloader
            if webcam:
                cudnn.benchmark = True  # set True to speed up constant image size inference
                auto_size=32
            else:
                auto_size=64

            vid_cap = cv2.VideoCapture(source)
            assert vid_cap.isOpened(), 'Failed to open source'
            fourcc = 'mp4v'  # output video codec
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            frame_number = 0
            vid_writer = None
            save_path = None
            csv_path = None

            t0 = time.time()
            
            inference_image = torch.zeros((1, 3, inference_size, inference_size), device=device)  # init inference_image
            _ = model(inference_image.half() if half else inference_image) if device.type != 'cpu' else None  # run once
            
            # -------------
            # Run Inference
            # -------------
            while (vid_cap.isOpened()):
                ret, frame_image = vid_cap.read()
                output_image = frame_image.copy()

                # Change frame size to inference size
                inference_image = letterbox(output_image, new_shape=inference_size, auto_size=auto_size)[0]
                inference_image = inference_image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                inference_image = np.ascontiguousarray(inference_image)

                # Frame pre-processing
                inference_image = torch.from_numpy(inference_image).to(device)
                inference_image = inference_image.half() if half else inference_image.float()  # uint8 to fp16/32
                inference_image /= 255.0  # 0 - 255 to 0.0 - 1.0
                if inference_image.ndimension() == 3:
                    inference_image = inference_image.unsqueeze(0)

                # Inference
                t1 = time_synchronized()

                detections = model(inference_image, augment=False)[0]
                detections = non_max_suppression(prediction=detections, conf_thres=0.5, iou_thres=0.5, merge=False, classes=class_filter, agnostic=False)
                
                t2 = time_synchronized()

                # Save files names
                if webcam:
                    if frame_number == 0 or frame_number % frame_save == 0:
                        time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
                        save_path = f'{Path(output_path)}_{time_str}.avi'
                        csv_path = f'{Path(output_path)}_{time_str}.csv'
                else:
                    save_path = f'{Path(output_path)}.avi'
                    csv_path = f'{Path(output_path)}.csv'

                # Process detections
                for det in detections:  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to output_image size
                        det[:, :4] = scale_coords(inference_image.shape[2:], det[:, :4], output_image.shape).round()

                        # Write results
                        xywh_bboxs = []
                        confs = []
                        oids = []
                        for *xyxy, conf, cls in det:
                            # to deep sort format
                            x_c, y_c, bbox_w, bbox_h = self.xyxy_to_xywh(*xyxy)
                            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                            xywh_bboxs.append(xywh_obj)
                            confs.append([conf.item()])
                            oids.append(int(cls))

                        xywhs = torch.Tensor(xywh_bboxs)
                        confss = torch.Tensor(confs)
                        
                        outputs = deepsort.update(xywhs, confss, oids, output_image)
                        
                        # Results presentation
                        if len(outputs) > 0:
                            bbox_xyxy = outputs[:, :4]
                            identities = outputs[:, -2]
                            object_id = outputs[:, -1]

                            if show_boxes:
                                self.draw_boxes(output_image, bbox_xyxy, object_id, identities, class_names)
                            if show_trajectories:
                                self.draw_trajectory(output_image, bbox_xyxy, object_id, identities, trail_length)
                            if save_text:
                                self.save_csv(bbox_xyxy, object_id, identities, csv_path, frame_number, class_names)

                    # Print inference + NMS time
                    print(f'Frame: {frame_number}. Inference + NMS done ({(t2 - t1):.3f} s).')
                    
                    # View live results
                    if view_image:
                        cv2.imshow('output', output_image)
                        
                    # Save results (image with detections)
                    if save_video:
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(output_image)
                
                frame_number += 1

                if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
                    break
            
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()
                
            print(f'Video processing done. Elapsed time: {time.time() - t0:.3f} s')


    def xyxy_to_xywh(self, *xyxy):
        """" Calculates the relative bounding box from absolute pixel values. """
        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h


    def compute_color(self, label: int) -> tuple:
        """
        Adds color depending on the class
        """
        palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        if label == 0: #person  #BGR
            color = (85, 45, 255)
        elif label == 1: #bicycle
            color = (7, 127, 15)
        elif label == 2: # Car
            color = (255, 149, 0)
        elif label == 3:  # Motobike
            color = (0, 204, 255)
        elif label == 5:  # Bus
            color = (0, 149, 255)
        elif label == 7:  # truck
            color = (222, 82, 175)
        else:
            color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return color


    def draw_boxes(self, inference_image: np.array, bounding_boxes: np.array, object_id: np.array,
                    identities: np.array, class_names: list) -> None:
        """
        Draw bounding boxes on frame
        """
        for index, box in enumerate(bounding_boxes):
            x1, y1, x2, y2 = [int(j) for j in box]

            id = int(identities[index]) if identities is not None else 0
            
            color = self.compute_color(object_id[index])
            label = f'{class_names[object_id[index]]} {id}'

            # Draw box
            cv2.rectangle(inference_image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            
            # Draw labels
            t_size = cv2.getTextSize(label, 0, 2/3, 1)[0]
            cv2.rectangle(inference_image, (x1, y1-t_size[1]-3), (x1 + t_size[0], y1+3), color, -1, cv2.LINE_AA)
            cv2.putText(inference_image, label, (x1, y1 - 2), 0, 2/3, [225, 255, 255], 1, cv2.LINE_AA)
            

    def draw_trajectory(self, inference_image: np.array, bounding_boxes: np.array, object_id: np.array,
                        identities: np.array, trail_length: int) -> None:
        """
        Draw trajectory on frame
        """
        # Remove tracked point from buffer if object is lost
        for key in list(self.data_deque):
            if key not in identities:
                self.data_deque.pop(key)

        for index, box in enumerate(bounding_boxes):
            x1, y1, x2, y2 = [int(j) for j in box]
            center = (int((x2+x1)/2), int((y2+y1)/2))

            id = int(identities[index]) if identities is not None else 0
            color = self.compute_color(object_id[index])

            if id not in self.data_deque:
                self.data_deque[id] = deque(maxlen=trail_length)
            self.data_deque[id].appendleft(center)

            # Draw trajectory
            for k in range(1, len(self.data_deque[id])):
                if self.data_deque[id][k-1] is None or self.data_deque[id][k] is None:
                    continue

                cv2.line(inference_image, self.data_deque[id][k-1], self.data_deque[id][k], color, 2)


    def save_csv(self, bounding_boxes: np.array, object_id: np.array,
                identities: np.array, csv_path: str, frame_number: int, class_names: list) -> None:
        """
        Saves results in CSV file
        """
        for index, box in enumerate(bounding_boxes):
            x1, y1, x2, y2 = [int(j) for j in box]

            id = int(identities[index]) if identities is not None else 0
        
            # Save results in CSV
            with open(csv_path, 'a') as f:
                f.write(f'{frame_number},{id},{str(class_names[object_id[index]])},{x1},{y1},{x2-x1},{y2-y1}\n')


    def load_classes(self, path: str) -> list:
        """
        Extract class names from file *.names
        """
        with open(path, 'r') as f:
            class_names = f.read().split('\n')
        return list(filter(None, class_names))  # filter removes empty strings (such as last line)
    