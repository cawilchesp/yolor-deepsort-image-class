import cv2
import numpy as np

from collections import deque

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

data_deque = {}

def xyxy_to_xywh(*xyxy):
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


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color(label: int) -> tuple:
    """
    Adds color depending on the class
    """
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


def draw_boxes(img: np.array, bbox: np.array, object_id: np.array,
                identities: np.array, csv_path: str, frame_num: int, names: list, opt_trailslen) -> None:
    """
    Draw bounding boxes on frame and saves results in CSV
    """
        
    # Remove tracked point from buffer if object is lost
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(j) for j in box]

        center = (int((x2+x1)/2), int((y2+y1)/2))

        id = int(identities[i]) if identities is not None else 0
        if id not in data_deque:
            data_deque[id] = deque(maxlen=opt_trailslen)

        data_deque[id].appendleft(center)

        color = compute_color(object_id[i])
        label = f'{names[object_id[i]]} {id}'

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        
        # Draw trajectory
        for k in range(1, len(data_deque[id])):
            if data_deque[id][k-1] is None or data_deque[id][k] is None:
                continue

            cv2.line(img, data_deque[id][k-1], data_deque[id][k], color, 2)

        # Draw labels
        t_size = cv2.getTextSize(label, 0, 2/3, 1)[0]
        cv2.rectangle(img, (x1, y1-t_size[1]-3), (x1 + t_size[0], y1+3), color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x1, y1 - 2), 0, 2/3, [225, 255, 255], 1, cv2.LINE_AA)

        # Save results in CSV
        with open(csv_path, 'a') as f:
            f.write(f'{frame_num},{id},{str(names[object_id[i]])},{x1},{y1},{x2-x1},{y2-y1},0,\n')


def load_classes(path: str) -> list:
    """
    Extract class names from file *.names
    """
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)