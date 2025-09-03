import cv2
import numpy as np
import logging

# Setup logging
import os
os.makedirs("output/logs", exist_ok=True)
logging.basicConfig(filename="output/logs/utils_log.txt", level=logging.DEBUG, filemode='w', format='[%(levelname)s] %(message)s')
logger = logging.getLogger("UtilsLogger")


def draw_bbox(frame, bbox, class_id, label="", color=(0, 255, 0), draw_label=True):
    x, y, w, h = bbox
    logger.debug(f"\n[DRAW_BBOX] Drawing box: {bbox} | Class ID: {class_id} | Label: {label} | Color: {color}")
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    if draw_label and label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        label_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_rect_top_left = (x, y - label_size[1] - 10)
        label_rect_bottom_right = (x + label_size[0], y)
        label_bg_color = tuple([max(c - 60, 0) for c in color])

        logger.debug(f" → Drawing label background at: {label_rect_top_left} to {label_rect_bottom_right} | Label size: {label_size}")
        cv2.rectangle(frame, label_rect_top_left, label_rect_bottom_right, label_bg_color, -1)
        cv2.putText(frame, label, (x, y - 5), font, font_scale, (255, 255, 255), thickness)


def draw_detections(frame, tracks, class_names):
    logger.debug("\n[DRAW_DETECTIONS] Drawing active tracks...")
    for track in tracks:
        if track.box is None or track.skipped_frames > 0:
            continue

        box = track.box
        class_id = int(track.class_id) if track.class_id is not None else -1
        label = class_names[class_id] if 0 <= class_id < len(class_names) else f"ID {track.track_id}"
        label += f" {int(track.velocity)} mph"

        logger.debug(f" → Track ID: {track.track_id} | Class ID: {class_id} | Label: {label} | Box: {box}")
        color = get_color(class_id)
        draw_bbox(frame, box, class_id, label=label, color=color)


def draw_filled_rect(frame, rect, color=(0, 255, 0), alpha=0.4):
    overlay = frame.copy()
    x, y, w, h = rect
    logger.debug(f"\n[DRAW_FILLED_RECT] Drawing filled rectangle: {rect} | Color: {color} | Alpha: {alpha}")
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_filled_circle(frame, center, radius=10, color=(0, 255, 0), alpha=0.4):
    overlay = frame.copy()
    logger.debug(f"\n[DRAW_FILLED_CIRCLE] Drawing circle at: {center} | Radius: {radius} | Color: {color} | Alpha: {alpha}")
    cv2.circle(overlay, center, radius, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_trace(frame, trace, color=(255, 255, 255), thickness=2, step=1):
    logger.debug(f"\n[DRAW_TRACE] Drawing trajectory trace. Total points: {len(trace)} | Color: {color} | Thickness: {thickness}")
    for i in range(0, len(trace) - 1, step):
        logger.debug(f" → Line: {trace[i]} to {trace[i + 1]}")
        cv2.line(frame, trace[i], trace[i + 1], color, thickness)


def compute_overlap_ratio(box1, box2):
    logger.debug(f"\n[COMPUTE_IOU] Calculating IoU between Box1: {box1} and Box2: {box2}")
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area + 1e-6

    iou = inter_area / union_area if union_area > 0 else 0.0
    logger.debug(f" → Intersection area: {inter_area} | Union area: {union_area:.2f} | IoU: {iou:.4f}")
    return iou


def get_color(class_id):
    np.random.seed(class_id + 100)
    color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
    logger.debug(f"\n[GET_COLOR] Generated color for Class ID {class_id}: {color}")
    return color


def letterbox_resize(image, target_size=(640, 640), color=(114, 114, 114)):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    logger.debug(f"\n[LETTERBOX_RESIZE] Original size: {w}x{h}, Target size: {target_size}")
    logger.debug(f" → Scale: ({scale:.4f}, {scale:.4f}) | New size: {new_w}x{new_h}")

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    top, bottom = pad_y, target_h - new_h - pad_y
    left, right = pad_x, target_w - new_w - pad_x

    logger.debug(f" → Padding (top: {top}, bottom: {bottom}, left: {left}, right: {right})")

    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=color)

    logger.debug(f"[LETTERBOX_RESIZE] Resized image shape: {padded_image.shape}")
    return padded_image, (scale, scale), (pad_x, pad_y)
