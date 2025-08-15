import cv2
import numpy as np
from datetime import datetime


def draw_box(img, box, color=(0,255,0), label=None, thickness=2):
    x1,y1,x2,y2 = [int(x) for x in box]
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, max(10, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def arrow_between(img, box_a, box_b, color=(255,255,0)):
    xa = int((box_a[0]+box_a[2]) / 2.0); ya = int((box_a[1]+box_a[3]) / 2.0)
    xb = int((box_b[0]+box_b[2]) / 2.0); yb = int((box_b[1]+box_b[3]) / 2.0)
    cv2.arrowedLine(img, (xa, ya), (xb, yb), color, 2, tipLength=0.03)


def render_image(image_cv, boxes, best, out_path='output.jpg', names=None, parsed_query=None):
    out = image_cv.copy()
    if best is None:
        cv2.putText(out, 'No matching relation found', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.imwrite(out_path, out)
        return out_path

    s_idx = best['sub_idx']; o_idx = best['obj_idx']
    subj_box = boxes[s_idx]
    obj_box = boxes[o_idx]

    # labels
    subj_label = None
    obj_label = None
    if names is not None and isinstance(names, dict):
        subj_label = names.get(int(subj_box[4]) if subj_box.shape[0] > 4 else 0, None)

    # Draw boxes
    draw_box(out, subj_box, (0,255,0), label=(parsed_query.get('subject', {}).get('class') if parsed_query else 'sub'))
    draw_box(out, obj_box, (0,0,255), label=(parsed_query.get('object', {}).get('class') if parsed_query else 'obj'))

    # arrow
    arrow_between(out, subj_box, obj_box)

    # relation label near arrow midpoint
    midx = int((subj_box[0] + obj_box[2]) / 4.0 + (subj_box[2] + obj_box[0]) / 4.0)
    midy = int((subj_box[1] + obj_box[3]) / 4.0 + (obj_box[1] + subj_box[3]) / 4.0)
    cv2.putText(out, best.get('rel_label', ''), (midx, midy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    # timestamp and score
    cv2.putText(out, f"score={best.get('score',0):.2f}", (20, out.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    cv2.imwrite(out_path, out)
    return out_path