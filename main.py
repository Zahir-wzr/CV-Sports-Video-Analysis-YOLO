import argparse
from collections import deque
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Please install ultralytics first: pip install ultralytics")


def iou_xyxy(a, b):
    # a, b: (x1,y1,x2,y2)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def pick_main_person(person_boxes, prev_box=None):
    """
    person_boxes: list of (x1,y1,x2,y2,conf)
    Strategy:
      - if prev_box exists: choose box with highest IoU to prev_box
      - else: choose largest area
    """
    if not person_boxes:
        return None

    if prev_box is not None:
        best = max(person_boxes, key=lambda b: iou_xyxy(prev_box, b[:4]))
        return best[:4], best[4]
    else:
        best = max(person_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return best[:4], best[4]


def draw_trajectory(frame, points, thickness=3):
    # points: deque of (x,y)
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        p1 = tuple(map(int, points[i - 1]))
        p2 = tuple(map(int, points[i]))
        cv2.line(frame, p1, p2, (0, 255, 0), thickness)  # green line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video, e.g. data/swimming.mp4")
    parser.add_argument("--output", default="results/output_with_traj.mp4", help="Path to output video")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model path (n/s/m/l/x) or local .pt")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--device", default=None, help="cuda, cpu, or None(auto)")
    parser.add_argument("--trail", type=int, default=80, help="How many points to keep for trajectory")
    parser.add_argument("--person_only", action="store_true", help="Only draw person boxes (default: yes)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    model = YOLO(args.model)

    # trajectory storage
    pts = deque(maxlen=args.trail)
    prev_box = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # YOLO inference
        results = model.predict(frame, conf=args.conf, device=args.device, verbose=False)
        r = results[0]

        person_boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            # COCO class 0 is 'person'
            for (x1, y1, x2, y2), cf, c in zip(boxes, confs, clss):
                if c == 0:  # person
                    person_boxes.append((float(x1), float(y1), float(x2), float(y2), float(cf)))

        picked = pick_main_person(person_boxes, prev_box=prev_box)
        if picked is not None:
            (x1, y1, x2, y2), cf = picked
            prev_box = (x1, y1, x2, y2)

            # center point for trajectory
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            pts.append((cx, cy))

            # draw box + label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"person {cf:.2f}",
                (int(x1), max(0, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        else:
            pts.append(None)

        # draw trajectory line
        draw_trajectory(frame, pts, thickness=3)

        # small HUD
        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        out.write(frame)

    cap.release()
    out.release()
    print(f"Done. Saved to: {args.output}")


if __name__ == "__main__":
    main()
