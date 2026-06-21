"""
yolov8_detect.py
Run YOLOv8m zero-shot object detection on BDD100K or KITTI frames.
Outputs per-frame PCU scores for downstream classification.

Usage:
    python yolov8_detect.py --dataset bdd100k --img_dir data/bdd100k/images/100k/train
    python yolov8_detect.py --dataset kitti   --img_dir data/kitti/training/image_2
"""

import argparse
import os
import json
from pathlib import Path
from ultralytics import YOLO

# COCO class IDs that map to PCU categories
COCO_TO_PCU = {
    0:  'person',
    2:  'car',
    3:  'motorcycle',
    5:  'bus',
    7:  'truck',
}

PCU_WEIGHTS = {
    'car':        1.0,
    'truck':      2.5,
    'bus':        3.0,
    'motorcycle': 0.5,
    'person':     0.3,
}

CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45


def detect_single_frame(model, img_path: str) -> dict:
    """
    Run YOLOv8m inference on one image.
    Returns {'counts': {...}, 'pcu_score': float, 'confidence_mean': float}
    """
    results = model(img_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
    counts  = {cat: 0 for cat in PCU_WEIGHTS}
    confidences = []

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in COCO_TO_PCU:
            cat = COCO_TO_PCU[cls]
            counts[cat] += 1
            confidences.append(float(box.conf[0]))

    pcu_score = sum(counts[c] * PCU_WEIGHTS[c] for c in counts)
    conf_mean = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        'counts':          counts,
        'pcu_score':       pcu_score,
        'confidence_mean': conf_mean,
        'n_detections':    sum(counts.values()),
    }


def detect_batch(img_dir: str, output_path: str, max_frames: int = None):
    """
    Process all images in img_dir and save results to output_path (JSON).
    """
    model      = YOLO('yolov8m.pt')
    img_dir    = Path(img_dir)
    extensions = {'.jpg', '.jpeg', '.png'}
    img_files  = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in extensions])

    if max_frames:
        img_files = img_files[:max_frames]

    print(f"Processing {len(img_files)} frames from {img_dir}")

    results = {}
    for i, img_path in enumerate(img_files):
        det = detect_single_frame(model, str(img_path))
        results[img_path.name] = det
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(img_files)} frames processed")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved detection results to {output_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    choices=['bdd100k', 'kitti'], required=True)
    parser.add_argument('--img_dir',    required=True)
    parser.add_argument('--output',     default=None)
    parser.add_argument('--max_frames', type=int, default=None)
    args = parser.parse_args()

    output = args.output or f'results/{args.dataset}_detections.json'
    detect_batch(args.img_dir, output, args.max_frames)
