"""
load_kitti.py
Load KITTI object detection annotations, map to PCU categories,
and compute per-frame congestion scores.
Dataset: https://www.cvlibs.net/datasets/kitti/eval_object.php
"""

import os
from pathlib import Path
from pcu_scoring import PCU_WEIGHTS, compute_pcu_score, RollingNormaliser, assign_congestion_class

# Map KITTI category names to PCU categories
KITTI_TO_PCU = {
    'Car':             'car',
    'Van':             'car',
    'Truck':           'truck',
    'Tram':            'truck',
    'Cyclist':         'motorcycle',
    'Pedestrian':      'person',
    'Person_sitting':  'person',
    'Misc':            None,
    'DontCare':        None,
}


def parse_kitti_label_file(label_path: str) -> dict:
    """
    Parse a single KITTI .txt label file.
    Returns count dict per PCU category.
    """
    counts = {cat: 0 for cat in PCU_WEIGHTS}
    if not os.path.exists(label_path):
        return counts

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            kitti_cat = parts[0]
            pcu_cat   = KITTI_TO_PCU.get(kitti_cat)
            if pcu_cat:
                counts[pcu_cat] += 1

    return counts


def build_kitti_records(label_dir: str) -> list[dict]:
    """
    Process all KITTI label files and return PCU records.
    KITTI has no weather/timeofday metadata so we use neutral defaults.
    """
    normaliser = RollingNormaliser(window=100)
    records    = []
    label_dir  = Path(label_dir)

    for label_file in sorted(label_dir.glob('*.txt')):
        frame_id = label_file.stem
        counts   = parse_kitti_label_file(str(label_file))
        raw_pcu  = compute_pcu_score(counts)
        norm_pcu = normaliser.normalise('urban', raw_pcu)
        label    = assign_congestion_class(norm_pcu)

        records.append({
            'frame':     frame_id,
            'weather':   'clear',       # KITTI: no weather metadata
            'scene':     'urban',
            'timeofday': 'daytime',     # KITTI: recorded daytime only
            'counts':    counts,
            'raw_pcu':   raw_pcu,
            'norm_pcu':  norm_pcu,
            'label':     label,
        })

    return records


if __name__ == '__main__':
    import sys
    label_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/kitti/training/label_2'
    records   = build_kitti_records(label_dir)
    print(f"Loaded {len(records)} KITTI frames")

    from collections import Counter
    dist = Counter(r['label'] for r in records)
    for cls, cnt in sorted(dist.items()):
        print(f"  {cls}: {cnt} ({100*cnt/len(records):.1f}%)")
