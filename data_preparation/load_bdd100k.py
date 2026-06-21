"""
load_bdd100k.py
Load BDD100K detection annotations and compute per-frame PCU scores.
Dataset: https://bdd-data.berkeley.edu/ (detection task, 70K frames)
"""

import json
import os
from pathlib import Path
from pcu_scoring import PCU_WEIGHTS, compute_pcu_score, RollingNormaliser, assign_congestion_class

RELEVANT_CATEGORIES = set(PCU_WEIGHTS.keys())


def load_annotations(ann_path: str) -> dict:
    """
    Parse BDD100K detection JSON.
    Returns {frame_name: {'labels': [...], 'weather': str, 'scene': str, 'timeofday': str}}
    """
    with open(ann_path, 'r') as f:
        data = json.load(f)

    frames = {}
    for item in data:
        fname = item['name']
        attrs = item.get('attributes', {})
        labels = [
            l for l in item.get('labels', [])
            if l.get('category') in RELEVANT_CATEGORIES
        ]
        frames[fname] = {
            'labels':    labels,
            'weather':   attrs.get('weather', 'clear'),
            'scene':     attrs.get('scene', 'city street'),
            'timeofday': attrs.get('timeofday', 'daytime'),
        }
    return frames


def build_pcu_records(frames: dict) -> list[dict]:
    """
    Convert frame annotation dicts into per-frame PCU records.
    Returns list of dicts with keys: frame, weather, scene, timeofday,
    counts, raw_pcu, norm_pcu, label.
    """
    normaliser = RollingNormaliser(window=100)
    records = []

    for fname, meta in frames.items():
        counts = {cat: 0 for cat in PCU_WEIGHTS}
        for lbl in meta['labels']:
            cat = lbl.get('category')
            if cat in counts:
                counts[cat] += 1

        raw_pcu  = compute_pcu_score(counts)
        norm_pcu = normaliser.normalise(meta['scene'], raw_pcu)
        label    = assign_congestion_class(norm_pcu)

        records.append({
            'frame':     fname,
            'weather':   meta['weather'],
            'scene':     meta['scene'],
            'timeofday': meta['timeofday'],
            'counts':    counts,
            'raw_pcu':   raw_pcu,
            'norm_pcu':  norm_pcu,
            'label':     label,
        })

    return records


if __name__ == '__main__':
    import sys
    ann_path = sys.argv[1] if len(sys.argv) > 1 else 'data/bdd100k/labels/det_20/det_train.json'
    frames  = load_annotations(ann_path)
    records = build_pcu_records(frames)
    print(f"Loaded {len(records)} frames")

    from collections import Counter
    dist = Counter(r['label'] for r in records)
    for cls, cnt in sorted(dist.items()):
        print(f"  {cls}: {cnt} ({100*cnt/len(records):.1f}%)")
