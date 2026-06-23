# Quick summary tables: overall dataset stats + per-feature mean/std/min/max.
#   python -m analysis.dataset_statistics --records results/bdd100k_records.json

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from fusion.feature_engineering import build_feature_matrix
from analysis.feature_importance import FEATURE_NAMES

vehicle_keys = ['car', 'truck', 'bus', 'motorcycle', 'person']
class_order  = ['FREE_FLOW', 'MODERATE', 'HEAVY', 'GRIDLOCK']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', default='results/bdd100k_records.json')
    args = ap.parse_args()

    records = json.load(open(args.records))
    X, y = build_feature_matrix(records)
    X = X.astype(np.float64)

    n = len(records)
    dist = Counter(y)
    n_test = int(round(n * config.TEST_SPLIT))
    avg_veh = np.mean([sum(r['counts'].get(k, 0) for k in vehicle_keys) for r in records])

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    ds = os.path.join(config.RESULTS_DIR, 'dataset_statistics.csv')
    with open(ds, 'w') as f:
        f.write('statistic,value\n')
        f.write(f'total_frames,{n}\n')
        f.write(f'train_frames,{n - n_test}\n')
        f.write(f'test_frames,{n_test}\n')
        f.write(f'num_classes,{len(dist)}\n')
        f.write(f'avg_vehicles_per_frame,{avg_veh:.2f}\n')
        for c in class_order:
            cnt = dist.get(c, 0)
            f.write(f'class_{c}_count,{cnt}\n')
            f.write(f'class_{c}_pct,{100*cnt/n:.1f}\n')

    fs = os.path.join(config.RESULTS_DIR, 'feature_summary.csv')
    with open(fs, 'w') as f:
        f.write('feature,mean,std,min,max\n')
        for i, name in enumerate(FEATURE_NAMES):
            col = X[:, i]
            f.write(f'{name},{col.mean():.4f},{col.std():.4f},{col.min():.4f},{col.max():.4f}\n')

    print('saved', ds, 'and', fs)


if __name__ == '__main__':
    main()
