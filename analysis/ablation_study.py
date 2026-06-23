"""
analysis/ablation_study.py
Feature-group ablation for the congestion classifier. Trains Gradient Boosting
on four feature subsets and reports accuracy / precision / recall / F1.

    Camera only
    Camera + Kalman (fusion group)
    Camera + GPS
    Full Fusion (all 27 features)

Output:
    results/ablation_study.csv

Usage:
    python -m analysis.ablation_study --records results/bdd100k_records.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import config
from fusion.feature_engineering import build_feature_matrix
from classifiers.train_classifiers import build_pipelines

# Feature-group column indices (see feature_engineering build order)
CAMERA  = list(range(0, 9))
GPS     = list(range(9, 14))
FUSION  = list(range(14, 18))
CONTEXT = list(range(18, 27))

CONFIGS = [
    ('Camera only',     CAMERA),
    ('Camera + Kalman', CAMERA + FUSION),
    ('Camera + GPS',    CAMERA + GPS),
    ('Full Fusion',     CAMERA + GPS + FUSION + CONTEXT),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', default='results/bdd100k_records.json')
    args = parser.parse_args()

    with open(args.records) as f:
        records = json.load(f)

    X, y = build_feature_matrix(records)
    X = X.astype(np.float64)
    y = np.array(y)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_SEED, stratify=y)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = os.path.join(config.RESULTS_DIR, 'ablation_study.csv')
    with open(out, 'w') as f:
        f.write('configuration,n_features,accuracy,precision_macro,recall_macro,f1_macro\n')
        print('Ablation study (Gradient Boosting):')
        for label, cols in CONFIGS:
            gb = build_pipelines()['GradientBoosting']
            gb.fit(Xtr[:, cols], ytr)
            pred = gb.predict(Xte[:, cols])
            acc = accuracy_score(yte, pred)
            p, r, f1, _ = precision_recall_fscore_support(
                yte, pred, average='macro', zero_division=0)
            f.write(f'{label},{len(cols)},{acc:.4f},{p:.4f},{r:.4f},{f1:.4f}\n')
            print(f'  {label:18s} ({len(cols):2d} feat): '
                  f'acc {acc*100:5.2f}%  F1 {f1*100:5.2f}')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
