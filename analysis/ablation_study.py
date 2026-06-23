# Ablation over the four feature groups. Trains GB on each subset so we can see
# how much GPS / Kalman / context actually add on top of the camera features.
#   python -m analysis.ablation_study --records results/bdd100k_records.json

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

# column ranges for each feature group (see build order in feature_engineering)
camera  = list(range(0, 9))
gps     = list(range(9, 14))
fusion  = list(range(14, 18))
context = list(range(18, 27))

configs = [
    ('Camera only',     camera),
    ('Camera + Kalman', camera + fusion),
    ('Camera + GPS',    camera + gps),
    ('Full Fusion',     camera + gps + fusion + context),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', default='results/bdd100k_records.json')
    args = ap.parse_args()

    records = json.load(open(args.records))
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
        print('Ablation (Gradient Boosting):')
        for label, cols in configs:
            gb = build_pipelines()['GradientBoosting']
            gb.fit(Xtr[:, cols], ytr)
            pred = gb.predict(Xte[:, cols])
            acc = accuracy_score(yte, pred)
            p, r, f1, _ = precision_recall_fscore_support(
                yte, pred, average='macro', zero_division=0)
            f.write(f'{label},{len(cols)},{acc:.4f},{p:.4f},{r:.4f},{f1:.4f}\n')
            print(f'  {label:18s} ({len(cols):2d} feat)  acc {acc*100:.2f}%  f1 {f1*100:.2f}')

    print('saved', out)


if __name__ == '__main__':
    main()
