# Is the camera-vs-fusion accuracy difference actually significant, or just
# noise? McNemar test + a bootstrap CI on the difference. Used to back up the
# claim in the dissertation that fusion didn't help in a meaningful way.

import argparse
import json
import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split

import config
from fusion.feature_engineering import build_feature_matrix
from classifiers.train_classifiers import build_pipelines

camera_cols = list(range(9))   # first 9 features are the camera group


def mcnemar(correct_a, correct_b):
    # b = a wrong / b right, c = a right / b wrong
    b = int(np.sum(~correct_a & correct_b))
    c = int(np.sum(correct_a & ~correct_b))
    n = b + c
    if n == 0:
        return {'b': b, 'c': c, 'statistic': 0.0, 'p_value': 1.0}
    # exact two-sided binomial, p=0.5
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    p = min(1.0, 2.0 * tail)
    stat = (abs(b - c) - 1) ** 2 / n   # chi-square with continuity correction
    return {'b': b, 'c': c, 'statistic': float(stat), 'p_value': float(p)}


def bootstrap_diff(correct_a, correct_b, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = correct_b[idx].mean() - correct_a[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {'mean_diff': float(diffs.mean()),
            'ci_low': float(lo), 'ci_high': float(hi)}


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

    # full model
    full = build_pipelines()['GradientBoosting']
    full.fit(Xtr, ytr)
    pred_full = full.predict(Xte)

    # camera-only, same hyperparameters
    cam = build_pipelines()['GradientBoosting']
    cam.fit(Xtr[:, camera_cols], ytr)
    pred_cam = cam.predict(Xte[:, camera_cols])

    ok_cam  = (pred_cam == yte)
    ok_full = (pred_full == yte)
    acc_cam, acc_full = float(ok_cam.mean()), float(ok_full.mean())

    mc = mcnemar(ok_cam, ok_full)
    bs = bootstrap_diff(ok_cam, ok_full)
    sig = mc['p_value'] < 0.05

    result = {
        'n_test': int(len(yte)),
        'accuracy_camera_only': acc_cam,
        'accuracy_full_fusion': acc_full,
        'accuracy_difference_pp': (acc_full - acc_cam) * 100,
        'mcnemar': mc,
        'bootstrap_95ci': bs,
        'significant_at_0.05': bool(sig),
    }

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = os.path.join(config.RESULTS_DIR, 'statistical_tests.json')
    json.dump(result, open(out, 'w'), indent=2)

    print(f'camera-only : {acc_cam*100:.2f}%')
    print(f'full fusion : {acc_full*100:.2f}%')
    print(f'difference  : {(acc_full-acc_cam)*100:+.2f} pp')
    print(f'McNemar p   : {mc["p_value"]:.4f}')
    print(f'bootstrap CI : [{bs["ci_low"]*100:+.2f}, {bs["ci_high"]*100:+.2f}] pp')
    print('=>', 'significant' if sig else 'not significant', 'at alpha=0.05')
    print('saved', out)


if __name__ == '__main__':
    main()
