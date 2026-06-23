"""
analysis/statistical_tests.py
Formal significance testing of camera-only vs full-fusion congestion
classification. Reports a McNemar test and a paired bootstrap confidence
interval on the accuracy difference.

This supports the dissertation statement that adding GPS/fusion features does
NOT produce a statistically significant improvement over camera-only features.

Outputs:
    results/statistical_tests.json

Usage:
    python -m analysis.statistical_tests --records results/bdd100k_records.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split

import config
from fusion.feature_engineering import build_feature_matrix
from classifiers.train_classifiers import build_pipelines

# Indices 0-8 are the 9 camera-group features (see feature_engineering order).
CAMERA_IDX = list(range(9))


def mcnemar(correct_a, correct_b):
    """
    McNemar test on two boolean correctness vectors.
    b = A wrong, B right ; c = A right, B wrong.
    Uses the exact binomial form for robustness on small discordant counts.
    """
    b = int(np.sum(~correct_a & correct_b))
    c = int(np.sum(correct_a & ~correct_b))
    n = b + c
    if n == 0:
        return {'b': b, 'c': c, 'statistic': 0.0, 'p_value': 1.0}
    # Exact two-sided binomial p-value with p=0.5
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p = min(1.0, 2.0 * tail)
    stat = (abs(b - c) - 1) ** 2 / n  # continuity-corrected chi-square
    return {'b': b, 'c': c, 'statistic': float(stat), 'p_value': float(p)}


def paired_bootstrap(correct_a, correct_b, n_boot=10000, seed=42):
    """Bootstrap CI on accuracy difference (B - A)."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', default='results/bdd100k_records.json')
    args = parser.parse_args()

    with open(args.records) as f:
        records = json.load(f)

    X, y = build_feature_matrix(records)
    X = X.astype(np.float64)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_SEED, stratify=y)

    # Full fusion model
    gb_full = build_pipelines()['GradientBoosting']
    gb_full.fit(X_train, y_train)
    pred_full = gb_full.predict(X_test)

    # Camera-only model (same hyperparameters, 9 features)
    gb_cam = build_pipelines()['GradientBoosting']
    gb_cam.fit(X_train[:, CAMERA_IDX], y_train)
    pred_cam = gb_cam.predict(X_test[:, CAMERA_IDX])

    correct_cam  = (pred_cam  == y_test)
    correct_full = (pred_full == y_test)

    acc_cam  = float(correct_cam.mean())
    acc_full = float(correct_full.mean())

    mc = mcnemar(correct_cam, correct_full)
    bs = paired_bootstrap(correct_cam, correct_full)

    result = {
        'n_test': int(len(y_test)),
        'accuracy_camera_only': acc_cam,
        'accuracy_full_fusion': acc_full,
        'accuracy_difference_pp': (acc_full - acc_cam) * 100,
        'mcnemar': mc,
        'bootstrap_95ci': bs,
        'significant_at_0.05': bool(mc['p_value'] < 0.05),
    }

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = os.path.join(config.RESULTS_DIR, 'statistical_tests.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Camera-only accuracy : {acc_cam*100:.2f}%")
    print(f"Full fusion accuracy : {acc_full*100:.2f}%")
    print(f"Difference           : {(acc_full-acc_cam)*100:+.2f} pp")
    print(f"McNemar p-value      : {mc['p_value']:.4f}")
    print(f"Bootstrap 95% CI     : [{bs['ci_low']*100:+.2f}, {bs['ci_high']*100:+.2f}] pp")
    verdict = ("significant" if result['significant_at_0.05']
               else "NOT statistically significant")
    print(f"Conclusion           : difference is {verdict} at alpha=0.05")
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
