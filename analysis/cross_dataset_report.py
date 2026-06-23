"""
analysis/cross_dataset_report.py
Detailed BDD100K -> KITTI zero-shot cross-dataset evaluation.
Trains Gradient Boosting on BDD100K, tests on KITTI, and reports per-class
precision/recall/F1, the confusion matrix, and the overall domain gap.

Outputs:
    results/cross_dataset_report.csv
    figures/cross_dataset_confusion.png

Usage:
    python -m analysis.cross_dataset_report \
        --bdd results/bdd100k_records.json \
        --kitti results/kitti_records.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

import config
from fusion.feature_engineering import build_feature_matrix
from classifiers.train_classifiers import build_pipelines

CLASS_ORDER = ['FREE_FLOW', 'MODERATE', 'HEAVY', 'GRIDLOCK']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bdd',   default='results/bdd100k_records.json')
    parser.add_argument('--kitti', default='results/kitti_records.json')
    args = parser.parse_args()

    with open(args.bdd) as f:
        bdd = json.load(f)
    with open(args.kitti) as f:
        kitti = json.load(f)

    X_bdd, y_bdd = build_feature_matrix(bdd)
    X_kit, y_kit = build_feature_matrix(kitti)
    X_bdd, X_kit = X_bdd.astype(np.float64), X_kit.astype(np.float64)

    # Train on BDD100K train split, record its own test accuracy for the gap
    Xtr, Xte, ytr, yte = train_test_split(
        X_bdd, y_bdd, test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_SEED, stratify=y_bdd)

    gb = build_pipelines()['GradientBoosting']
    gb.fit(Xtr, ytr)

    acc_bdd   = accuracy_score(yte, gb.predict(Xte))
    y_kit_pred = gb.predict(X_kit)
    acc_kitti = accuracy_score(y_kit, y_kit_pred)

    present = [c for c in CLASS_ORDER if c in set(y_kit) | set(y_kit_pred)]
    p, r, f1, support = precision_recall_fscore_support(
        y_kit, y_kit_pred, labels=present, zero_division=0)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR, 'cross_dataset_report.csv')
    with open(csv_path, 'w') as f:
        f.write('class,precision,recall,f1,support\n')
        for i, c in enumerate(present):
            f.write(f'{c},{p[i]:.4f},{r[i]:.4f},{f1[i]:.4f},{int(support[i])}\n')
        f.write(f'OVERALL_BDD100K_test,,,{acc_bdd:.4f},{len(yte)}\n')
        f.write(f'OVERALL_KITTI_zeroshot,,,{acc_kitti:.4f},{len(y_kit)}\n')
        f.write(f'DOMAIN_GAP_pp,,,{(acc_bdd-acc_kitti)*100:.2f},\n')
    print(f'Wrote {csv_path}')

    # Confusion matrix figure
    cm = confusion_matrix(y_kit, y_kit_pred, labels=present)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(present)))
    ax.set_yticks(range(len(present)))
    ax.set_xticklabels(present, rotation=45, ha='right')
    ax.set_yticklabels(present)
    for i in range(len(present)):
        for j in range(len(present)):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'BDD100K -> KITTI Zero-Shot  (acc {acc_kitti*100:.1f}%)')
    fig.colorbar(im)
    plt.tight_layout()
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    png = os.path.join(config.FIGURES_DIR, 'cross_dataset_confusion.png')
    plt.savefig(png, dpi=150)
    print(f'Wrote {png}')

    print(f'\nBDD100K test accuracy : {acc_bdd*100:.2f}%')
    print(f'KITTI zero-shot       : {acc_kitti*100:.2f}%')
    print(f'Domain gap            : {(acc_bdd-acc_kitti)*100:.2f} pp')


if __name__ == '__main__':
    main()
