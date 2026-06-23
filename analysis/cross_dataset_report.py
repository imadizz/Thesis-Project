# Train GB on BDD100K, test on KITTI, and write out the per-class scores +
# confusion matrix + the overall domain gap.
#   python -m analysis.cross_dataset_report --bdd ... --kitti ...

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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import config
from fusion.feature_engineering import build_feature_matrix
from classifiers.train_classifiers import build_pipelines

class_order = ['FREE_FLOW', 'MODERATE', 'HEAVY', 'GRIDLOCK']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bdd',   default='results/bdd100k_records.json')
    ap.add_argument('--kitti', default='results/kitti_records.json')
    args = ap.parse_args()

    bdd   = json.load(open(args.bdd))
    kitti = json.load(open(args.kitti))

    X_bdd, y_bdd = build_feature_matrix(bdd)
    X_kit, y_kit = build_feature_matrix(kitti)
    X_bdd, X_kit = X_bdd.astype(np.float64), X_kit.astype(np.float64)

    # train on the BDD train split, keep its test acc for the gap
    Xtr, Xte, ytr, yte = train_test_split(
        X_bdd, y_bdd, test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_SEED, stratify=y_bdd)

    gb = build_pipelines()['GradientBoosting']
    gb.fit(Xtr, ytr)

    acc_bdd = accuracy_score(yte, gb.predict(Xte))
    pred_kit = gb.predict(X_kit)
    acc_kit = accuracy_score(y_kit, pred_kit)

    present = [c for c in class_order if c in set(y_kit) | set(pred_kit)]
    p, r, f1, support = precision_recall_fscore_support(
        y_kit, pred_kit, labels=present, zero_division=0)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR, 'cross_dataset_report.csv')
    with open(csv_path, 'w') as f:
        f.write('class,precision,recall,f1,support\n')
        for i, c in enumerate(present):
            f.write(f'{c},{p[i]:.4f},{r[i]:.4f},{f1[i]:.4f},{int(support[i])}\n')
        f.write(f'OVERALL_BDD100K_test,,,{acc_bdd:.4f},{len(yte)}\n')
        f.write(f'OVERALL_KITTI_zeroshot,,,{acc_kit:.4f},{len(y_kit)}\n')
        f.write(f'DOMAIN_GAP_pp,,,{(acc_bdd-acc_kit)*100:.2f},\n')

    cm = confusion_matrix(y_kit, pred_kit, labels=present)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(present))); ax.set_yticks(range(len(present)))
    ax.set_xticklabels(present, rotation=45, ha='right')
    ax.set_yticklabels(present)
    for i in range(len(present)):
        for j in range(len(present)):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'BDD100K -> KITTI zero-shot ({acc_kit*100:.1f}%)')
    fig.colorbar(im)
    plt.tight_layout()
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    png = os.path.join(config.FIGURES_DIR, 'cross_dataset_confusion.png')
    plt.savefig(png, dpi=150)

    print(f'BDD100K test : {acc_bdd*100:.2f}%')
    print(f'KITTI 0-shot : {acc_kit*100:.2f}%')
    print(f'domain gap   : {(acc_bdd-acc_kit)*100:.2f} pp')
    print('saved', csv_path, 'and', png)


if __name__ == '__main__':
    main()
