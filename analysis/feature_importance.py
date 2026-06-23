# Gradient Boosting feature importances -> csv + bar chart. Also drops a SHAP
# summary if shap happens to be installed.
#   python -m analysis.feature_importance --records results/bdd100k_records.json

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

import config
from fusion.feature_engineering import build_feature_matrix
from classifiers.train_classifiers import build_pipelines

# 27 feature names, same order as build_features()
FEATURE_NAMES = [
    'raw_pcu', 'norm_pcu', 'diversity', 'conf_mean',
    'car', 'truck', 'bus', 'motorcycle', 'person',
    'gps_speed', 'flow_density', 'rush_hour', 'speed_dev', 'stability',
    'kalman_state', 'kalman_P', 'kalman_innovation', 'kalman_gain',
    'weather_code', 'time_code', 'scene_code',
    'weather_x_pcu', 'time_x_pcu', 'speed_x_flow',
    'pcu_per_lane', 'conf_x_pcu', 'pcu_rolling_mean',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', default='results/bdd100k_records.json')
    ap.add_argument('--top_n', type=int, default=15)
    args = ap.parse_args()

    records = json.load(open(args.records))
    X, y = build_feature_matrix(records)
    X = X.astype(np.float64)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_SEED, stratify=y)

    gb = build_pipelines()['GradientBoosting']
    gb.fit(Xtr, ytr)
    imp = gb.named_steps['clf'].feature_importances_
    order = np.argsort(imp)[::-1]

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    csv_path = os.path.join(config.RESULTS_DIR, 'feature_importance.csv')
    with open(csv_path, 'w') as f:
        f.write('rank,feature,importance\n')
        for rank, idx in enumerate(order, 1):
            f.write(f'{rank},{FEATURE_NAMES[idx]},{imp[idx]:.6f}\n')

    top = order[:args.top_n][::-1]
    plt.figure(figsize=(8, 6))
    plt.barh([FEATURE_NAMES[i] for i in top], [imp[i] for i in top], color='#2E75B6')
    plt.xlabel('Gini importance')
    plt.title(f'Top {args.top_n} features (Gradient Boosting)')
    plt.tight_layout()
    png_path = os.path.join(config.FIGURES_DIR, 'feature_importance.png')
    plt.savefig(png_path, dpi=150)

    print('top features:')
    for rank, idx in enumerate(order[:args.top_n], 1):
        print(f'  {rank:2d}. {FEATURE_NAMES[idx]:20s} {imp[idx]*100:.1f}%')
    print('saved', csv_path, 'and', png_path)

    # optional SHAP
    try:
        import shap
        expl = shap.TreeExplainer(gb.named_steps['clf'])
        Xs = gb.named_steps['scaler'].transform(Xte[:1000])
        sv = expl.shap_values(Xs)
        shap.summary_plot(sv, Xs, feature_names=FEATURE_NAMES, show=False)
        shap_path = os.path.join(config.FIGURES_DIR, 'shap_summary.png')
        plt.tight_layout()
        plt.savefig(shap_path, dpi=150, bbox_inches='tight')
        print('saved', shap_path)
    except ImportError:
        print('(shap not installed, skipping SHAP plot)')


if __name__ == '__main__':
    main()
