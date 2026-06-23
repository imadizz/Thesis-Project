"""
analysis/feature_importance.py
Compute and save Gradient Boosting feature importances for the 27-feature
congestion model. Optionally adds a SHAP summary plot if shap is installed.

Outputs (written to results/ and figures/):
    results/feature_importance.csv
    figures/feature_importance.png
    figures/shap_summary.png   (only if shap is available)

Usage:
    python -m analysis.feature_importance --records results/bdd100k_records.json
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

import config
from fusion.feature_engineering import build_feature_matrix
from classifiers.train_classifiers import build_pipelines

# Names of the 27 features, in the order build_features() produces them.
FEATURE_NAMES = [
    # Camera (9)
    'raw_pcu', 'norm_pcu', 'diversity', 'conf_mean',
    'car', 'truck', 'bus', 'motorcycle', 'person',
    # GPS (5)
    'gps_speed', 'flow_density', 'rush_hour', 'speed_dev', 'stability',
    # Fusion (4)
    'kalman_state', 'kalman_P', 'kalman_innovation', 'kalman_gain',
    # Context (9)
    'weather_code', 'time_code', 'scene_code',
    'weather_x_pcu', 'time_x_pcu', 'speed_x_flow',
    'pcu_per_lane', 'conf_x_pcu', 'pcu_rolling_mean',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', default='results/bdd100k_records.json')
    parser.add_argument('--top_n', type=int, default=15)
    args = parser.parse_args()

    with open(args.records) as f:
        records = json.load(f)

    X, y = build_feature_matrix(records)
    X = X.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_SEED, stratify=y)

    gb = build_pipelines()['GradientBoosting']
    gb.fit(X_train, y_train)
    importances = gb.named_steps['clf'].feature_importances_

    order = np.argsort(importances)[::-1]
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    # CSV
    csv_path = os.path.join(config.RESULTS_DIR, 'feature_importance.csv')
    with open(csv_path, 'w') as f:
        f.write('rank,feature,importance\n')
        for rank, idx in enumerate(order, 1):
            f.write(f'{rank},{FEATURE_NAMES[idx]},{importances[idx]:.6f}\n')
    print(f'Wrote {csv_path}')

    # Bar chart of top N
    top = order[:args.top_n][::-1]
    plt.figure(figsize=(8, 6))
    plt.barh([FEATURE_NAMES[i] for i in top],
             [importances[i] for i in top], color='#2E75B6')
    plt.xlabel('Gini importance')
    plt.title(f'Top {args.top_n} Feature Importances (Gradient Boosting)')
    plt.tight_layout()
    png_path = os.path.join(config.FIGURES_DIR, 'feature_importance.png')
    plt.savefig(png_path, dpi=150)
    print(f'Wrote {png_path}')

    print('\nTop features:')
    for rank, idx in enumerate(order[:args.top_n], 1):
        print(f'  {rank:2d}. {FEATURE_NAMES[idx]:20s} {importances[idx]*100:5.1f}%')

    # Optional SHAP summary
    try:
        import shap
        explainer = shap.TreeExplainer(gb.named_steps['clf'])
        Xs = gb.named_steps['scaler'].transform(X_test[:1000])
        shap_values = explainer.shap_values(Xs)
        shap.summary_plot(shap_values, Xs, feature_names=FEATURE_NAMES,
                          show=False)
        shap_path = os.path.join(config.FIGURES_DIR, 'shap_summary.png')
        plt.tight_layout()
        plt.savefig(shap_path, dpi=150, bbox_inches='tight')
        print(f'Wrote {shap_path}')
    except ImportError:
        print('shap not installed; skipping SHAP summary (pip install shap)')


if __name__ == '__main__':
    main()
