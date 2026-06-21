"""
cross_dataset.py
Zero-shot cross-dataset evaluation: apply BDD100K-trained classifier to KITTI.
Tests geographic generalisation from US to German urban driving.

Usage:
    python cross_dataset.py --model_path models/gradientboosting_pipeline.pkl
                            --kitti_records results/kitti_records.json
"""

import argparse
import json
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',     default='models/gradientboosting_pipeline.pkl')
    parser.add_argument('--kitti_records',  default='results/kitti_records.json')
    args = parser.parse_args()

    import sys
    sys.path.append('../fusion')
    from feature_engineering import build_feature_matrix

    print(f"Loading KITTI records from {args.kitti_records}...")
    with open(args.kitti_records) as f:
        records = json.load(f)

    X_kitti, y_kitti = build_feature_matrix(records)
    X_kitti = X_kitti.astype(np.float64)
    print(f"KITTI features: {X_kitti.shape}")

    print(f"Loading trained model from {args.model_path}...")
    pipeline = joblib.load(args.model_path)

    y_pred = pipeline.predict(X_kitti)
    acc    = accuracy_score(y_kitti, y_pred)

    print(f"\nKITTI Cross-Dataset Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_kitti, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_kitti, y_pred))

    # Feature importance if GradientBoosting
    clf = pipeline.named_steps.get('clf')
    if hasattr(clf, 'feature_importances_'):
        fi = clf.feature_importances_
        FEATURE_NAMES = [
            'raw_pcu', 'norm_pcu', 'diversity', 'conf_mean',
            'n_car', 'n_truck', 'n_bus', 'n_motorcycle', 'n_person',
            'est_speed', 'flow_density', 'rush_hour', 'speed_dev', 'stability',
            'kalman_state', 'kalman_P', 'innovation', 'kalman_gain',
            'weather_code', 'time_code', 'scene_code',
            'wx_pcu', 'tx_pcu', 'sx_flow', 'pcu_lane', 'conf_pcu', 'rolling_mean'
        ]
        ranked = sorted(zip(FEATURE_NAMES, fi), key=lambda x: -x[1])
        print("\nTop-10 Feature Importances:")
        for name, imp in ranked[:10]:
            print(f"  {name:20s}: {imp:.4f}")


if __name__ == '__main__':
    main()
