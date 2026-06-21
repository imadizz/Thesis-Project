"""
train_classifiers.py
Train and evaluate Random Forest, Gradient Boosting, and MLP classifiers
on the 27-feature congestion dataset.

Usage:
    python train_classifiers.py --records_path results/bdd100k_records.json
                                --cv_folds 5 --test_size 0.25

Note on scikit-learn 1.7.2 bug:
    MLPClassifier raises TypeError when early_stopping=True with float64 input.
    Workaround: set early_stopping=False and cast X to np.float64 explicitly.
"""

import argparse
import json
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)

RANDOM_STATE = 42


def build_pipelines() -> dict:
    """Return the three classifier pipelines."""
    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            max_features='sqrt',
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ))
    ])

    gb = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=1.0,
            loss='log_loss',
            random_state=RANDOM_STATE,
        ))
    ])

    # early_stopping=False: workaround for scikit-learn 1.7.2 float64 TypeError
    mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=300,
            early_stopping=False,
            random_state=RANDOM_STATE,
        ))
    ])

    return {'RandomForest': rf, 'GradientBoosting': gb, 'MLP': mlp}


def evaluate(name, pipeline, X_train, X_test, y_train, y_test, cv_folds=5):
    print(f"\n{'='*50}")
    print(f"Classifier: {name}")
    print(f"{'='*50}")

    # Cross-validation on training set
    skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_acc = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
    print(f"CV Accuracy ({cv_folds}-fold): {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")

    # Fit and evaluate on held-out test set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return acc, pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records_path', default='results/bdd100k_records.json')
    parser.add_argument('--cv_folds',     type=int, default=5)
    parser.add_argument('--test_size',    type=float, default=0.25)
    parser.add_argument('--save_models',  action='store_true')
    args = parser.parse_args()

    import sys
    sys.path.append('../fusion')
    sys.path.append('../data_preparation')
    from feature_engineering import build_feature_matrix
    from load_bdd100k import load_annotations, build_pcu_records

    print(f"Loading annotations from {args.records_path}...")

    # Expect records as JSON (from load_bdd100k output)
    with open(args.records_path) as f:
        records = json.load(f)

    X, y = build_feature_matrix(records)
    X    = X.astype(np.float64)  # ensure float64 for sklearn 1.7.2

    print(f"Feature matrix: {X.shape}, Classes: {set(y)}")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    pipelines = build_pipelines()
    results   = {}

    for name, pipeline in pipelines.items():
        acc, fitted = evaluate(name, pipeline, X_train, X_test, y_train, y_test, args.cv_folds)
        results[name] = {'accuracy': acc, 'pipeline': fitted}

        if args.save_models:
            os.makedirs('models', exist_ok=True)
            joblib.dump(fitted, f'models/{name.lower()}_pipeline.pkl')
            print(f"Saved models/{name.lower()}_pipeline.pkl")

    print("\n\nSUMMARY")
    print("-" * 40)
    for name, res in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
        print(f"  {name:20s}: {res['accuracy']:.4f}")


if __name__ == '__main__':
    main()
