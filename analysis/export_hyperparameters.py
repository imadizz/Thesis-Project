# Dump every hyperparameter to one json so the settings behind a result are
# saved next to it.  python -m analysis.export_hyperparameters

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def main():
    hp = {
        'random_seed': config.RANDOM_SEED,
        'yolo_model': config.YOLO_MODEL,
        'yolo_conf': config.YOLO_CONF,
        'test_split': config.TEST_SPLIT,
        'cv_folds': config.CV_FOLDS,
        'kalman': {
            'process_noise_Q': config.KF_PROCESS_NOISE,
            'measurement_noise_R_baseline': config.KF_MEASUREMENT_NOISE,
            'R_table': {f'{w}_{t}': v for (w, t), v in config.KF_R_TABLE.items()},
        },
        'gradient_boosting': {'n_estimators': 200, 'max_depth': 6,
                              'learning_rate': 0.1, 'loss': 'log_loss'},
        'random_forest': {'n_estimators': 200, 'max_depth': 20, 'max_features': 'sqrt'},
        'mlp': {'hidden_layer_sizes': [128, 64, 32], 'activation': 'relu',
                'solver': 'adam', 'alpha': 0.0001, 'max_iter': 300, 'early_stopping': False},
    }

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = os.path.join(config.RESULTS_DIR, 'hyperparameters.json')
    json.dump(hp, open(out, 'w'), indent=2)
    print('saved', out)


if __name__ == '__main__':
    main()
