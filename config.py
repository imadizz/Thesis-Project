"""
config.py
Central configuration for the congestion detection pipeline.
All scripts read their constants from here so experiments stay reproducible
and there is a single place to change a hyperparameter.
"""

# -- Reproducibility --
RANDOM_SEED = 42

# -- Detection --
YOLO_MODEL  = 'yolov8m.pt'   # COCO pre-trained weights, applied zero-shot
YOLO_CONF   = 0.25           # detection confidence threshold

# -- Train / test split --
TEST_SPLIT  = 0.25           # held-out fraction (stratified)
CV_FOLDS    = 5

# -- Kalman filter --
KF_PROCESS_NOISE      = 0.01    # Q: process noise variance
KF_MEASUREMENT_NOISE  = 0.05    # R baseline (clear/daytime); weather-adjusted in filter

# Weather/time adjusted measurement noise R (higher R = trust camera less)
KF_R_TABLE = {
    ('clear',    'daytime'): 0.05,
    ('overcast', 'daytime'): 0.15,
    ('rainy',    'daytime'): 0.15,
    ('foggy',    'daytime'): 0.20,
    ('clear',    'night'):   0.20,
}

# -- PCU weighting (Highway Capacity Manual, 2010) --
PCU_WEIGHTS = {
    'car':        1.0,
    'truck':      2.5,
    'bus':        3.0,
    'motorcycle': 0.5,
    'bicycle':    0.5,
    'person':     0.3,
}

# -- Congestion thresholds on normalised 0-1 PCU score --
CONGESTION_THRESHOLDS = {
    'FREE_FLOW': (0.00, 0.30),
    'MODERATE':  (0.30, 0.60),
    'HEAVY':     (0.60, 0.85),
    'GRIDLOCK':  (0.85, 1.01),
}

# -- Output locations --
RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'
MODELS_DIR  = 'models'

# -- V2V analytical simulation --
SIM_BASELINE_MEAN   = 47.3    # minutes, NJ-NYC corridor calibrated
SIM_BASELINE_STD    = 8.2
SIM_CONGESTION_RATE = 0.573
SIM_REROUTE_MIN     = 0.26
SIM_REROUTE_MAX     = 0.38
SIM_N_TRIPS         = 500
