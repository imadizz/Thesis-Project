# Shared settings for the whole pipeline. Keeping them in one place so the
# experiments stay reproducible and I'm not hunting for magic numbers.

RANDOM_SEED = 42

YOLO_MODEL = 'yolov8m.pt'   # COCO weights, used zero-shot
YOLO_CONF  = 0.25

TEST_SPLIT = 0.25
CV_FOLDS   = 5

# Kalman noise. Q is fixed, R depends on weather/time (camera trusted less
# when visibility is bad).
KF_PROCESS_NOISE     = 0.01
KF_MEASUREMENT_NOISE = 0.05

KF_R_TABLE = {
    ('clear',    'daytime'): 0.05,
    ('overcast', 'daytime'): 0.15,
    ('rainy',    'daytime'): 0.15,
    ('foggy',    'daytime'): 0.20,
    ('clear',    'night'):   0.20,
}

# PCU weights from the Highway Capacity Manual (2010)
PCU_WEIGHTS = {
    'car': 1.0, 'truck': 2.5, 'bus': 3.0,
    'motorcycle': 0.5, 'bicycle': 0.5, 'person': 0.3,
}

CONGESTION_THRESHOLDS = {
    'FREE_FLOW': (0.00, 0.30),
    'MODERATE':  (0.30, 0.60),
    'HEAVY':     (0.60, 0.85),
    'GRIDLOCK':  (0.85, 1.01),
}

RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'
MODELS_DIR  = 'models'

# V2V simulation (NJ-NYC corridor)
SIM_BASELINE_MEAN   = 47.3
SIM_BASELINE_STD    = 8.2
SIM_CONGESTION_RATE = 0.573
SIM_REROUTE_MIN     = 0.26
SIM_REROUTE_MAX     = 0.38
SIM_N_TRIPS         = 500
