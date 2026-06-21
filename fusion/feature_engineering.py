"""
feature_engineering.py
Build the 27-feature vector per frame used by all three classifiers.

Feature groups:
  Camera  (9):  raw PCU, normalised score, diversity, confidence, per-category counts
  GPS     (5):  estimated speed, flow density, rush-hour flag, speed deviation, stability
  Fusion  (4):  Kalman state, uncertainty P, innovation, gain K
  Context (9):  weather code, timeofday code, scene code, 3 interaction terms,
                PCU per lane, confidence-weighted PCU, 3-frame rolling mean
"""

import numpy as np
from collections import deque
from kalman_filter import AdaptiveKalman

# Encoding maps for categorical metadata
WEATHER_CODE  = {'clear': 0, 'overcast': 1, 'rainy': 2, 'foggy': 3, 'snowy': 4}
TIMEOFDAY_CODE = {'daytime': 0, 'dawn/dusk': 1, 'night': 2}
SCENE_CODE     = {'highway': 0, 'city street': 1, 'residential': 2, 'tunnel': 3}

# Estimated baseline speeds km/h by scene and time (synthesised from US traffic surveys)
SPEED_BASELINE = {
    ('highway',     'daytime'): 105.0,
    ('highway',     'night'):    95.0,
    ('city street', 'daytime'):  50.0,
    ('city street', 'night'):    55.0,
    ('residential', 'daytime'):  40.0,
    ('residential', 'night'):    40.0,
    ('tunnel',      'daytime'):  60.0,
    ('tunnel',      'night'):    60.0,
}
RUSH_HOUR_SCENES = {'city street', 'residential'}


def is_rush_hour(timeofday: str, scene: str) -> int:
    """Return 1 if frame is likely during rush hour, else 0."""
    return 1 if (timeofday == 'daytime' and scene in RUSH_HOUR_SCENES) else 0


def estimate_gps_speed(scene: str, timeofday: str, norm_pcu: float) -> float:
    """Estimate vehicle speed in km/h from scene type and congestion level."""
    base = SPEED_BASELINE.get((scene, timeofday),
           SPEED_BASELINE.get((scene, 'daytime'), 50.0))
    # Speed drops linearly with congestion
    return base * max(0.1, 1.0 - 0.8 * norm_pcu)


def build_features(record: dict, kalman: AdaptiveKalman,
                   pcu_history: deque) -> np.ndarray:
    """
    Build a 27-element feature vector for one frame record.

    Args:
        record:      Dict from load_bdd100k / load_kitti with detection results
        kalman:      AdaptiveKalman instance (maintains state across frames)
        pcu_history: deque of last 3 normalised PCU scores (maxlen=3)

    Returns:
        np.ndarray of shape (27,), dtype float64
    """
    counts    = record['counts']
    norm_pcu  = record['norm_pcu']
    raw_pcu   = record['raw_pcu']
    weather   = record.get('weather',   'clear')
    scene     = record.get('scene',     'city street')
    timeofday = record.get('timeofday', 'daytime')
    conf_mean = record.get('confidence_mean', 0.5)

    # -- Camera group (9 features) --
    diversity     = sum(1 for v in counts.values() if v > 0)
    f_camera = [
        raw_pcu,
        norm_pcu,
        float(diversity),
        conf_mean,
        float(counts.get('car',        0)),
        float(counts.get('truck',      0)),
        float(counts.get('bus',        0)),
        float(counts.get('motorcycle', 0)),
        float(counts.get('person',     0)),
    ]

    # -- GPS group (5 features) --
    est_speed    = estimate_gps_speed(scene, timeofday, norm_pcu)
    flow_density = raw_pcu / max(est_speed, 1.0)
    rush         = is_rush_hour(timeofday, scene)
    base_speed   = SPEED_BASELINE.get((scene, 'daytime'), 50.0)
    speed_dev    = base_speed - est_speed
    stability    = 1.0 / (1.0 + abs(flow_density - 0.5))

    f_gps = [est_speed, flow_density, float(rush), speed_dev, stability]

    # -- Fusion group (4 features) --
    kout = kalman.update(norm_pcu, weather=weather, timeofday=timeofday)
    f_fusion = [kout['state'], kout['P_prior'], kout['innovation'], kout['gain']]

    # -- Context group (9 features) --
    w_code  = float(WEATHER_CODE.get(weather, 0))
    t_code  = float(TIMEOFDAY_CODE.get(timeofday, 0))
    s_code  = float(SCENE_CODE.get(scene, 0))
    wx_pcu  = w_code * norm_pcu
    tx_pcu  = t_code * norm_pcu
    sx_flow = est_speed * flow_density
    n_lanes = 2.0 if scene == 'highway' else 1.0
    pcu_lane = raw_pcu / n_lanes
    conf_pcu = conf_mean * norm_pcu

    pcu_history.append(norm_pcu)
    rolling_mean = float(np.mean(pcu_history))

    f_context = [w_code, t_code, s_code, wx_pcu, tx_pcu, sx_flow,
                 pcu_lane, conf_pcu, rolling_mean]

    features = f_camera + f_gps + f_fusion + f_context   # 9+5+4+9 = 27
    return np.array(features, dtype=np.float64)


def build_feature_matrix(records: list[dict]) -> tuple:
    """
    Build X (features) and y (labels) arrays from a list of frame records.

    Returns:
        X: np.ndarray shape (N, 27)
        y: list of N congestion class strings
    """
    kalman      = AdaptiveKalman()
    pcu_history = deque([0.0] * 3, maxlen=3)
    X, y        = [], []

    for rec in records:
        feat = build_features(rec, kalman, pcu_history)
        X.append(feat)
        y.append(rec['label'])

    return np.array(X, dtype=np.float64), y
