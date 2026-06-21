# Traffic Congestion Detection via Multi-Source Fusion

**MSc Artificial Intelligence | BSBI Berlin / University for the Creative Arts**  
**Student:** Aditya Lokhande | Q1093411  
**Supervisor:** Dr. Vincent English  
**Submitted:** July 2026

---

## Project Title

*Improving Traffic Congestion Detection through Multi-Source Fusion of Camera and GPS Data: A Route Replanning Study Using Driving Datasets*

---

## Overview

This repository contains all code for my MSc dissertation. The project investigates whether fusing dashcam-based object detection data with GPS movement features improves traffic congestion classification, and whether that improvement translates into journey time savings through Vehicle-to-Vehicle (V2V) cooperative routing.

**Datasets used:**
- [BDD100K](https://bdd-data.berkeley.edu/) — 70,000 US dashcam frames (Berkeley)
- [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php) — 7,481 German urban frames (Karlsruhe)

---

## Research Questions

- **RQ1:** How effectively can traffic congestion be detected using camera-based driving datasets?
- **RQ2:** Does combining camera-derived traffic indicators with GPS data improve congestion detection accuracy?
- **RQ3:** To what extent can improved congestion detection support more efficient route replanning in a simulated V2V context?

---

## Key Results

| Metric | Value |
|--------|-------|
| Best classifier (BDD100K) | **94.7%** — Gradient Boosting |
| Camera-only baseline | 90.5% |
| Fusion improvement | +4.2 percentage points |
| MODERATE class improvement | +5.8 percentage points |
| KITTI cross-dataset accuracy | 81.3% (zero-shot) |
| V2V journey time saving (0% packet loss) | **18.4%** |
| V2V journey time saving (60% packet loss) | 5.2% |

---

## Repository Structure

```
Thesis-Project/
│
├── data_preparation/
│   ├── pcu_scoring.py          # PCU weighting and congestion class assignment
│   ├── load_bdd100k.py         # BDD100K annotation loader
│   └── load_kitti.py           # KITTI annotation loader
│
├── detection/
│   └── yolov8_detect.py        # YOLOv8m zero-shot inference on frames
│
├── fusion/
│   ├── kalman_filter.py        # Adaptive scalar Kalman filter
│   └── feature_engineering.py  # 27-feature vector construction
│
├── classifiers/
│   ├── train_classifiers.py    # RF, GB, MLP training and evaluation
│   └── cross_dataset.py        # BDD100K -> KITTI zero-shot transfer
│
├── simulation/
│   ├── v2v_routing.py          # SUMO TraCI V2V routing protocol
│   └── run_simulation.py       # Batch simulation runner
│
├── figures/                    # Result figures from the dissertation
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/imadizz/Thesis-Project.git
cd Thesis-Project
pip install -r requirements.txt
```

For the V2V simulation, install [SUMO](https://sumo.dlr.de/docs/Installing/index.html) and set `SUMO_HOME`.

---

## Running the Pipeline

### 1. Detect vehicles with YOLOv8m

```bash
python detection/yolov8_detect.py \
    --dataset bdd100k \
    --img_dir data/bdd100k/images/100k/train \
    --output results/bdd100k_detections.json
```

### 2. Build features and train classifiers

```bash
python classifiers/train_classifiers.py \
    --records_path results/bdd100k_records.json \
    --cv_folds 5 \
    --test_size 0.25 \
    --save_models
```

### 3. Cross-dataset evaluation (BDD100K model on KITTI)

```bash
python classifiers/cross_dataset.py \
    --model_path models/gradientboosting_pipeline.pkl \
    --kitti_records results/kitti_records.json
```

### 4. V2V routing simulation

```bash
python simulation/run_simulation.py \
    --config simulation/njnyc.sumocfg \
    --packet_loss 0 0.2 0.4 0.6 \
    --n_runs 3
```

---

## Kalman Filter

Fuses PCU-weighted camera score with GPS features. Measurement noise R adapts to weather:

```python
R_TABLE = {
    ('clear',    'daytime'): 0.05,   # trust camera
    ('overcast', 'daytime'): 0.15,
    ('rainy',    'daytime'): 0.15,
    ('foggy',    'daytime'): 0.20,   # trust GPS more
    ('clear',    'night'):   0.20,
}
```

## PCU Weighting (Highway Capacity Manual, 2010)

| Vehicle | PCU | Congestion Class | Score |
|---------|-----|-----------------|-------|
| Car | 1.0 | FREE_FLOW | < 0.30 |
| Truck | 2.5 | MODERATE | 0.30–0.60 |
| Bus | 3.0 | HEAVY | 0.60–0.85 |
| Motorcycle | 0.5 | GRIDLOCK | ≥ 0.85 |
| Pedestrian | 0.3 | | |

---

## Known Issue: scikit-learn 1.7.2 Bug

`MLPClassifier` raises `TypeError` when `early_stopping=True` with `float64` input arrays.

**Workaround** (applied in `train_classifiers.py`):
```python
X = X.astype(np.float64)
MLPClassifier(early_stopping=False, ...)
```

---

## Citation

```
Lokhande, A. (2026). Improving Traffic Congestion Detection through Multi-Source
Fusion of Camera and GPS Data. MSc Dissertation, BSBI Berlin / UCA.
Supervisor: Dr. Vincent English.
GitHub: https://github.com/imadizz/Thesis-Project
```
