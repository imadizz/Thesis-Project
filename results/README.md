# Results Directory

Every experiment writes its artifacts here so dissertation figures and tables
are directly reproducible. Re-running the pipeline regenerates these files.

| File | Produced by | Contents |
|------|-------------|----------|
| `bdd100k_records.json` | `data_preparation/load_bdd100k.py` | Per-frame PCU records (input to training) |
| `kitti_records.json` | `data_preparation/load_kitti.py` | KITTI per-frame PCU records |
| `feature_importance.csv` | `analysis/feature_importance.py` | Ranked Gradient Boosting feature importances |
| `statistical_tests.json` | `analysis/statistical_tests.py` | McNemar + bootstrap, camera-only vs fusion |
| `cross_dataset_report.csv` | `analysis/cross_dataset_report.py` | Per-class P/R/F1 + domain gap on KITTI |
| `v2v_simulation.json` | `simulation/v2v_analytical.py` | Journey-time savings vs packet loss |

All scripts read shared constants (seed, split, hyperparameters) from
`config.py`, so a single change there propagates to every experiment.
