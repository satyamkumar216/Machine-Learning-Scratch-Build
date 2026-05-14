# Machine Learning From Scratch

> A pure-Python implementation of core Machine Learning algorithms. No black boxes, just pure math and logic.

##  Project Motivation
This repository is dedicated to understanding what actually happens under the hood of standard Data Science libraries. By building algorithms entirely from scratch without relying on `scikit-learn` for the core logic, this project explores the raw mathematics, spatial mechanics, and algorithmic time complexities that power modern Machine Learning.

## Repository Structure

```text
ML-From-Scratch/
│
├── 01_Data_Prep/                 # Data cleaning and preparation tools
│   └─ Outlier_Detection/        # Algorithmic anomaly detection
|     ├──DBSCAN
│       ├── dbscan.py             # Custom DBSCAN logic built from scratch
|       |__dbscangraph.py            # using graph algo
|       └── demo_dbscan.ipynb     # Benchmarking: Custom vs. Scikit-Learn
│
├── datasets/                     # Raw CSV/Data files (Local only)
├── .gitignore                    # Python/Jupyter cache ignoring
└── README.md                     # Project documentation