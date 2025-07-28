# MLP Forward Model Replication on ADM Dataset

This repository contains a clean replication of the **MLP forward model** described in the paper:  
[Benchmarking Data-driven Surrogate Simulators for Artificial Electromagnetic Materials](https://openreview.net/pdf?id=-or413Lh_aF)

The model maps geometric structure inputs to electromagnetic spectral responses for **Artificial Electromagnetic Materials (AEMs)** using the **ADM dataset**.

---

## What's Included

- Trained MLP model on ADM (`models/MLP/20250728_010501`)
- Cleaned training and evaluation pipeline
- `generate_all_eval_plots.py`: Spectrum comparison, MAE and MSE distributions
- Visuals and plots under `images/` and `evaluation_results/`

---

## Folder Structure

```bash
ML_MM_Benchmark/
├── data/                       # Contains ADM dataset
│   └── ADM/
│       ├── data_x.csv         # Geometric parameters
│       ├── data_y.csv         # Spectral responses
│       └── testset/
├── models/
│   └── MLP/
│       ├── train.py           # MLP training script
│       ├── evaluate.py        # Evaluation logic
│       ├── generate_all_eval_plots.py  # Custom evaluation plotting
│       ├── model_maker.py     # Model architecture (Forward MLP)
│       ├── class_wrapper.py   # Wrapper for model evaluation
│       ├── parameters.py      # Default hyperparameters
│       └── models/
│           └── MLP/20250728_010501  # Trained checkpoint
├── evaluation_results/        # Evaluation output and plots
├── images/                    # Paper visualizations and architecture
├── README.md
└── .gitignore
