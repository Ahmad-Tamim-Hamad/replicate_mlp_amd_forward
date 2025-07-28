# MLP Forward Model Replication on ADM Dataset

This repository contains a clean replication of the **MLP forward model** described in the paper:  
[Benchmarking Data-driven Surrogate Simulators for Artificial Electromagnetic Materials](https://openreview.net/pdf?id=-or413Lh_aF)

The goal is to predict electromagnetic spectral responses from geometric structures for **Artificial Electromagnetic Materials (AEMs)** using the **ADM dataset**.

---

## Code Attribution

This work is based on the official benchmark repository:  
➡ [https://github.com/yangdeng-EML/ML_MM_Benchmark](https://github.com/yangdeng-EML/ML_MM_Benchmark)

All training, evaluation, and plotting pipelines were adapted and cleaned for clarity and reuse.

---

## What's Included

- Trained MLP model on ADM (`models/MLP/models/MLP/20250728_010501`)
- Modular training and evaluation scripts (`train.py`, `evaluate.py`)
- Visualizations: spectrum comparison, MAE, MSE plots
- Post-evaluation analysis script (`plotsAnalysis.py`)
- Auto-generated evaluation results in `evaluation_results/`

---

## Folder Structure

```bash
replicate_mlp_amd_forward/
├── data/
│   ├── __init__.py
│   └── loader.py
├── models/
│   └── MLP/
│       ├── class_wrapper.py
│       ├── evaluate.py
│       ├── flag_reader.py
│       ├── generate_all_eval_plots.py
│       ├── model_maker.py
│       ├── parameters.py
│       ├── train.py
│       ├── evaluation_results/
│       │   ├── _MLP_20250728_010501.png
│       │   ├── mae_distribution_MLP_20250728_010501.png
│       │   ├── mse_distribution_MLP_20250728_010501.png
│       │   ├── mse_summary_errorbar_MLP_20250728_010501.png
│       │   ├── spectrum_comparison_0.png
│       │   ├── spectrum_comparison_10.png
│       │   ├── spectrum_comparison_100.png
│       │   └── spectrum_comparison_500.png
│       └── utils/
│           ├── data_reader.py
│           ├── evaluation_helper.py
│           ├── get_mse_list.py
│           ├── get_outcome_stats.py
│           ├── helper_functions.py
│           ├── plot_swipe.py
│           ├── plotsAnalysis.py
│           ├── time_recorder.py
│           └── total_training_time.py
├── metrics.py
├── LICENSE
├── pyproject.toml
├── setup.py
└── README.md
