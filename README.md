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
├── data/                       
│   └── ADM/
│       ├── data_x.csv        
│       ├── data_y.csv       
│       └── testset/
├── models/
│   └── MLP/
│       ├── train.py           
│       ├── evaluate.py        
│       ├── generate_all_eval_plots.py 
│       ├── model_maker.py    
│       ├── class_wrapper.py   
│       ├── parameters.py      
│       └── models/
│           └── MLP/20250728_010501  # Trained checkpoint
├── evaluation_results/                          
├── README.md
└── .gitignore
