# Federated-Learning-Optimizer-Comparison
Decentralized Federated Learning on RSSCN7 using EfficientNet-B0. Compares GWO, PSO, and ABC optimizers for model selection in a fog-cloud simulation with 10 drones, FogBroker filtering, and FedAvg aggregation. Research project — NIT Delhi Internship 2025.
## Overview

This project implements a **decentralized federated learning** system where 10 drones collaboratively train an image classification model **without sharing raw data**. Three bio-inspired optimizers are compared for selecting the best local models before aggregation:

| Optimizer | Full Name | Inspiration |
|-----------|-----------|-------------|
| **GWO** | Grey Wolf Optimizer | Wolf hunting hierarchy |
| **PSO** | Particle Swarm Optimization | Bird flocking behaviour |
| **ABC** | Artificial Bee Colony | Bee foraging strategy |

---

## System Architecture

```
Drones 1–5  ──►  Router A (FogBroker)  ─┐
                                          ├──► Cloud (FedAvg) ──► Global Model
Drones 6–10 ──►  Router B (FogBroker)  ─┘
```

Each round:
1. All 10 drones train locally on their data partition
2. A `Router` filters images by pixel intensity before training
3. A `VM` runs pre-validation inference on filtered images
4. Each `FogBroker` filters models into: **VALID / REPROCESS / DROP** queues
5. The selected optimizer picks the **top 3 models** from each router's valid queue
6. `FedAvg` aggregates selected models at the cloud
7. The global model is redistributed to all drones

---

## Dataset

**RSSCN7** — Remote Sensing Scene Classification Dataset  
- 7 scene categories, 400 images each (2800 total)
- Train/Test split: 2300 train, 500 test
- Download via Kaggle API (see setup below)

---

## Model

**EfficientNet-B0** (pretrained on ImageNet)
- All feature layers fine-tuned
- Final classifier replaced: `Linear(1280 → 7)`
- Input size: 224×224

---

## Project Structure

```
federated-learning-optimizer-comparison/
│
├── federated_learning_optimizer_comparison.py   # Main unified script
│
├── notebooks/
│   ├── Code_7_GWO.ipynb                         # Original GWO notebook
│   ├── Code_8_PSO.ipynb                         # Original PSO notebook
│   └── Code_9_ABC.ipynb                         # Original ABC notebook
│
├── logs/
│   ├── gwo_federated_logs.xls                   # GWO run results
│   ├── fed_pso_effnetb0_logs.xls                # PSO run results
│   └── fed_logs_abc_optimizer.xls               # ABC run results
│
├── results/                                     # Saved plots (auto-generated)
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/saraswatsakshi12/federated-learning-optimizer-comparison.git
cd federated-learning-optimizer-comparison
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Kaggle API key
- Go to [kaggle.com](https://www.kaggle.com) → Account → Create API Token
- Place the downloaded `kaggle.json` in the project root directory

---

## Usage

Run with any of the three optimizers:

```bash
# Grey Wolf Optimizer
python federated_learning_optimizer_comparison.py --optimizer gwo

# Particle Swarm Optimization
python federated_learning_optimizer_comparison.py --optimizer pso

# Artificial Bee Colony
python federated_learning_optimizer_comparison.py --optimizer abc
```

Logs are saved to `logs/` and plots to `results/` automatically.

---

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Drones | 10 |
| Rounds | 10 |
| Local Epochs | 5 |
| Batch Size | 32 |
| Bandwidth | 50 Mbps |
| Optimizer: models selected per router | Top 3 |
| Fitness function | `α(1 - acc) + β(latency/max_latency)` |
| α (accuracy weight) | 1.0 |
| β (latency weight) | 0.05 |

---

## Results

Logs from actual runs are available in the `logs/` folder (`.xls` files).  
Each log tracks per round:
- Global accuracy (%)
- Total latency (ms)
- Cloud upload latency (ms)
- Average system cost
- Model acceptance rate (Router A & B)
- Average fitness score (Router A & B)
- Per-drone training loss

---

## Optimizer Details

### GWO — Grey Wolf Optimizer
Models are ranked by fitness score. The top 3 act as **Alpha** (best), **Beta** (2nd), and **Delta** (3rd) wolves — guiding the selection, inspired by the social hierarchy of grey wolf packs.

### PSO — Particle Swarm Optimization
Candidate subsets of models act as particles. Over multiple iterations, each particle's position is updated toward its personal best and the global best, converging on the lowest-fitness model subset.

### ABC — Artificial Bee Colony
**Employed bees** exploit known model candidates; **onlooker bees** switch to better candidates based on fitness probability; **scout bees** explore random alternatives. The best solution is tracked across iterations.

---

## Tech Stack

- Python 3.9+
- PyTorch + TorchVision
- EfficientNet-B0 (pretrained)
- NumPy, Matplotlib
- Kaggle API

---
