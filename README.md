# BreakHis Federated Learning Benchmark: Normalization & Fairness

A benchmarking framework designed to evaluate the trade-offs between accuracy, training performance (resource efficiency), and class-level fairness under Federated Learning (FL) using standard Federated Averaging (FedAvg).

The project uses a non-IID partitioning of the **BreakHis** (breast cancer histopathology) dataset across 5 clients.

## 🚀 Key Features

* **Federated Learning Framework**: Implemented from scratch using PyTorch with a 4-block Convolutional Neural Network (`BreakHisCNN`).
* **Normalization Analysis**: Benchmark the performance and convergence of 5 different normalization schemes:
  * **Baseline (None)**: No normalization layers.
  * **BatchNorm**: `BatchNorm2d` (standard for centralized setups, but has challenges under non-IID client distributions).
  * **GroupNorm**: `GroupNorm` (independent of batch size, often preferred in FL).
  * **LayerNorm**: `LayerNorm` (spatial-channel normalization).
  * **InstanceNorm**: `InstanceNorm2d` (style-normalization flavor).
* **Robust Optimizations**: Benchmark configurations using Focal Loss (handling class imbalance) and L2 Regularization.
* **Fairness & Resource Evaluation**: Measures the training time per client per round alongside standard ML metrics (Accuracy, Precision, Recall, Macro-F1). Class fairness is quantified based on performance disparity across classes.
* **Visualization Tools**: Automatically generate comparison charts for convergence, training times, loss curves, and accuracy-fairness tradeoffs.

---

## 📁 Repository Structure

* `main.py`: Entry point to execute the benchmark across all configurations.
* `analysis.py`: Tooling to process results and generate plots.
* `training_loop.py`: The federated training, client local updating, and server aggregation routines.
* `config.py`: Centralized training hyperparameters and dataset paths.
* `data_loader.py`: Handles non-IID Dirichlet distribution dataset partition among clients.
* `models/`: Contains the CNN architecture (`models/cnn.py`).
* `solutions/`: Contains configuration files defining standard parameters for normalization, loss functions, and optimization settings.

---

## 🛠️ Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Ensure that you place the BreakHis dataset in the following folder structure:
```text
data/
└── breakhis/
    ├── benign/
    └── malignant/
```

### 3. Run Benchmark
Execute the federated benchmark on all configurations:
```bash
python main.py
```
This will train each configuration for 20 communication rounds (configurable in `config.py`) and save the logs in the `results/` directory.

### 4. Analyze and Visualize
Generate the analysis plots and metrics table:
```bash
python analysis.py
```
This generates the following visualizations inside `results/`:
* `accuracy_fairness_tradeoff.png`
* `convergence_curves.png`
* `f1_precision_recall.png`
* `training_time_comparison.png`
* `loss_curves.png`

---

## 📊 Evaluation Metrics
* **Accuracy (%)**: Overall test set accuracy.
* **Fairness Score**: Class fairness calculated as:
  $$\text{Fairness} = 1 - \frac{\max(\text{Class Accuracies}) - \min(\text{Class Accuracies})}{100}$$
* **Macro-averaged Precision, Recall, and F1-score**.
* **Training Time**: Total computing time in seconds per round.
