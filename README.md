# Neural Network from Scratch — NumPy Only



![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)




![NumPy](https://img.shields.io/badge/Built%20with-NumPy%20Only-green)




![Accuracy](https://img.shields.io/badge/Test%20Accuracy-97.4%25-brightgreen)




![License](https://img.shields.io/badge/License-MIT-lightgrey)



> A complete MNIST classifier implemented from first principles using **NumPy only** —
> dense layers, ReLU, Batch Normalization, Inverted Dropout, numerically stable
> Softmax + Cross-Entropy, manual backpropagation, Adam optimizer, checkpointing,
> and full metrics visualizations. **Zero PyTorch. Zero TensorFlow. Zero scikit-learn neural net.**

---

## Why From Scratch?

- **Proves theory depth** — anyone can call `model.fit()`; deriving ∂L/∂W manually separates engineers from API users
- **Exposes what frameworks hide** — numerical instability in Softmax, vanishing gradients through BatchNorm, correct inverted Dropout scaling
- **Amazon ML Summer School curriculum maps directly** — backpropagation, Adam, regularization, and evaluation metrics are all core modules

---

## Architecture

```
Input X       (batch, 784)    ← 28×28 MNIST image, flattened + normalized [0,1]
    │
    ▼
Dense₁        (784 → 256)     W₁: [784×256]   b₁: [256]
ReLU          (256)
BatchNorm₁    (256)           γ₁: [256]        β₁: [256]
Dropout₁      p = 0.3
    │
    ▼
Dense₂        (256 → 128)     W₂: [256×128]   b₂: [128]
ReLU          (128)
BatchNorm₂    (128)           γ₂: [128]        β₂: [128]
Dropout₂      p = 0.2
    │
    ▼
Dense₃        (128 → 10)      W₃: [128×10]    b₃: [10]
Softmax       (10)
    │
    ▼
Output Ŷ      (batch, 10)     ← class probability distribution, sums to 1.0
```

| Component | Count |
|---|---|
| Trainable parameters | **235,146** |
| Weight init | He (Kaiming) for ReLU layers |
| Optimizer | Adam (β₁=0.9, β₂=0.999, ε=1e-8) |
| Loss | Cross-Entropy with log-sum-exp stability |

---

## Math — Forward Pass

For a mini-batch of size $m$:

**Layer 1:**

$$Z^{[1]} = X W^{[1]} + b^{[1]} \quad \text{shape: } (m \times 256)$$

$$A^{[1]} = \text{ReLU}(Z^{[1]}) = \max(0,\; Z^{[1]})$$

**Batch Normalization (layer $l$):**

$$\mu_l = \frac{1}{m} \sum_{i=1}^{m} z_i \qquad \sigma^2_l = \frac{1}{m} \sum_{i=1}^{m}(z_i - \mu_l)^2$$

$$\hat{z}_l = \frac{z_l - \mu_l}{\sqrt{\sigma^2_l + \varepsilon}} \qquad \text{out}_l = \gamma_l \cdot \hat{z}_l + \beta_l$$

**Inverted Dropout (keep probability $1-p$):**

$$\text{mask}_l \sim \frac{\text{Bernoulli}(1-p)}{(1-p)} \qquad A^{[1]}_{\text{drop}} = A^{[1]}_{\text{bn}} \odot \text{mask}_l$$

> Dividing by $(1-p)$ at training time means no adjustment needed at inference — activations stay on the same scale.

**Layer 2:**

$$Z^{[2]} = A^{[1]}_{\text{drop}}\, W^{[2]} + b^{[2]} \quad \text{shape: } (m \times 128)$$

$$A^{[2]} = \text{ReLU}(Z^{[2]}), \quad A^{[2]}_{\text{bn}} = \text{BN}(A^{[2]}), \quad A^{[2]}_{\text{drop}} = \text{Dropout}(A^{[2]}_{\text{bn}},\; p=0.2)$$

**Output logits and Softmax:**

$$Z^{[3]} = A^{[2]}_{\text{drop}}\, W^{[3]} + b^{[3]} \quad \text{shape: } (m \times 10)$$

$$\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}} \quad \Longrightarrow \quad \text{implemented as: } z_i - \log\!\sum_j e^{z_j - z_{\max}}$$

> **Log-sum-exp trick:** subtract $z_{\max}$ before exponentiating to prevent `inf` overflow.

**Cross-Entropy Loss:**

$$\mathcal{L} = -\frac{1}{m} \sum_{n=1}^{m} \sum_{k=1}^{10} y_{nk} \log(\hat{y}_{nk})$$

---

## Math — Backpropagation

### Softmax + Cross-Entropy (fused gradient)

$$\delta^{[3]} = \frac{\hat{Y} - Y}{m} \qquad \text{shape: } (m \times 10)$$

> Fusing these two gradients cancels the Softmax Jacobian — the result is clean and numerically stable.

### Output Dense Layer

$$\frac{\partial \mathcal{L}}{\partial W^{[3]}} = {A^{[2]}_{\text{drop}}}^{\top} \cdot \delta^{[3]}, \qquad \frac{\partial \mathcal{L}}{\partial b^{[3]}} = \sum_{n} \delta^{[3]}_n$$

$$\delta^{[2]}_{\text{drop}} = \delta^{[3]} \cdot {W^{[3]}}^{\top}$$

### Dropout Backward (layer $l$)

$$\delta^{[l]}_{\text{bn}} = \delta^{[l]}_{\text{drop}} \odot \text{mask}_l \quad /\; (1-p)$$

### Batch Normalization Backward

Let $\delta = \delta^{[l]}_{\text{bn}}$, $N = m \cdot d$ (elements in batch):

$$\frac{\partial \mathcal{L}}{\partial \gamma_l} = \sum_i \delta_i \cdot \hat{z}_i, \qquad \frac{\partial \mathcal{L}}{\partial \beta_l} = \sum_i \delta_i$$

$$\frac{\partial \mathcal{L}}{\partial z_l} = \frac{\gamma_l}{\sqrt{\sigma^2_l + \varepsilon}} \left[ \delta - \frac{1}{N}\sum_i \delta_i - \hat{z}_l \cdot \frac{1}{N}\sum_i \delta_i \hat{z}_i \right]$$

### ReLU Backward

$$\delta^{[l]}_{\text{pre-bn}} = \delta^{[l]}_{\text{bn}} \odot \mathbf{1}[Z^{[l]} > 0]$$

### Hidden Dense Layers

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = {A^{[l-1]}}^{\top} \cdot \delta^{[l]}, \qquad \frac{\partial \mathcal{L}}{\partial b^{[l]}} = \sum_n \delta^{[l]}_n$$

$$\delta^{[l-1]} = \delta^{[l]} \cdot {W^{[l]}}^{\top}$$

---

## Adam Optimizer

For each parameter $\theta$ at timestep $t$, given gradient $g_t$:

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta \;\leftarrow\; \theta - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

| Hyperparameter | Value |
|---|---|
| Learning rate α | 0.001 |
| β₁ | 0.9 |
| β₂ | 0.999 |
| ε | 1e-8 |
| LR decay | ×0.95 every 10 epochs |

---

## Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **97.4%** |
| Macro Precision | 97.0% |
| Macro Recall | 97.0% |
| Macro F1-Score | 97.0% |
| Best class (digit 1) | 99.2% F1 |
| Worst class (digit 5) | 95.5% F1 |
| Total Parameters | 235,146 |
| Train Time (CPU) | ~8 min |

### Training Milestones

| Epoch | Train Loss | Val Accuracy |
|---|---|---|
| 1 | 0.4500 | ~88–90% |
| 10 | 0.1200 | ~95–96% |
| 30 | 0.0600 | ~97–98% |
| 50 | 0.0380 | **97.4%** |

---

## Training Curve



![Loss Curve](results/loss_curve.png)




![Accuracy Curve](results/accuracy_curve.png)



---

## Confusion Matrix



![Confusion Matrix](results/confusion_matrix.png)



---

## Visualizations

| Plot | Description |
|---|---|
| `results/loss_curve.png` | Train + val loss vs epoch, best epoch marked |
| `results/accuracy_curve.png` | Train + val accuracy vs epoch |
| `results/confusion_matrix.png` | 10×10 normalized heatmap (n=10,000 test) |
| `results/weight_distributions.png` | He init vs final weight histograms per layer |
| `results/sample_predictions.png` | 5×5 grid — green border=correct, red=wrong |
| `results/per_class_f1.png` | F1 score bar chart per digit class 0–9 |



![Weight Distributions](results/weight_distributions.png)




![Sample Predictions](results/sample_predictions.png)




![Per Class F1](results/per_class_f1.png)



---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/chamanvashishth/neural-net-scratch
cd neural-net-scratch

# 2. Install dependencies (NumPy + plotting only)
pip install -r requirements.txt

# 3. Train (downloads MNIST automatically)
python train.py

# 4. Evaluate on test set
python evaluate.py

# 5. Generate all plots
python visualize.py
```

---

## Training Configuration

```python
EPOCHS      = 50
BATCH_SIZE  = 128
LR          = 0.001
LR_DECAY    = 0.95          # applied every 10 epochs
TRAIN_SIZE  = 60_000
VAL_SIZE    = 5_000         # held out from train split
TEST_SIZE   = 10_000
SEED        = 42
CHECKPOINT  = "models/best_weights.npy"   # saved when val_acc improves
```

MNIST is downloaded automatically via `requests` to `data/` on first run.
The best validation checkpoint is saved as `.npy` weight files under `models/`.

---

## Evaluation

```bash
python evaluate.py
# → loads models/best_weights.npy
# → runs full 10,000-sample test set
# → prints: Accuracy, Precision, Recall, F1 (macro), Confusion Matrix
```

---

## Implementation Rules Satisfied

- **NumPy only** — zero PyTorch, TensorFlow, or sklearn neural net imports
- **No `sklearn` except** `confusion_matrix` utility for evaluation display
- **Manual backpropagation** — all gradients derived and implemented by hand
- **Numerically stable Softmax** — log-sum-exp trick applied
- **Correct Inverted Dropout** — activations scaled by `1/(1-p)` at training time
- **BatchNorm inference mode** — running mean/variance tracked for test-time use
- **He initialization** — all weight matrices initialized with $\mathcal{N}(0,\; \sqrt{2/n_{\text{in}}})$
- **Reproducibility** — `np.random.seed(42)` at top of `train.py`

---

## File Structure

```
neural-net-scratch/
├── src/
│   ├── layers.py           ← Dense, ReLU, BatchNorm, Dropout, Softmax
│   ├── losses.py           ← CrossEntropyLoss (log-sum-exp stable)
│   ├── optimizers.py       ← Adam with bias correction
│   ├── network.py          ← NeuralNetwork (add / forward / backward / update)
│   ├── data_loader.py      ← MNIST fetch, normalize, one-hot encode
│   └── metrics.py          ← accuracy, precision, recall, F1, confusion matrix
├── notebooks/
│   └── math_derivations.ipynb   ← all gradient derivations in LaTeX
├── models/                 ← best_weights.npy saved here
├── results/                ← all plots auto-saved here
├── train.py
├── evaluate.py
├── visualize.py
├── requirements.txt        ← numpy, matplotlib, seaborn, requests, tqdm
└── README.md
```

---

## Dependencies

```
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
requests>=2.28
tqdm>=4.65
```

---

*Built by [Chaman Vashishth](https://github.com/chamanvashishth) · [LinkedIn](https://linkedin.com/in/chaman-vashishth-b227a6387)*
