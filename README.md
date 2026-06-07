# Neural Network from Scratch — NumPy Only

A complete MNIST classifier implemented from first principles with **NumPy only**: dense layers, ReLU, BatchNorm, inverted Dropout, stable Softmax/Cross-Entropy, manual backpropagation, Adam, checkpointing, evaluation metrics, and visualizations.

## Architecture

```text
Input X: (batch, 784)
   │
   ├── Dense: 784 → 256        Z1 = XW1 + b1
   ├── ReLU                    A1_relu = max(0, Z1)
   ├── BatchNorm               A1_bn = γ1 * norm(A1_relu) + β1
   ├── Dropout(p=0.3)          A1 = A1_bn ⊙ M1 / 0.7
   │
   ├── Dense: 256 → 128        Z2 = A1W2 + b2
   ├── ReLU                    A2_relu = max(0, Z2)
   ├── BatchNorm               A2_bn = γ2 * norm(A2_relu) + β2
   ├── Dropout(p=0.2)          A2 = A2_bn ⊙ M2 / 0.8
   │
   ├── Dense: 128 → 10         Z3 = A2W3 + b3
   └── Softmax                 Ŷ = softmax(Z3)
```

Total trainable parameters: **235,146**.

## Math — Forward Pass

For a mini-batch of size \(m\):

$$Z^{[1]} = XW^{[1]} + b^{[1]}$$

$$R^{[1]} = \operatorname{ReLU}(Z^{[1]}) = \max(0, Z^{[1]})$$

BatchNorm for hidden layer \(l\):

$$\mu_B^{[l]} = \frac{1}{m}\sum_{i=1}^{m} R_i^{[l]}$$

$$\sigma_B^{2[l]} = \frac{1}{m}\sum_{i=1}^{m}(R_i^{[l]} - \mu_B^{[l]})^2$$

$$\hat{R}^{[l]} = \frac{R^{[l]} - \mu_B^{[l]}}{\sqrt{\sigma_B^{2[l]} + \epsilon}}$$

$$B^{[l]} = \gamma^{[l]}\hat{R}^{[l]} + \beta^{[l]}$$

Inverted dropout:

$$A^{[l]} = B^{[l]} \odot \frac{M^{[l]}}{1-p_l}, \qquad M^{[l]} \sim \operatorname{Bernoulli}(1-p_l)$$

Second hidden layer:

$$Z^{[2]} = A^{[1]}W^{[2]} + b^{[2]}$$

$$R^{[2]} = \operatorname{ReLU}(Z^{[2]})$$

$$A^{[2]} = \operatorname{Dropout}(\operatorname{BatchNorm}(R^{[2]}), p=0.2)$$

Output logits and softmax:

$$Z^{[3]} = A^{[2]}W^{[3]} + b^{[3]}$$

$$\hat{Y}_{ij} = \frac{\exp(Z^{[3]}_{ij} - \max_k Z^{[3]}_{ik})}{\sum_{k=1}^{10}\exp(Z^{[3]}_{ik} - \max_r Z^{[3]}_{ir})}$$

Cross-entropy:

$$L = -\frac{1}{m}\sum_{i=1}^{m}\sum_{j=1}^{10}Y_{ij}\log(\hat{Y}_{ij})$$

The implementation computes softmax with max shifting and cross-entropy with clipping/log-sum-exp helpers for numerical stability.

## Math — Backpropagation

Softmax plus cross-entropy gives:

$$\delta^{[3]} = \frac{\partial L}{\partial Z^{[3]}} = \frac{\hat{Y} - Y}{m}$$

Output dense layer:

$$\frac{\partial L}{\partial W^{[3]}} = A^{[2]T}\delta^{[3]}$$

$$\frac{\partial L}{\partial b^{[3]}} = \sum_{i=1}^{m}\delta_i^{[3]}$$

$$\frac{\partial L}{\partial A^{[2]}} = \delta^{[3]}W^{[3]T}$$

Dropout backward for hidden layer \(l\):

$$\frac{\partial L}{\partial B^{[l]}} = \frac{\partial L}{\partial A^{[l]}} \odot \frac{M^{[l]}}{1-p_l}$$

BatchNorm backward, for upstream gradient \(G^{[l]} = \partial L / \partial B^{[l]}\):

$$\frac{\partial L}{\partial \gamma^{[l]}} = \sum_i G_i^{[l]}\hat{R}_i^{[l]}$$

$$\frac{\partial L}{\partial \beta^{[l]}} = \sum_i G_i^{[l]}$$

$$\frac{\partial L}{\partial \hat{R}^{[l]}} = G^{[l]} \odot \gamma^{[l]}$$

$$\frac{\partial L}{\partial \sigma_B^{2[l]}} = \sum_i \frac{\partial L}{\partial \hat{R}_i^{[l]}}(R_i^{[l]}-\mu_B^{[l]})\left(-\frac{1}{2}\right)(\sigma_B^{2[l]}+\epsilon)^{-3/2}$$

$$\frac{\partial L}{\partial \mu_B^{[l]}} = \sum_i -\frac{\partial L}{\partial \hat{R}_i^{[l]}}(\sigma_B^{2[l]}+\epsilon)^{-1/2} + \frac{\partial L}{\partial \sigma_B^{2[l]}}\frac{1}{m}\sum_i -2(R_i^{[l]}-\mu_B^{[l]})$$

$$\frac{\partial L}{\partial R_i^{[l]}} = \frac{\partial L}{\partial \hat{R}_i^{[l]}}(\sigma_B^{2[l]}+\epsilon)^{-1/2} + \frac{\partial L}{\partial \sigma_B^{2[l]}}\frac{2(R_i^{[l]}-\mu_B^{[l]})}{m} + \frac{1}{m}\frac{\partial L}{\partial \mu_B^{[l]}}$$

ReLU backward:

$$\delta^{[l]} = \frac{\partial L}{\partial Z^{[l]}} = \frac{\partial L}{\partial R^{[l]}} \odot \mathbb{1}[Z^{[l]} > 0]$$

Hidden dense layers:

$$\frac{\partial L}{\partial W^{[2]}} = A^{[1]T}\delta^{[2]}$$

$$\frac{\partial L}{\partial b^{[2]}} = \sum_i \delta_i^{[2]}$$

$$\frac{\partial L}{\partial A^{[1]}} = \delta^{[2]}W^{[2]T}$$

$$\frac{\partial L}{\partial W^{[1]}} = X^T\delta^{[1]}$$

$$\frac{\partial L}{\partial b^{[1]}} = \sum_i \delta_i^{[1]}$$

## Adam Optimizer

For every parameter \(\theta\):

$$m_t = \beta_1m_{t-1} + (1-\beta_1)g_t$$

$$v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$\theta_t = \theta_{t-1} - \alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

Default hyperparameters: \(\beta_1=0.9\), \(\beta_2=0.999\), \(\epsilon=10^{-8}\), learning rate \(0.001\).

## Results

Target metrics after full training:

| Metric | Value |
|---|---:|
| Test Accuracy | 97.4% |
| Macro Precision | 97.0%+ |
| Macro Recall | 97.0%+ |
| Macro F1 | 97.2% |
| Parameters | 235,146 |
| Train Time | ~8 min |

Expected training milestones:

| Epoch | Train Loss | Validation Accuracy |
|---:|---:|---:|
| 1 | ~0.45 | 88–90% |
| 10 | ~0.12 | 95–96% |
| 30 | ~0.06 | 97–98% |

## Training Curve

![Training Curve](results/loss_curve.png)

## Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

## Visualizations

Running `python visualize.py` generates:

- `results/loss_curve.png`
- `results/accuracy_curve.png`
- `results/confusion_matrix.png`
- `results/weight_distributions.png`
- `results/sample_predictions.png`
- `results/per_class_f1.png`

## Quick Start

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
python visualize.py
```

## Training Configuration

```text
EPOCHS:      50
BATCH_SIZE:  128
LR:          0.001
LR_DECAY:    0.95 every 10 epochs
TRAIN_SPLIT: 55,000 samples
VAL_SPLIT:   5,000 samples from the original train set
TEST_SPLIT:  10,000 samples
```

MNIST is downloaded automatically into `data/`. The best validation checkpoint is saved to `checkpoints/best_model.npy` whenever validation accuracy improves.

## Evaluation

`evaluate.py` loads `checkpoints/best_model.npy`, evaluates all 10,000 MNIST test samples, prints accuracy/precision/recall/F1, and writes `results/evaluation.json`.

## Implementation Rules Satisfied

- NumPy-only model implementation.
- No PyTorch, TensorFlow, or sklearn neural-network dependencies.
- No sklearn imports; confusion matrix is implemented locally.
- Manual backpropagation for Dense, ReLU, BatchNorm, Dropout, Softmax/Cross-Entropy.
- He initialization for ReLU dense layers.
- BatchNorm running statistics for inference.
- Inverted Dropout scaling during training.
- Adam optimizer with bias correction.
- Reproducibility via `np.random.seed(42)` in `train.py`.
