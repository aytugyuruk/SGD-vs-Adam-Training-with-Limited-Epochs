# SGD vs Adam: Training with Limited Epochs

This project investigates how **SGD** and **Adam** optimizers behave when training time is limited.  
Rather than focusing only on final performance, the study analyzes **training dynamics**, **convergence speed**, and **generalization behavior** under strict epoch budgets.

This is a learning-focused project designed to build intuition through **controlled and reproducible experiments**.
You can access the paper [here](paper.pdf).

---

## Motivation

In practice, models are often trained under **tight time or resource constraints**.  
In such settings, optimizer choice can significantly affect not only convergence speed but also generalization.

This project explores a simple question:

> *How do SGD and Adam differ when we only have a few epochs to train?*

---

## Experiment Setup

- **Model:** Multilayer Perceptron (784 → 128 → 10, ReLU)
- **Dataset:** MNIST  
  - 90% training, 10% validation (manual split)
  - Fixed random seed used to ensure identical splits across experiments
- **Optimizers:**  
  - SGD (lr = 0.001)  
  - Adam (lr = 0.001)
- **Learning Rate Schedulers:** Step, Cosine, Exponential
- **Weight Initialization:** Xavier, He
- **Epoch Budgets:** 5, 30

**Total experiments:**  
2 (optimizers) × 2 (initializations) × 3 (schedulers) × 2 (epoch budgets) = **24**

Each experiment differs by only one factor at a time, ensuring **fair comparisons**.

---

## Training Protocol

- Training and validation loss are recorded after every epoch
- Model selection is based on **validation F1 score**
- The model state from the **best validation epoch** is used for final test evaluation
- Test data is never used during training or model selection

All experiments are run with a fixed random seed to ensure **reproducibility**.

---

## Key Observations

- **Adam converges faster**, especially under very small epoch budgets
- **SGD converges more slowly**, but shows a **more stable generalization gap**
- Faster optimization does not necessarily imply better generalization
- These patterns remain consistent across different initializations and schedulers

The focus of this project is not the final metric values, but *how* the optimizers reach them.

---

## Results & Visualizations

The repository includes:
- Training and validation loss curves
- Generalization gap plots (validation loss − training loss)
- Test F1 score comparisons
- Appendix figures covering all 24 experiments

These visualizations help illustrate optimizer behavior beyond a single metric.

---

## Reproducing the Results

To reproduce the experiments and generate the visualizations:

1. **Train the models:**
   ```bash
   python training/train.py
   ```
   This will run all 24 experiment configurations and save checkpoints.

2. **Generate the plots:**
   ```bash
   python analysis/plot.py
   ```
   This will create all visualizations in the `analysis/figures/` directory.

---

## Limitations

- Only one dataset (MNIST) and one model architecture (simple MLP) are used
- Epoch budgets are limited to 5 and 30
- Learning rate is fixed at 0.001

The goal is controlled analysis, not broad generalization.

---

## What I Learned

- Optimizer choice has a clear impact under limited training budgets
- Adam prioritizes speed; SGD prioritizes stability
- Careful experimental design is essential for fair comparisons
- Looking only at final accuracy can hide important training dynamics

---

## Future Work

Possible extensions include:
- Testing on more complex datasets (e.g., CIFAR-10)
- Using deeper architectures (CNNs, ResNets)
- Exploring learning rate sensitivity
- Analyzing optimizer behavior under noisy or imbalanced data

---

## Notes

This project was developed as a **personal learning exercise**.  
Some parts of the implementation were created with AI-assisted coding tools, with
