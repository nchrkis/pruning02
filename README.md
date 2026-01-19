# Selective Medical Intelligence – Synthetic Data Supplementary Material

This repository contains the **complete supplementary material** for the paper:

> **Selective Medical Intelligence: Optimising AI-Based Breast Cancer Diagnosis Classification Through Adaptive Data Filtering**
> 
> Nicholas Christakis, Panagiotis Tirchas, Dimitris Drikakis
> 
> *Submitted to* **NEUROCOMPUTING** (2026)

The repository includes:
1. The **synthetic data generator** used to produce controlled tabular datasets.
2. A **supplementary PDF** containing all synthetic experiments, including:
   - Mutual Information (MI) analyses
   - SHAP explanations
   - Confusion matrices
   - ROC curves
3. Full instructions to **reproduce the synthetic datasets** used in the study.

---

## 📁 Repository Structure
├── synthetic_data_generator.py

├── Synthetic_data_analysis.pdf

├── README.md

└── LICENSE

### File descriptions

- **`synthetic_data_generator.py`**  
  Python script that generates synthetic tabular datasets with controlled statistical properties (correlation strength, feature relevance, redundancy, and temporal structure).

- **`Synthetic_data_analysis.pdf`**  
  Supplementary material document containing all figures and experimental results based on synthetic data, referenced in the paper.

- **`LICENSE`**  
  Open-source licence governing reuse of the code.

---

## 🧪 Synthetic Data Generator Overview

The synthetic dataset generator is designed to support **controlled feature-selection and robustness experiments**, particularly in the context of **medical AI classification**.

### Supported feature types

The generator creates a mixture of:

- **Informative continuous predictors**  
  Features with controlled marginal association to the target.

- **Categorical predictors**  
  Generated via quantile binning of the target with added noise.

- **Time-dependent predictors**  
  Features with trend + seasonality components.

- **Redundant / noise predictors**  
  Oracle-irrelevant features independent of the target.

- **Optional low-variance features**  
  Used to test robustness to degenerate predictors.

### Target types

- **Binary classification** (default, used in the paper)
- **Continuous regression**

All randomness is controlled via a fixed `random_state` for full reproducibility.

---

## 🔗 Correlation Control (Important Note)

The parameter `correlation_strength` specifies the **intended marginal dependence** between informative features and the target.

Due to finite-sample effects and the iterative adjustment used in the **binary classification case**, the *achieved* empirical correlation may be slightly lower than the requested value.

> **Practical guideline:**  
> To obtain an empirical point-biserial correlation of approximately **0.3**, set:
> ```python
> correlation_strength = 0.4
> ```
> and verify the realised correlations using the provided utilities.

This behaviour is explicitly accounted for in the supplementary analysis.

---

## ▶️ How to Run the Code

### 1. Requirements

Install the required Python packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

Run the script directly:

```bash
python synthetic_data_generator.py
```

This will generate:
**synthetic_feature_selection_dataset.csv**
containing:

Continuous predictors X1 ... Xn

Optional transformed features

Binary target variable Y

---

## 📊 Supplementary PDF Contents

The file **Synthetic_data_analysis.pdf** contains the full synthetic experiment suite used in the paper.

Experimental design

Total features: 20 continuous predictors

Sample size: 10,000 (plus 10% and 1% subsamples)

Oracle-informative features:

k=2

k=10

k=18

Binary outcome: approximately balanced classes

Target correlation: empirical point-biserial ≈ 0.3

Figures included (for each k and sample size)

Mutual Information (MI) estimates

SHAP summary plots for Class 0 and Class 1, all features vs oracle-informative only

Confusion matrices

ROC curves

Each configuration compares:

Training with all features

Training with oracle-informative features only

**Key conclusion supported by the figures**

Feature pruning yields the largest performance gains under data scarcity, with:

Clearer ROC separation

More stable confusion matrices

Concentrated SHAP attributions

As sample size increases, performance differences diminish, indicating that sufficient data allows models to attenuate irrelevant inputs through learning and regularisation.

---

## 📜 Licence

This code is released under the MIT License, allowing reuse, modification, and redistribution with attribution.

---

## 👤 Author

Nicholas Christakis
Institute for Advanced Modelling and Simulation
University of Nicosia, Cyprus

📧 christakis.n@unic.ac.cy

## 📌 Citation

If you use this code or supplementary material, please cite the corresponding NEUROCOMPUTING submission.
