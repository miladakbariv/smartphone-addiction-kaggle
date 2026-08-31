# Predicting Smartphone Addiction

An end-to-end machine learning project for the Kaggle Playground Series S6E8 binary classification task.

The project focuses on building a strong, reproducible solution from the competition data itself, with particular attention to robust validation, leakage-safe feature engineering, and clear experiment tracking.

## Problem

Predict whether a smartphone user belongs to the addicted class based on behavioral and demographic features such as:

- daily screen time
- weekend screen time
- social media usage
- gaming time
- work/study time
- sleep
- notifications and app opens
- gender
- stress level
- academic/work impact

**Target:** `addicted_label`  
**Evaluation metric:** ROC-AUC

## Dataset

Training data:

- 691,369 rows
- 12 predictive features
- binary target with approximately 70.9% positive class

Test data:

- 296,302 rows

The data contains substantial missingness across several behavioral features, with some differences in missing-value rates between train and test.

## Project Workflow

### 1. Exploratory Data Analysis

The EDA notebook investigates:

- dataset schema and target distribution
- missing-value patterns
- train/test missingness differences
- whether missingness itself carries target signal
- univariate numerical ROC-AUC
- categorical target-rate differences
- feature correlations
- nonlinear relationships between screen time and addiction probability

A key finding was the strongly nonlinear relationship between daily screen time and the target, motivating the use of gradient-boosted tree models.

### 2. Baseline Models

All models were compared using the same 5-fold stratified cross-validation split.

| Model | Mean CV ROC-AUC |
|---|---:|
| Logistic Regression | 0.911449 |
| CatBoost | 0.953460 |
| LightGBM | 0.960023 |
| XGBoost baseline | 0.963626 |

The large gap between Logistic Regression and the boosted-tree models confirmed that nonlinear interactions are important in this dataset.

### 3. Leakage-Safe Target Encoding

Repeated exact numerical values contained useful predictive signal, but a naive row-dependent encoding introduced a train/validation representation mismatch.

The final solution therefore uses **nested cross-fitted exact-value target encoding**:

- inner folds create out-of-fold encodings for each outer training fold
- outer validation rows are encoded only from the corresponding outer training data
- smoothing reduces noise for rare exact values

This improved mean CV ROC-AUC from approximately:

`0.96363 -> 0.96690`

### 4. Structural Feature Engineering

The strongest engineered features model consistency between total screen time and its component activities:

- `screen_components_observed`
- `observed_component_sum`
- `screen_budget_slack`
- `other_screen_complete`

An ablation study showed that `screen_budget_slack` and `other_screen_complete` provided most of the gain.

### 5. Final Independent Model

The final model combines:

- XGBoost
- native categorical handling
- nested cross-fitted exact-value target encoding
- structural screen-time features
- 5-fold out-of-fold validation
- fold-averaged test predictions

## Results

| Experiment | ROC-AUC |
|---|---:|
| Logistic Regression | 0.91145 CV |
| CatBoost | 0.95346 CV |
| LightGBM | 0.96002 CV |
| Raw XGBoost | 0.96363 CV |
| XGBoost + cross-fitted target encoding | 0.96690 CV |
| **Final XGBoost + structural features + cross-fitted TE** | **0.96774 CV** |
| **Kaggle Public Leaderboard — independent model** | **0.96921** |

The leaderboard result reported here comes from the independently developed pipeline in this repository and does not rely on external prediction files.

## What Did Not Help

Several experiments were evaluated and rejected:

- generic ratio/difference feature engineering
- stronger XGBoost regularization
- periodic sine/cosine transformations
- decimal-precision features
- early naive target encoding with representation leakage
- computationally expensive RealMLP experiments

Keeping unsuccessful experiments was useful because it helped separate reproducible improvements from changes that only increased complexity.

## Repository Structure

```text
smartphone-addiction-kaggle/
├── data/
├── notebooks/
│   ├── 01_eda_portfolio.ipynb
│   ├── 02_baseline_portfolio.ipynb
│   └── 03_xgboost_advanced_portfolio.ipynb
├── src/
├── models/
├── submissions/
├── README.md
└── requirements.txt
```

## Reproducibility

The notebooks support both:

- Kaggle competition paths
- a local `../data/` directory

To reproduce locally, place:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

inside the `data/` folder.

## Key Technical Lessons

This project demonstrates:

- stratified cross-validation
- ROC-AUC based model evaluation
- careful train/test distribution checks
- missing-value handling
- gradient boosting with LightGBM, CatBoost, and XGBoost
- leakage-safe target encoding
- nested cross-fitting
- feature ablation
- experiment comparison
- rank-based test prediction averaging

The largest gain came not from increasing model complexity, but from improving the representation of the data while preserving strict validation integrity.

## Tech Stack

Python, pandas, NumPy, scikit-learn, LightGBM, CatBoost, XGBoost, Matplotlib, Jupyter Notebook

