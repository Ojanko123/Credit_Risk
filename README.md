# Credit Risk Modelling Pipeline
### Probability of Default (PD), Credit Scorecard, Model Calibration & IFRS 9 Expected Credit Loss

## Overview

This project implements an end-to-end credit risk modelling pipeline using the LendingClub loan dataset, following many of the modelling principles used in modern banking and financial institutions.

The objective is to estimate the Probability of Default (PD) of loan applicants, compare an interpretable regulatory-style scorecard against a machine learning model, calibrate predicted probabilities, optimise lending decision thresholds and finally estimate portfolio-level Expected Credit Loss (ECL) under an IFRS 9 framework.

The project demonstrates the complete lifecycle of a credit risk model, from raw data through deployment-ready predictions.


## Business Objectives

The project answers several real-world credit risk questions:

- What is the probability that a borrower will default?
- Which borrower characteristics are most predictive of default?
- How well do Logistic Regression and XGBoost compare?
- Are predicted probabilities well calibrated?
- What lending threshold should be used?
- How stable is the model across different populations?
- What credit score should be assigned to a new borrower?
- What is the Expected Credit Loss (ECL) under different macroeconomic scenarios?

# Dataset

**Source**

LendingClub Issued Loans (Kaggle)

https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans

**Original Dataset**

- 759,338 loans
- 72 variables
- 2016-2017 LendingClub consumer loans

Target variable:

| Value | Meaning |
|-------|---------|
| 0 | Fully Paid |
| 1 | Charged Off / Default / Late (>16 days) |

Loans with status **Current** were removed because their final outcome is unknown.

# Technologies

| Library | Purpose |
|----------|----------|
| Pandas | Data cleaning & manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Model training & evaluation |
| Statsmodels | Logistic Regression statistical analysis |
| XGBoost | Gradient Boosting |
| SHAP | Explainable AI |
| SciPy | Statistical testing |
| Matplotlib | Visualisations |

# Methodology

## Phase 1 - Data Loading & Exploration

- Load LendingClub dataset
- Explore class distribution
- Examine missing values
- Understand borrower characteristics

## Phase 2 - Feature Selection

Columns were selected according to two rules:

- Available at loan origination
- No information leakage

Target variable:

```
1 = Default
0 = Fully Paid
```

Current loans were excluded.

## Phase 3 - Data Cleaning

Missing values:

### Numerical

Filled using **5% Trimmed Mean**

Advantages:

- robust to outliers
- more informative than median
- less biased than mean

### Categorical

Filled using mode.

Additional cleaning included:

- converting employment length to integers
- percentage strings to floats
- date conversion
- 
## Phase 4 - Feature Engineering

Nine business-driven variables were created.

| Feature | Business Interpretation |
|----------|-------------------------|
| loan_to_income | Loan burden |
| payment_to_income | Monthly affordability |
| revol_to_income | Revolving debt burden |
| has_delinq | Previous delinquency |
| has_pub_rec | Public records |
| high_inq | Aggressive credit seeking |
| high_revol_util | High credit utilisation |
| issue_month | Seasonality |
| issue_quarter | Seasonality |

## Phase 5 - Time-Based Train/Test Split

Rather than using a random split, loans were divided chronologically.

Training:
80%

Testing:
20%

This prevents future information leaking into historical observations.


## Phase 6 - Weight of Evidence (WoE)

Weight of Evidence encoding was built using **training data only**.

Information Value (IV) was calculated for every feature.

Variables with

```
IV < 0.02
```

were discarded.

This follows traditional regulatory scorecard development.

## Phase 7 - Logistic Regression

A scorecard model was fitted using WoE-transformed variables.

Evaluation metrics included:

- ROC AUC
- Gini
- KS Statistic
- Brier Score

Additionally:

- Odds Ratios
- p-values
- McFadden Pseudo R^2
- AIC

were produced using Statsmodels.

# Machine Learning Model

## Phase 8 - XGBoost

Categorical variables were One-Hot encoded.

Hyperparameters were optimised using RandomizedSearchCV.

Parameters tuned included:

- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree
- min_child_weight

Evaluation:

- ROC AUC
- Gini
- KS
- Brier Score
- Classification Report

# Probability Calibration

## Phase 8b - Calibration

Tree-based models typically produce poorly calibrated probabilities.

Two calibration techniques were evaluated:

- Platt Scaling
- Isotonic Regression

A dedicated validation dataset was used to avoid leakage.

Calibration quality was evaluated using:

- Calibration Curves
- Brier Score
- Hosmer-Lemeshow Test
- Calibration by Deciles

The best calibration model was automatically selected.

# Threshold Optimisation

Rather than using the default 0.50 threshold, the classification threshold was selected using the validation dataset.

The following metrics were evaluated:

- Precision
- Recall
- F1-score
- Specificity
- Balanced Accuracy

The threshold that maximised F1-score was selected and applied once to the independent test set.

This mirrors production credit decision systems where lending cut-offs are chosen according to business objectives rather than arbitrary defaults.

# Calibration Analysis

Model calibration was analysed using:

- Calibration Curves
- Probability Deciles
- Hosmer-Lemeshow Test
- Brier Score

This provides both graphical and statistical validation of probability quality.

# Credit Scorecard

Predicted probabilities were converted into an industry-style credit score using the standard Points-to-Double-Odds (PDO) methodology.

```
Score = Offset + Factor × log(Odds)
```

Resulting scores ranged approximately between:

```
577-727
```

# Population Stability Index (PSI)

Model stability was assessed by comparing the distribution of predicted probabilities between the training and testing populations.

Interpretation:

| PSI | Meaning |
|------|----------|
| <0.10 | Stable |
| 0.10-0.20 | Moderate Shift |
| >0.20 | Significant Shift |

SHAP values were computed for the calibrated XGBoost model.

SHAP visualisations identify:

- most influential variables
- contribution of each feature
- direction of impact on PD

# Prediction Engine

A production-style prediction function was developed.

For any new applicant it returns:

- Probability of Default
- Credit Score
- Lending Decision

using

- identical preprocessing
- identical feature engineering
- identical calibration
- identical decision threshold

ensuring consistency between training and deployment.

# IFRS 9 Expected Credit Loss Framework

The calibrated Probability of Default model is integrated into an IFRS 9 Expected Credit Loss framework.

The standard formula is applied:

```
ECL = PD × LGD × EAD
```

where

### PD

Model-estimated calibrated probability of default.

### LGD

Tiered assumptions based on loan amount.

| Loan Amount | LGD |
|-------------|------|
| ≤ £5,000 | 60% |
| £5,001–15,000 | 65% |
| £15,001–25,000 | 70% |
| > £25,000 | 75% |

### EAD

Outstanding exposure assuming

```
EAD = Loan Amount × 90%
```

reflecting partial amortisation before default.

---

## Macroeconomic Scenarios

Three forward-looking scenarios were implemented.

| Scenario | Weight | PD Multiplier |
|-----------|---------|---------------|
| Optimistic | 30% | 0.75× |
| Base | 50% | 1.00× |
| Downturn | 20% | 1.50× |

The final IFRS 9 provision is computed as the probability-weighted average of the three scenarios.

Portfolio ECL is also broken down by credit grade.

# Model Results

| Metric | Logistic Regression | XGBoost | XGBoost + Platt |
|---------|--------------------|----------------|----------------|
| ROC AUC | 0.7097 | **0.7188** | **0.7185** |
| Gini | 0.4195 | **0.4376** | **0.4371** |
| KS | 0.3060 | **0.3197** | **0.3152** |
| Brier Score | 0.1905 | 0.2064 | **0.1898** |
| PSI | Stable | Stable | Stable |

### IFRS 9 Portfolio Results

- Portfolio EAD: **£525.4 million**
- Weighted Expected Credit Loss: **£110.8 million**
- Portfolio ECL Rate: **21.1%**

# Visualisations

The pipeline automatically generates:

- ROC Curves
- Confusion Matrices
- Calibration Curves
- Calibration by Decile
- Information Value Rankings
- Credit Score Distribution
- PSI Charts
- SHAP Summary Plot
- IFRS 9 ECL Charts

# Concepts Demonstrated

- Credit Risk Modelling
- Probability of Default (PD)
- Data Leakage Prevention
- Weight of Evidence (WoE)
- Information Value (IV)
- Logistic Regression Scorecards
- Statistical Model Interpretation
- XGBoost
- Hyperparameter Optimisation
- Probability Calibration
- Platt Scaling
- Isotonic Regression
- Threshold Optimisation
- ROC AUC
- Gini
- KS Statistic
- Precision / Recall / F1
- Brier Score
- Hosmer-Lemeshow Test
- Credit Scorecard Scaling
- Population Stability Index (PSI)
- SHAP Explainability
- IFRS 9 Expected Credit Loss
- Macroeconomic Scenario Analysis
- Portfolio Credit Risk Analytics


# How to Run

1. Download the LendingClub dataset from Kaggle.

2. Place the CSV file in the project directory.

3. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels scipy xgboost shap
```

4. Run

```bash
python credit_risk.py
```

---

# Future Improvements

- Bayesian Hyperparameter Optimisation
- Monotonic XGBoost Constraints
- Beta Calibration
- Expected Calibration Error (ECE)
- Stage 1 / Stage 2 IFRS 9 modelling
- Lifetime PD modelling
- Survival Analysis
- LGD model estimation from recoveries
- EAD modelling using Credit Conversion Factors (CCF)
- Interactive dashboard deployment with Streamlit

  Oresti Janko
  Bsc Statistics and Insurance Science - University of Piraeus
  Focus: Financial modelling, Credit risk modelling, Quantitative analysis, Python
