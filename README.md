# Credit Risk Modelling Pipeline
### Probability of Default (PD), Credit Scorecard, Model Calibration & IFRS 9 Expected Credit Loss

## Overview

This project implements an end-to-end credit risk modelling pipeline using the LendingClub loan dataset.

The objective is to estimate borrower Probability of Default (PD), develop a traditional banking-style credit scorecard, compare interpretable statistical modelling against machine learning approaches, calibrate predicted probabilities, optimise lending decision thresholds and estimate portfolio-level Expected Credit Loss (ECL) under an IFRS 9 inspired framework.

The project demonstrates the complete lifecycle of a credit risk model:

Raw data --> Data cleaning --> Feature engineering --> WoE scorecard --> Machine learning model --> Calibration --> Credit scoring --> Portfolio risk estimation.


# Business Objectives

The project addresses several real-world credit risk questions:

- What is the probability that a borrower will default?
- Which borrower characteristics are strongest predictors of credit risk?
- How does a traditional scorecard compare against gradient boosting?
- Are predicted probabilities reliable enough for risk decisions?
- What lending cutoff should be used?
- How stable is the model over time?
- How can PD estimates be translated into credit scores?
- What is the expected portfolio loss under different economic scenarios?

# Dataset

**Source**

LendingClub Issued Loans (Kaggle)

https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans


## Original Dataset

- 759,338 loans
- 72 variables
- LendingClub consumer loans issued during 2016-2017


## Target Definition

The modelling population excludes loans with unknown outcomes.

Current loans are removed because their final repayment status is unavailable.

Target variable:

| Value | Meaning |
|------|---------|
| 0 | Fully Paid |
| 1 | Charged Off / Default / Late (31-120 days) |

Loans classified as **Late (16-30 days)** were excluded because they represent an early delinquency state without a confirmed final outcome.

Final modelling dataset:

- 183,305 loans
- Default rate: 28.69%

# Technologies

| Library | Purpose |
|---------|---------|
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-learn | Modelling and evaluation |
| Statsmodels | Statistical logistic regression analysis |
| XGBoost | Gradient boosting model |
| SHAP | Model explainability |
| SciPy | Statistical testing |
| Matplotlib | Visualisation |

# Methodology

## Phase 1 - Data Cleaning

The dataset was prepared using only information available at loan origination.

Cleaning steps included:

- Missing value treatment
- Percentage conversion
- Employment length transformation
- Date processing
- Removal of leakage variables

Numerical missing values were imputed using a 5% trimmed mean to reduce sensitivity to extreme observations.

Categorical missing values were replaced using the most frequent category.

# Phase 2 - Feature Engineering

Business-driven credit risk variables were created.

| Feature | Interpretation |
|---------|---------------|
| loan_to_income | Loan burden relative to income |
| revol_to_income | Revolving debt burden |
| payment_to_income | Monthly repayment affordability |
| has_delinq | Previous delinquency indicator |
| has_pub_rec | Public record indicator |
| high_inq | Aggressive credit search behaviour |
| high_revol_util | High utilisation indicator |
| issue_month | Origination seasonality |
| issue_quarter | Origination quarter |

After feature engineering:

- 27 total variables

Highly correlated variables were removed using training-set correlations.

Dropped:

- issue_quarter
- payment_to_income

# Phase 3 - Time Based Train/Test Split

A chronological split was used instead of a random split.

This better represents real credit model deployment, where future borrowers must be predicted using historical information.

Training:

- 146,644 observations

Testing:

- 36,661 observations

Default rate:

| Dataset | Default Rate |
|---------|-------------|
| Train | 28.21% |
| Test | 30.62% |

# Phase 4 - Weight of Evidence (WoE) Scorecard

A traditional banking scorecard approach was implemented.

WoE transformations were created using training data only.

Information Value (IV) was calculated for every variable.

Features with:

```
IV < 0.02
```

were removed.

Selected variables:

- 10 features

Strongest predictors:

| Feature | IV |
|---------|----|
| grade | 0.359 |
| int_rate | 0.354 |
| loan_to_income | 0.127 |
| dti | 0.099 |
| revol_util | 0.079 |

# Phase 5 - Logistic Regression Scorecard

A WoE transformed logistic regression model was developed following traditional credit scorecard methodology.

Evaluation metrics:

- ROC AUC
- Gini coefficient
- KS statistic
- Brier score
- Odds ratios
- Statistical significance
- McFadden pseudo R²
- AIC


Results:

| Metric | Logistic Regression |
|--------|---------------------|
| AUC | 0.7109 |
| Gini | 0.4219 |
| KS | 0.3099 |
| Brier Score | 0.1881 |


McFadden pseudo R²:

```
0.0842
```

---

# Phase 6 - XGBoost Machine Learning Model

A gradient boosting model was developed using one-hot encoded variables.

Categorical variables were encoded without imposing artificial ordering.

Hyperparameter optimisation was performed using RandomizedSearchCV.

Optimised parameters:

```
n_estimators = 400
max_depth = 3
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.7
min_child_weight = 1
```

Cross-validation AUC:

```
0.7136
```

---

# Phase 7 - Probability Calibration

Tree-based models often rank borrowers well but their raw probabilities may not represent true default frequencies.

Two calibration approaches were tested:

- Platt Scaling
- Isotonic Regression

Calibration was performed using a separate validation dataset to avoid leakage.

The best calibration method was selected using Brier Score.

Selected method:

```
Platt Scaling
```

# Phase 8 - Threshold Optimisation

The classification threshold was not set to the default 0.50.

A validation dataset was used to evaluate different thresholds based on:

- Precision
- Recall
- F1-score
- Specificity
- Balanced Accuracy


The F1-optimal threshold was:

```
0.29
```

with:

- Precision: 42.3%
- Recall: 66.4%
- F1-score: 51.7%


However, credit risk decisions have asymmetric costs.

Failing to identify a future defaulter is generally more costly than incorrectly rejecting a good borrower.

Therefore, a more conservative threshold was selected:

```
Final Decision Threshold = 0.25
```

Performance:

- Precision: 39.6%
- Recall: 73.6%
- F1-score: 51.5%
- Balanced Accuracy: 64.7%

The threshold sacrifices almost no overall performance while improving default detection.

# Model Comparison

| Model | AUC | Gini | KS | Brier |
|------|-----|------|----|-------|
| Logistic Regression | 0.7109 | 0.4219 | 0.3099 | 0.1881 |
| XGBoost + Platt Scaling | **0.7187** | **0.4374** | **0.3196** | 0.1881 |

XGBoost achieved superior discriminatory power while maintaining comparable probability calibration.

---

# Calibration Diagnostics

Calibration was evaluated using:

- Calibration curves
- Probability deciles
- Brier score
- Hosmer-Lemeshow test


Hosmer-Lemeshow:

```
Logistic Regression: FAIL
XGBoost: FAIL
```

The failure reflects the large sample size and sensitivity of the test rather than poor ranking performance.

Additional calibration plots and decile analysis are provided.


# Credit Scorecard Scaling

Predicted probabilities were converted into an industry-style credit score.

The standard Points-To-Double-Odds methodology was applied:

```
Score = Offset + Factor × log(Good Odds)
```

Parameters:

```
PDO = 50
Base Score = 600
```

The score uses:

```
Good : Bad Odds
```

so safer borrowers receive higher scores.

Final score distribution:

```
Range: 447 - 784
Mean: 609
```

Example predictions:

High risk borrower:

```
PD = 68.44%
Score = 477
Decision = REJECT
```

Low risk borrower:

```
PD = 6.19%
Score = 729
Decision = APPROVE
```

---

# Population Stability Index (PSI)

Model stability was evaluated by comparing predicted risk distributions between training and testing populations.

Interpretation:

| PSI | Meaning |
|-----|---------|
| <0.10 | Stable |
| 0.10-0.20 | Moderate shift |
| >0.20 | Significant shift |

Result:

```
PSI = 0.0048
```

Interpretation:

The model population remained highly stable.

---

# Explainable AI

SHAP analysis was performed on the XGBoost model.

The analysis identifies:

- Most influential variables
- Feature contribution direction
- Borrower risk drivers

# Prediction Engine

A production-style scoring function was developed.

For a new borrower it returns:

- Probability of Default
- Credit Score
- Lending Decision

The function applies:

- identical preprocessing
- identical feature engineering
- identical encoding
- identical scoring methodology

ensuring consistency between development and deployment.

# IFRS 9 Expected Credit Loss Framework

The calibrated PD model was integrated into an IFRS 9 inspired Expected Credit Loss framework.

The calculation follows:

```
ECL = PD × LGD × EAD
```

## Probability of Default (PD)

Obtained from the calibrated XGBoost model.

## Loss Given Default (LGD)

LGD assumptions were assigned based on loan amount:

| Loan Amount | LGD |
|-------------|-----|
| ≤ £5,000 | 60% |
| £5,001–15,000 | 65% |
| £15,001–25,000 | 70% |
| > £25,000 | 75% |

## Exposure at Default (EAD)

Estimated as:

```
EAD = Loan Amount × 90%
```

assuming partial principal repayment before default.

# Macroeconomic Scenario Analysis

Three forward-looking scenarios were implemented:

| Scenario | Weight | PD Multiplier |
|----------|--------|---------------|
| Optimistic | 30% | 0.75x |
| Base | 50% | 1.00x |
| Downturn | 20% | 1.50x |

The final provision is calculated as a probability-weighted expected loss.

---

# IFRS 9 Results

Portfolio:

```
EAD: £514.5 million
```

Weighted Expected Credit Loss:

```
£100.9 million
```

ECL Rate:

```
19.61%
```

The estimated loss rate is elevated because:

- the default definition includes severe delinquency states
- current unresolved loans were excluded
- LGD and EAD assumptions are conservative

# Visualisations Generated

The pipeline produces:

- ROC curves
- Confusion matrices
- Calibration curves
- Calibration by probability decile
- Information Value ranking
- Credit score distribution
- PSI analysis
- SHAP feature importance
- IFRS 9 scenario analysis

# Concepts Demonstrated

- Probability of Default Modelling
- Credit Risk Analytics
- Banking Scorecards
- Weight of Evidence
- Information Value
- Logistic Regression
- XGBoost
- Model Calibration
- Platt Scaling
- Threshold Optimisation
- ROC AUC
- Gini Coefficient
- KS Statistic
- Brier Score
- SHAP Explainability
- Population Stability Index
- Credit Score Scaling
- IFRS 9 Expected Credit Loss
- Scenario Analysis


# How to Run

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels scipy xgboost shap
```

Run:

```bash
python credit_risk.py
```


---

# Future Improvements

Possible extensions:

- Bayesian hyperparameter optimisation
- Monotonic gradient boosting constraints
- Beta calibration
- Expected Calibration Error (ECE)
- Lifetime PD modelling
- Stage 1 / Stage 2 IFRS 9 framework
- Survival analysis
- Dedicated LGD model using recovery data
- EAD modelling using Credit Conversion Factors
- Interactive dashboard deployment with Streamlit


**Oresti Janko**

BSc Statistics and Insurance Science - University of Piraeus

Focus:
Financial Modelling, Credit Risk Modelling, Quantitative Analysis, Python
