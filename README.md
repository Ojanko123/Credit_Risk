# Credit Risk Model - PD Modelling, Scorecard & IFRS 9 ECL (Python)

## Overview

An end-to-end credit risk modelling pipeline built on the LendingClub loan dataset. The project implements industry-standard techniques used by banks and financial institutions to assess borrower creditworthiness and measure expected credit losses under IFRS 9, including WoE/IV feature selection, logistic regression PD modelling, XGBoost with hyperparameter tuning, scorecard scaling, calibration analysis, SHAP explainability, PSI monitoring, a prediction function and a full three-scenario Expected Credit Loss framework.

This project directly addresses real-world credit risk questions:

- What is the probability that a borrower will default on their loan?
- Which borrower characteristics are most predictive of default?
- How stable is the model across different time periods and populations?
- What credit score should be assigned to a given borrower?
- What is the portfolio's Expected Credit Loss under IFRS 9, and how does it change under stress?

## Dataset

- **Source:** [LendingClub Issued Loans - Kaggle](https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans)
- **Size:** 759,338 loans × 72 columns
- **Period:** 2016-2017
- **Target variable:** `loan_status` --> binary (1 = Default/Charged Off/Late, 0 = Fully Paid)

## Tools & Libraries

| Library | Purpose |
|---|---|
| Pandas | Data loading, cleaning, feature engineering |
| NumPy | Mathematical operations, array handling |
| Statsmodels | Logistic regression with statistical output (odds ratios, p-values, pseudo R^2) |
| Scikit-learn | Model training, train/test split, evaluation metrics, calibration |
| XGBoost | Gradient boosting model with hyperparameter tuning |
| Matplotlib / Seaborn | Visualisations and confusion matrix heatmaps |
| SciPy | Trimmed mean imputation, Hosmer-Lemeshow test, KS normality test |
| SHAP | XGBoost feature importance and explainability |

## Methodology

### Phase 1 - Data Loading & Exploration

Loaded 759,338 loan records and examined the distribution of loan statuses to understand the composition of the dataset before any modelling decisions.

### Phase 2 - Feature Selection & Target Definition

**Column selection:** Reduced 72 columns to 19 features based on two criteria:

- Available at loan application time (no data leakage from post-approval columns)
- Logical business relevance to default prediction

**Target variable definition:**

| Value | Meaning |
|---|---|
| 0 | Fully Paid - good borrower |
| 1 | Charged Off / Default / Late 16+ days - bad borrower |

"Current" loans are excluded because the outcome is unknown and cannot be labelled. Loans that do not meet the credit policy are included in the appropriate class based on their outcome.

### Phase 3 - Data Cleaning

**Missing value treatment:**

- Numeric columns --> filled with **5% trimmed mean** (removes the top and bottom 2.5% of values before averaging, more robust than the mean against outliers, more informative than the median)
- Categorical columns --> filled with mode (most frequent value)

**Text cleaning:** `emp_length` converted from strings ("10+ years") to integers. `int_rate` and `revol_util` stripped of `%` characters and cast to float.

### Phase 4 - Feature Engineering

Nine features were created from existing columns to capture relationships the raw variables cannot express alone:

| Feature | Formula | Business Meaning |
|---|---|---|
| `loan_to_income` | loan_amnt / annual_inc | Loan size relative to earnings |
| `payment_to_income` | installment / (annual_inc / 12) | Monthly repayment burden |
| `revol_to_income` | revol_bal / annual_inc | Credit card debt relative to income |
| `has_pub_rec` | pub_rec > 0 --> 1 | Any derogatory public record |
| `has_delinq` | delinq_2yrs > 0 --> 1 | Any recent delinquency |
| `high_inq` | inq_last_6mths > 3 --> 1 | Excessive recent credit-seeking |
| `high_revol_util` | revol_util > 80% --> 1 | Near-maxed credit card utilisation |
| `issue_month` | from issue_d | Seasonality - month of loan issue |
| `issue_quarter` | from issue_d | Seasonality - quarter of loan issue |

### Phase 5 - Time-Based Train/Test Split

Credit risk models are time-series problems. A random split ignores temporal structure and can leak future information into training. A deterministic 80/20 split preserving chronological order is used instead.

- Training set: first 80% of loans (by issue date)
- Test set: final 20% of loans
- Train and test default rates are reported separately to verify split integrity

### Phase 6 - WoE Encoding (Logistic Regression Pipeline Only)

Weight of Evidence (WoE) transforms each feature to measure how strongly each group of borrowers predicts default versus non-default. Information Value (IV) summarises total predictive power per feature.

**Critical anti-leakage rule:** WoE bins are computed on training data only. The resulting mapping is applied to the test set, test set information never influences the WoE values. Unseen categories in the test set are assigned a neutral WoE of 0.

Features with IV < 0.02 are excluded.

### Phase 7 - Logistic Regression (WoE Features)

WoE encoding enforces monotonic relationships between features and default probability — a regulatory requirement in real bank models. Evaluated with AUC, Gini, KS statistic, and Brier score.

A second fit via Statsmodels produces full statistical output:

- **Odds ratios:** > 1 increases default probability, < 1 decreases it
- **P-values:** features with p > 0.05 are not significant at the 95% level
- **McFadden Pseudo R^2 and AIC** for model fit assessment

### Phase 8 - XGBoost (One-Hot Encoding, Hyperparameter Tuning)

XGBoost is trained on raw features with One-Hot Encoding - avoiding false ordinal relationships for nominal categories like `purpose`, `home_ownership`, and `grade`. Encoder fitted on training data only.

Hyperparameters tuned via `RandomizedSearchCV` with 3-fold stratified cross-validation over 20 iterations, optimising for AUC.

| Parameter | Range | Purpose |
|---|---|---|
| `n_estimators` | 100-400 | Number of sequential trees |
| `max_depth` | 3-6 | Tree complexity |
| `learning_rate` | 0.01-0.15 | Shrinkage per step |
| `subsample` | 0.7-0.9 | Row sampling (prevents overfitting) |
| `colsample_bytree` | 0.7-0.9 | Feature sampling per tree |
| `scale_pos_weight` | neg/pos ratio | Handles class imbalance natively |

### Phase 9 - Calibration Analysis

Two diagnostics applied to both models:

- **Hosmer-Lemeshow test:** p > 0.05 indicates well-calibrated predictions
- **Calibration by decile:** predicted vs actual default rate by probability decile

### Phase 10 - Credit Scorecard Scaling

Logistic regression probabilities converted to a 300-850 credit score using the industry-standard PDO formula:

```
Score  = Offset + Factor × log-odds
Factor = PDO / ln(2)                          [PDO = 20]
Offset = Base Score − Factor × ln(Base Odds)  [Base Score = 600, Base Odds = 1/19]
```

### Phase 11 - PSI (Model Stability Monitoring)

Population Stability Index compares predicted probability distributions between training and test populations to detect distribution drift.

| PSI Value | Interpretation |
|---|---|
| < 0.10 | Stable - no significant shift |
| 0.10-0.20 | Moderate shift - monitor |
| > 0.20 | Significant shift - consider retraining |

### Phase 12 - SHAP Analysis

SHAP values computed for the tuned XGBoost model. Summary plot shows which features drive default probability and by how much — providing the explainability regulators expect.

### Phase 13 - Prediction Function

Deployment-ready function returning three outputs for any new loan application:

| Output | Description |
|---|---|
| Probability of Default | Model's estimated likelihood of default (0 to 1) |
| Credit Score | Converted to 300-850 via the PDO formula |
| Lending Decision | APPROVE / REVIEW / REJECT |

Decision thresholds: PD < 20% --> Approve, 20–40% --> Review, > 40% --> Reject.

### Phase 14 - Visualisations

## Phase 15 - IFRS 9 Expected Credit Loss Framework

### Overview

Built on top of the PD model, Phase 15 implements an IFRS 9-style ECL framework the same structure used by banks to calculate loan loss provisions under international accounting standards.

IFRS 9 requires banks to recognise forward-looking expected credit losses using:

```
ECL = PD × LGD × EAD
```

### PD - Probability of Default

Taken directly from the calibrated XGBoost model (Phase 8), adjusted per macroeconomic scenario.

### LGD - Loss Given Default

The fraction of the outstanding balance lost if the borrower defaults. LendingClub loans are unsecured consumer credit, where industry LGD typically ranges 60-75%. A tiered assumption is applied by loan size:

| Loan Amount | LGD |
|---|---|
| ≤ £5,000 | 60% |
| £5,001-£15,000 | 65% |
| £15,001-£25,000 | 70% |
| > £25,000 | 75% |

### EAD - Exposure at Default

The outstanding balance at the point of default. A 10% amortisation factor is applied to reflect partial principal repayment before default on average.

```
EAD = loan_amnt × 0.90
```

### Scenario Analysis

Three macroeconomic scenarios with probability weights:

| Scenario | Weight | PD Multiplier | Description |
|---|---|---|---|
| Optimistic | 30% | 0.75× | Benign credit environment, lower defaults |
| Base | 50% | 1.00× | Stable conditions, model PD unchanged |
| Downturn | 20% | 1.50× | Recessionary stress, elevated defaults |

### Probability-Weighted ECL

```
ECL_weighted = 0.30 × ECL_optimistic + 0.50 × ECL_base + 0.20 × ECL_downturn
```

Results reported at portfolio level and broken down by loan grade to identify where credit risk is concentrated.

> **Note:** LGD and EAD are fixed assumptions in this implementation. In a production IFRS 9 model, LGD would be estimated from historical recovery data and EAD from facility-level drawdown models.

---

## Key Results

| Metric | Value |
|---|---|
| Dataset size | 759,338 loans |
| Default rate | ~30% |
| Logistic Regression AUC (WoE) | 0.6933 |
| XGBoost AUC (One-Hot, Tuned) | 0.7100 |
| Mean credit score | ~540 |
| PSI (train vs test) | < 0.10 (stable) |
| IFRS 9 Weighted ECL Rate | run model to generate |

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans)
2. Place the CSV in the same directory as `credit_risk.py`
3. Update the file path in Phase 1
4. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy statsmodels shap
```

5. Run:

```bash
python credit_risk.py
```

## Concepts Demonstrated

- Data leakage prevention in feature selection and WoE encoding
- Robust missing value imputation (5% trimmed mean)
- Feature engineering - ratio features, binary flags, seasonality
- Weight of Evidence (WoE) and Information Value (IV)
- Logistic regression with WoE encoding (industry-standard regulatory model)
- XGBoost with One-Hot encoding and class imbalance handling
- Hyperparameter tuning via `RandomizedSearchCV`
- Model evaluation: AUC, Gini, KS statistic, Brier score
- Calibration analysis: Hosmer-Lemeshow test and decile calibration
- Credit scorecard scaling (PDO method, 300-850 scale)
- Population Stability Index (PSI) for model monitoring
- SHAP explainability for tree-based models
- IFRS 9 ECL framework: PD × LGD × EAD
- Three-scenario forward-looking provision calculation (Optimistic / Base / Downturn)
- Probability-weighted ECL consistent with IFRS 9 accounting standards
- ECL concentration analysis by loan grade
- Production-ready prediction function with consistent transformation enforcement

  Oresti Janko
  Bsc Statistics and Insurance Science - University of Piraeus
  Focus: Financial modelling, Credit risk modelling, Quantitative analysis, Python
