# Credit Risk Model - PD Modelling & Scorecard (Python)

## Overview

An end-to-end credit risk modelling pipeline built on the LendingClub loan dataset. The project implements industry-standard techniques used by banks and financial institutions to assess borrower creditworthiness, including robust missing value imputation, WoE/IV feature selection, logistic regression PD modelling, XGBoost with hyperparameter tuning, scorecard scaling, calibration analysis, and model stability monitoring via PSI including as well a prediction function.

This project directly addresses real-world credit risk questions:

- What is the probability that a borrower will default on their loan?
- Which borrower characteristics are most predictive of default?
- How stable is the model across different time periods and populations?
- What credit score should be assigned to a given borrower?
- Should a new application be approved, reviewed, or rejected?

---

## Dataset

- **Source:** [LendingClub Issued Loans — Kaggle](https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans)
- **Size:** 759,338 loans × 72 columns
- **Period:** 2016–2017

## Tools & Libraries

| Library | Purpose |
|---|---|
| Pandas | Data loading, cleaning, feature engineering |
| NumPy | Mathematical operations, array handling |
| Statsmodels | Logistic regression with statistical output (odds ratios, p-values, pseudo R²) |
| Scikit-learn | Model training, train/test split, evaluation metrics, calibration |
| XGBoost | Gradient boosting model with hyperparameter tuning |
| Matplotlib / Seaborn | Visualisations and confusion matrix heatmaps |
| SciPy | Trimmed mean imputation, Hosmer-Lemeshow test, KS normality test |
| SHAP | XGBoost feature importance explainability |

## Methodology

### Phase 1: Data Loading & Exploration

Loaded 759,338 loan records and examined the distribution of loan statuses to understand the composition of the dataset before any modelling decisions.

### Phase 2: Feature Selection & Target Definition

**Column selection:** Reduced 72 columns to 19 features based on two criteria:

- Available at loan application time (no data leakage from post-approval columns)
- Logical business relevance to default prediction

**Target variable definition:**

| Value | Meaning |
|---|---|
| 0 | Fully Paid — good borrower |
| 1 | Charged Off / Default / Late 16+ days - bad borrower |

"Current" loans are excluded because the outcome is unknown and cannot be labelled. Loans that do not meet the credit policy are included in the appropriate class based on their outcome.

### Phase 3: Data Cleaning

**Missing value treatment:**

- Numeric columns --> filled with **5% trimmed mean** (removes the top and bottom 2.5% of values before averaging - more robust than the mean against outliers, more informative than the median)
- Categorical columns --> filled with mode (most frequent value)

### Phase 4: Feature Engineering

Nine features were created from existing columns to capture relationships the raw variables cannot express alone:

| Feature | Formula | Business Meaning |
|---|---|---|
| `loan_to_income` | loan_amnt / annual_inc | Loan size relative to earnings |
| `payment_to_income` | installment / (annual_inc / 12) | Monthly repayment burden |
| `revol_to_income` | revol_bal / annual_inc | Credit card debt relative to income |
| `has_pub_rec` | pub_rec > 0 → 1 | Any derogatory public record |
| `has_delinq` | delinq_2yrs > 0 → 1 | Any recent delinquency |
| `high_inq` | inq_last_6mths > 3 → 1 | Excessive recent credit-seeking |
| `high_revol_util` | revol_util > 80% → 1 | Near-maxed credit card utilisation |
| `issue_month` | from issue_d | Seasonality — month of loan issue |
| `issue_quarter` | from issue_d | Seasonality — quarter of loan issue |

### Phase 5: Time-Based Train/Test Split

Credit risk models are time-series problems. A random split ignores temporal structure and can leak future information into training. A deterministic 80/20 split preserving chronological order is used instead.

- Training set: first 80% of loans (by issue date)
- Test set: final 20% of loans
- Train and test default rates are reported separately to verify split integrity

### Phase 6: WoE Encoding (Logistic Regression Pipeline Only)

Weight of Evidence (WoE) transforms each feature to measure how strongly each group of borrowers predicts default versus non-default. Information Value (IV) summarises total predictive power per feature.

**Critical anti-leakage rule:** WoE bins are computed on training data only. The resulting mapping is applied to the test set - test set information never influences the WoE values.

Features with IV < 0.02 are excluded. Unseen categories in the test set are assigned a neutral WoE of 0.

### Phase 7: Logistic Regression (WoE Features)

WoE encoding enforces monotonic relationships between features and default probability, a regulatory requirement in real bank models. A logistic regression is fitted on the WoE-transformed features and evaluated with AUC, Gini, KS statistic, and Brier score.

A second fit via Statsmodels produces full statistical output:

- **Odds ratios:** an odds ratio > 1 means higher values of that feature increase default probability; < 1 means they decrease it
- **P-values:** features with p > 0.05 are not statistically significant at the 95% confidence level
- **McFadden Pseudo R² and AIC** for model fit assessment

### Phase 8: XGBoost (One-Hot Encoding, Hyperparameter Tuning)

XGBoost is trained on raw features with One-Hot Encoding rather than Label Encoding. One-Hot encoding avoids implying false ordinal relationships for nominal categories like `purpose`, `home_ownership`, and `grade`. The encoder is fitted on training data only and applied to the test set.

Hyperparameters are tuned via `RandomizedSearchCV` with 3-fold stratified cross-validation over 20 iterations, optimising for AUC.

Key parameters searched:

| Parameter | Range | Purpose |
|---|---|---|
| `n_estimators` | 100-400 | Number of sequential trees |
| `max_depth` | 3-6 | Tree complexity |
| `learning_rate` | 0.01-0.15 | Shrinkage per step |
| `subsample` | 0.7-0.9 | Row sampling (prevents overfitting) |
| `colsample_bytree` | 0.7-0.9 | Feature sampling per tree |
| `scale_pos_weight` | neg/pos ratio | Handles class imbalance natively |

### Phase 9: Calibration Analysis

A well-calibrated model should produce predicted probabilities that match observed default rates, if the model says 20% probability of default, roughly 20% of those borrowers should actually default.

Two calibration diagnostics are applied to both models:

- **Hosmer-Lemeshow test:** a goodness-of-fit test where p > 0.05 indicates well-calibrated predictions (fail to reject H₀)
- **Calibration by decile:** predicted vs actual default rate broken down by probability decile, visualised as a bar chart

### Phase 10: Credit Scorecard Scaling

Logistic regression probabilities are converted to a credit score on the 300–850 scale using the industry-standard PDO (Points to Double the Odds) formula:

```
Score  = Offset + Factor × log-odds
Factor = PDO / ln(2)         [PDO = 20]
Offset = Base Score − Factor × ln(Base Odds)   [Base Score = 600, Base Odds = 1/19]
```

A Kolmogorov-Smirnov test verifies the distributional properties of the resulting scores.

### Phase 11: PSI (Model Stability Monitoring)

The Population Stability Index compares the predicted probability distributions between the training and test populations to detect distribution drift.

| PSI Value | Interpretation |
|---|---|
| < 0.10 | Stable - no significant shift |
| 0.10–0.20 | Moderate shift - monitor |
| > 0.20 | Significant shift - consider retraining |

> **Note:** In production, PSI would compare the development population against a future scoring population. Here it compares train vs test predicted probabilities as a proxy.

### Phase 12: SHAP Analysis (XGBoost Explainability)

SHAP (SHapley Additive exPlanations) values are computed for the tuned XGBoost model to explain individual predictions. A summary plot shows which features drive default probability up or down and by how much.

### Phase 13: Prediction Function

A deployment-ready prediction function takes a raw loan application and returns three outputs:

| Output | Description |
|---|---|
| Probability of Default | Model's estimated likelihood of default (0 to 1) |
| Credit Score | Converted to the 300–850 scale via the PDO formula |
| Lending Decision | APPROVE / REVIEW / REJECT based on calibrated thresholds |

**Decision thresholds:**

| PD Probability | Decision |
|---|---|
| < 20% | APPROVE |
| 20%–40% | REVIEW |
| > 40% | REJECT |

These thresholds are business decisions rather than model outputs. In a real institution they would be calibrated based on risk appetite, regulatory requirements, and expected loss targets.

**Critical deployment principle:** whatever transformations were applied during training must be applied identically to new input data — the same feature engineering, the same encoder fitted on training data, and the same column order. Any deviation produces meaningless predictions. This is among the most common failure modes when moving models from research to production.

### Example Output

```
==================================================
CREDIT DECISION - NEW CUSTOMER
==================================================
Loan Amount:       £15,000
Annual Income:     £35,000
DTI:               18.5%
Grade:             B

PD (Probability of Default):  22.77%
Credit Score:                 650
Decision:          *** REVIEW ***
==================================================
```

---

## Key Results

| Metric | Logistic Regression (WoE) | XGBoost (One-Hot, Tuned) |
|---|---|---|
| AUC | 0.6933 | 0.7100 |
| Gini | 0.3866 | 0.4200 |
| KS Statistic | — | — |
| Brier Score | — | — |
| Dataset size | 759,338 loans | |
| Default rate | ~30% | |
| Mean credit score | ~540 | |
| PSI (train vs test) | < 0.10 (stable) | |

XGBoost outperforms logistic regression on AUC because it makes no assumption of linear relationships — it builds decision trees that capture complex non-linear patterns without being constrained by WoE bins.

## Visualisations

The pipeline produces a 12-panel summary chart covering:

- ROC curves for both models with AUC and KS labels
- Confusion matrices (LR and XGBoost)
- Calibration curves with Brier scores
- Predicted vs actual default rate by decile
- Information Value chart (training set features)
- Credit score distribution histogram
- PSI contribution by PD bucket
- Model comparison summary

If SHAP is installed, a separate feature importance plot is saved to `shap_importance.png`.


## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans)
2. Place the CSV in the same directory as `credit_risk.py`
3. Update the file path in Phase 1 of the script
4. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy statsmodels shap
```

5. Run:

```bash
python credit_risk.py
```

---

## Concepts Demonstrated

- Data leakage prevention in feature selection and WoE encoding
- Robust missing value imputation (5% trimmed mean vs median)
- Feature engineering — ratio features, binary flags, seasonality
- Weight of Evidence (WoE) and Information Value (IV) for feature selection
- Logistic regression with WoE encoding (industry-standard regulatory model)
- XGBoost with One-Hot encoding and class imbalance handling
- Hyperparameter tuning via `RandomizedSearchCV` with stratified cross-validation
- Model evaluation: AUC, Gini, KS statistic, Brier score
- Calibration analysis: Hosmer-Lemeshow test and decile calibration
- Credit scorecard scaling (PDO method, 300–850 scale)
- Normality testing (KS test) on score distribution
- Population Stability Index (PSI) for model monitoring
- SHAP explainability for tree-based models
- Production-ready prediction function with consistent transformation enforcement
