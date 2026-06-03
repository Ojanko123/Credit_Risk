# CREDIT RISK MODEL 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split,
                                      RandomizedSearchCV,
                                      StratifiedKFold)
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from scipy.stats import trim_mean, chi2
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# PHASE 1: DATA LOADING & EXPLORATION
print("=" * 65)
print("PHASE 1 - DATA LOADING & EXPLORATION")
print("=" * 65)

loans = pd.read_csv(
    "C:\\Users\\ojank\\Desktop\\SQL\\lc_2016_2017.csv",
    low_memory=False)

print(f"Raw shape: {loans.shape}")
print("\nLoan Status distribution:")
print(loans['loan_status'].value_counts())


# PHASE 2: FEATURE SELECTION & TARGET DEFINITION

print("\n" + "=" * 65)
print("PHASE 2 - FEATURE SELECTION & TARGET DEFINITION")
print("=" * 65)

# Columns available at application time only (no data leakage)
cols_to_keep = [
    'loan_amnt', 'int_rate', 'grade', 'sub_grade',
    'emp_length', 'home_ownership', 'annual_inc',
    'verification_status', 'purpose', 'dti',
    'delinq_2yrs', 'inq_last_6mths', 'open_acc',
    'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
    'installment', 'issue_d', 'loan_status'
]
cols_to_keep = [c for c in cols_to_keep if c in loans.columns]
loans        = loans[cols_to_keep].copy()

# Filter to completed loans only
loans = loans[loans['loan_status'].isin([
    'Fully Paid', 'Charged Off', 'Default',
    'Late (31-120 days)', 'Late (16-30 days)',
    'Does not meet the credit policy. Status:Charged Off',
    'Does not meet the credit policy. Status:Fully Paid'
])]

loans['target'] = np.where(
    loans['loan_status'].isin([
        'Fully Paid',
        'Does not meet the credit policy. Status:Fully Paid'
    ]), 0, 1)

print(f"\nTarget distribution:")
print(loans['target'].value_counts())
print(f"Default rate: {loans['target'].mean():.2%}")

loans.drop('loan_status', axis=1, inplace=True)


# PHASE 3: CLEANING

print("\n" + "=" * 65)
print("PHASE 3 - DATA CLEANING")
print("=" * 65)

# 5% trimmed mean for numerics
numeric_cols = loans.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    tm = trim_mean(loans[col].dropna(), proportiontocut=0.025)
    loans[col] = loans[col].fillna(tm)

# Mode for categoricals
categorical_cols = loans.select_dtypes(include=['object']).columns
for col in categorical_cols:
    loans[col] = loans[col].fillna(loans[col].mode()[0])

# Clean emp_length
if loans['emp_length'].dtype == object:
    loans['emp_length'] = (loans['emp_length']
                           .str.replace(' years', '')
                           .str.replace(' year',  '')
                           .str.replace('< 1',    '0')
                           .str.replace('10+',    '10'))
    loans['emp_length'] = pd.to_numeric(
        loans['emp_length'], errors='coerce')
    loans['emp_length'] = loans['emp_length'].fillna(
        loans['emp_length'].median())

if loans['int_rate'].dtype == object:
    loans['int_rate'] = (loans['int_rate']
                         .str.replace('%', '')
                         .astype(float))

if loans['revol_util'].dtype == object:
    loans['revol_util'] = (loans['revol_util']
                           .str.replace('%', '')
                           .astype(float))
    loans['revol_util'] = loans['revol_util'].fillna(
        loans['revol_util'].median())

print("Cleaning complete.")
print(f"Shape: {loans.shape}")


# PHASE 4 - FEATURE ENGINEERING

print("\n" + "=" * 65)
print("PHASE 4 - FEATURE ENGINEERING")
print("=" * 65)

loans['loan_to_income']  = loans['loan_amnt'] / (loans['annual_inc'] + 1)
loans['revol_to_income'] = loans['revol_bal'] / (loans['annual_inc'] + 1)
loans['has_pub_rec']     = (loans['pub_rec']      > 0).astype(int)
loans['has_delinq']      = (loans['delinq_2yrs']  > 0).astype(int)
loans['high_inq']        = (loans['inq_last_6mths'] > 3).astype(int)
loans['high_revol_util'] = (loans['revol_util']   > 80).astype(int)

if 'installment' in loans.columns:
    loans['payment_to_income'] = (loans['installment'] /
                                   (loans['annual_inc'] / 12 + 1))
    loans.drop('installment', axis=1, inplace=True)

if 'issue_d' in loans.columns:
    try:
        loans['issue_d']       = pd.to_datetime(loans['issue_d'])
        loans['issue_month']   = loans['issue_d'].dt.month
        loans['issue_quarter'] = loans['issue_d'].dt.quarter
        loans.drop('issue_d', axis=1, inplace=True)
    except:
        loans.drop('issue_d', axis=1, inplace=True)

print(f"Shape after feature engineering: {loans.shape}")


# PHASE 5 - TIME-BASED TRAIN/TEST SPLIT

print("\n" + "=" * 65)
print("PHASE 5 - TIME-BASED TRAIN/TEST SPLIT")
print("=" * 65)

# Credit risk models are time-series problems.
# Using random split ignores temporal structure and can leak
# future information into training.
# We use a deterministic 80/20 split preserving order.


split        = int(len(loans) * 0.80)
train_loans  = loans.iloc[:split].copy()
test_loans   = loans.iloc[split:].copy()

print(f"Training set: {len(train_loans):,} rows")
print(f"Test set:     {len(test_loans):,}  rows")
print(f"Train default rate: {train_loans['target'].mean():.2%}")
print(f"Test default rate:  {test_loans['target'].mean():.2%}")

# PHASE 6: WoE ENCODING (TRAIN ONLY, APPLIED TO TEST)

print("\n" + "=" * 65)
print("PHASE 6 - WoE ENCODING (NO LEAKAGE)")
print("=" * 65)

# WoE bins are computed on TRAINING data only.
# The resulting WoE mapping is then applied to the test set.
# This prevents test set information from influencing WoE values.

def calculate_woe_iv(df, feature, target, bins=10):
    """
    WoE and IV calculation on a given dataframe.
    Always call this on training data only.
    Returns woe_map (bin --> WoE value) and IV score.
    """
    df = df[[feature, target]].copy()
    if df[feature].dtype in [np.float64, np.int64, np.float32,
                               np.int32]:
        try:
            df['bin'] = pd.qcut(df[feature], q=bins,
                                duplicates='drop')
        except:
            df['bin'] = pd.cut(df[feature], bins=bins)
    else:
        df['bin'] = df[feature]

    grouped = df.groupby('bin', observed=True)[target].agg(
        ['sum', 'count'])
    grouped.columns   = ['events', 'total']
    grouped['non_ev'] = grouped['total'] - grouped['events']

    total_ev    = grouped['events'].sum()
    total_nev   = grouped['non_ev'].sum()

    grouped['dist_ev']  = grouped['events'] / (total_ev + 1e-10)
    grouped['dist_nev'] = grouped['non_ev'] / (total_nev + 1e-10)

    grouped['dist_ev']  = grouped['dist_ev'].replace(0, 0.0001)
    grouped['dist_nev'] = grouped['dist_nev'].replace(0, 0.0001)

    grouped['woe'] = np.log(grouped['dist_ev'] /
                             grouped['dist_nev'])
    grouped['iv']  = ((grouped['dist_ev'] -
                        grouped['dist_nev']) * grouped['woe'])
    iv = grouped['iv'].sum()
    return grouped['woe'], iv

# Step 1: Calculate IV on TRAINING set only
print("\nCalculating IV on training set...")
iv_results = {}
features   = [c for c in train_loans.columns if c != 'target']

for feature in features:
    try:
        _, iv = calculate_woe_iv(train_loans, feature, 'target')
        iv_results[feature] = iv
    except Exception as e:
        pass

iv_df = pd.DataFrame.from_dict(
    iv_results, orient='index', columns=['IV'])
iv_df = iv_df.sort_values('IV', ascending=False)

print("\nInformation Values (train set):")
print(iv_df.to_string())

selected_features = iv_df[iv_df['IV'] > 0.02].index.tolist()
print(f"\nSelected {len(selected_features)} features (IV > 0.02)")

# Step 2: Build WoE maps on TRAINING set
print("\nBuilding WoE encodings from training set...")
woe_maps    = {}
train_woe   = pd.DataFrame()
test_woe    = pd.DataFrame()

for feature in selected_features:
    try:
        woe_map, _ = calculate_woe_iv(
            train_loans, feature, 'target')
        woe_maps[feature] = woe_map

        # Apply to TRAIN
        if train_loans[feature].dtype in [
                np.float64, np.int64, np.float32, np.int32]:
            try:
                bins_tr = pd.qcut(
                    train_loans[feature], q=10,
                    duplicates='drop', retbins=False)
                train_woe[feature + '_woe'] = bins_tr.map(woe_map)
            except:
                train_woe[feature + '_woe'] = (
                    train_loans[feature].map(
                        train_loans.groupby(feature)['target'].apply(
                            lambda x: np.log(
                                (x.mean() + 0.0001) /
                                (1 - x.mean() + 0.0001)))))
        else:
            woe_lookup = (
                train_loans.groupby(feature)['target'].apply(
                    lambda x: np.log(
                        (x.mean() + 0.0001) /
                        (1 - x.mean() + 0.0001))))
            train_woe[feature + '_woe'] = (
                train_loans[feature].map(woe_lookup))

        # Apply to TEST using TRAIN WoE map
        if test_loans[feature].dtype in [
                np.float64, np.int64, np.float32, np.int32]:
            try:
                # Use same bin edges from training
                _, bin_edges = pd.qcut(
                    train_loans[feature], q=10,
                    duplicates='drop', retbins=True)
                bins_te = pd.cut(
                    test_loans[feature],
                    bins=bin_edges,
                    include_lowest=True)
                test_woe[feature + '_woe'] = bins_te.map(woe_map)
            except:
                # Fallback: use train WoE lookup
                tr_lookup = (
                    train_loans.groupby(feature)['target'].apply(
                        lambda x: np.log(
                            (x.mean() + 0.0001) /
                            (1 - x.mean() + 0.0001))))
                test_woe[feature + '_woe'] = (
                    test_loans[feature].map(tr_lookup))
        else:
            # Categorical: use train WoE lookup, NaN for unseen
            tr_lookup = (
                train_loans.groupby(feature)['target'].apply(
                    lambda x: np.log(
                        (x.mean() + 0.0001) /
                        (1 - x.mean() + 0.0001))))
            test_woe[feature + '_woe'] = (
                test_loans[feature].map(tr_lookup))
            # Unseen categories --> 0 (neutral WoE) not arbitrary
            test_woe[feature + '_woe'].fillna(0, inplace=True)

    except Exception as e:
        print(f"  Skipping {feature}: {e}")

train_woe['target'] = train_loans['target'].values
test_woe['target']  = test_loans['target'].values

# Convert to numeric and fill NaNs
for col in train_woe.columns:
    train_woe[col] = pd.to_numeric(train_woe[col], errors='coerce')
for col in test_woe.columns:
    test_woe[col] = pd.to_numeric(test_woe[col], errors='coerce')

train_woe.fillna(0, inplace=True)
test_woe.fillna(0, inplace=True)

print(f"\nWoE train shape: {train_woe.shape}")
print(f"WoE test shape:  {test_woe.shape}")

X_train_woe = train_woe.drop('target', axis=1)
y_train_woe = train_woe['target']
X_test_woe  = test_woe.drop('target', axis=1)
y_test_woe  = test_woe['target']

# PHASE 7: LOGISTIC REGRESSION (WoE FEATURES)

print("\n" + "=" * 65)
print("PHASE 7 — LOGISTIC REGRESSION (WoE Features)")
print("=" * 65)

lr = LogisticRegression(max_iter=1000, random_state=42,
                         solver='saga')
lr.fit(X_train_woe, y_train_woe)

lr_probs = lr.predict_proba(X_test_woe)[:, 1]
lr_preds = lr.predict(X_test_woe)
lr_auc   = roc_auc_score(y_test_woe, lr_probs)
lr_gini  = 2 * lr_auc - 1
lr_brier = brier_score_loss(y_test_woe, lr_probs)

# KS statistic = max(TPR - FPR) across thresholds
fpr_lr, tpr_lr, _ = roc_curve(y_test_woe, lr_probs)
lr_ks = max(tpr_lr - fpr_lr)

print(f"\nLogistic Regression Results:")
print(f"  AUC:   {lr_auc:.4f}")
print(f"  Gini:  {lr_gini:.4f}")
print(f"  KS:    {lr_ks:.4f}")
print(f"  Brier: {lr_brier:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_woe, lr_preds))

# Statsmodels for odds ratios and p-values
print("\n Statistical Output (Logistic Regression)")
X_train_sm = sm.add_constant(X_train_woe)
try:
    logit_sm = sm.Logit(y_train_woe, X_train_sm)
    result_sm = logit_sm.fit(method='lbfgs', maxiter=500,
                              disp=False)
    odds_df = pd.DataFrame({
        'Coefficient': result_sm.params,
        'Odds Ratio':  np.exp(result_sm.params),
        'P-value':     result_sm.pvalues
    }).drop('const', errors='ignore')
    print("\nOdds Ratios:")
    print(odds_df.sort_values('Odds Ratio',
                               ascending=False).to_string())
    print(f"\nMcFadden Pseudo R²: {result_sm.prsquared:.4f}")
    print(f"AIC: {result_sm.aic:.2f}")
except Exception as e:
    print(f"Statsmodels failed: {e}")


# PHASE 8: XGBoost (OneHot Encoding - no ordinal assumption)

print("\n" + "=" * 65)
print("PHASE 8 - XGBoost (OneHot Encoding)")
print("=" * 65)

# FIX: Use OneHotEncoding instead of LabelEncoding
# LabelEncoding implies ordinal relationships that don't exist
# for categories like purpose, home_ownership, grade

loans_raw = loans.copy()

# Identify categorical columns for OneHot encoding
cat_cols_ohe = loans_raw.select_dtypes(
    include=['object']).columns.tolist()
num_cols_ohe = [c for c in loans_raw.columns
                if c != 'target' and c not in cat_cols_ohe]

print(f"Categorical features (OneHot): {cat_cols_ohe}")
print(f"Numeric features: {len(num_cols_ohe)}")

# Encode categoricals
ohe = OneHotEncoder(sparse_output=False,
                    handle_unknown='ignore',
                    drop='first')

train_raw = loans_raw.iloc[:split].copy()
test_raw  = loans_raw.iloc[split:].copy()

# Fit encoder on TRAIN only
ohe.fit(train_raw[cat_cols_ohe])

# Transform both
def encode_and_combine(df, ohe, cat_cols, num_cols):
    ohe_arr  = ohe.transform(df[cat_cols])
    ohe_cols = ohe.get_feature_names_out(cat_cols)
    ohe_df   = pd.DataFrame(ohe_arr, columns=ohe_cols,
                              index=df.index)
    return pd.concat([df[num_cols].reset_index(drop=True),
                      ohe_df.reset_index(drop=True)], axis=1)

X_train_ohe = encode_and_combine(
    train_raw, ohe, cat_cols_ohe, num_cols_ohe)
X_test_ohe  = encode_and_combine(
    test_raw,  ohe, cat_cols_ohe, num_cols_ohe)
y_train_ohe = train_raw['target'].values
y_test_ohe  = test_raw['target'].values

X_train_ohe = X_train_ohe.fillna(0)
X_test_ohe  = X_test_ohe.fillna(0)

print(f"\nOneHot encoded shapes:")
print(f"  Train: {X_train_ohe.shape}")
print(f"  Test:  {X_test_ohe.shape}")

# Hyperparameter tuning via RandomizedSearchCV
print("\nTuning XGBoost hyperparameters (RandomizedSearchCV)...")

neg = (y_train_ohe == 0).sum()
pos = (y_train_ohe == 1).sum()
scale = neg / pos

param_dist = {
    'n_estimators':     [100, 200, 300, 400],
    'max_depth':        [3, 4, 5, 6],
    'learning_rate':    [0.01, 0.05, 0.1, 0.15],
    'subsample':        [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
}

xgb_base = XGBClassifier(
    scale_pos_weight=scale,
    random_state=42,
    eval_metric='auc',
    verbosity=0
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rscv = RandomizedSearchCV(
    xgb_base,
    param_distributions=param_dist,
    n_iter=20,
    scoring='roc_auc',
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rscv.fit(X_train_ohe, y_train_ohe)

print(f"Best params: {rscv.best_params_}")
print(f"Best CV AUC: {rscv.best_score_:.4f}")

xgb_best   = rscv.best_estimator_
xgb_probs  = xgb_best.predict_proba(X_test_ohe)[:, 1]
xgb_preds  = xgb_best.predict(X_test_ohe)
xgb_auc    = roc_auc_score(y_test_ohe, xgb_probs)
xgb_gini   = 2 * xgb_auc - 1
xgb_brier  = brier_score_loss(y_test_ohe, xgb_probs)

fpr_xgb, tpr_xgb, _ = roc_curve(y_test_ohe, xgb_probs)
xgb_ks = max(tpr_xgb - fpr_xgb)

print(f"\nXGBoost Results (tuned):")
print(f"  AUC:   {xgb_auc:.4f}")
print(f"  Gini:  {xgb_gini:.4f}")
print(f"  KS:    {xgb_ks:.4f}")
print(f"  Brier: {xgb_brier:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_ohe, xgb_preds))

# PHASE 9: CALIBRATION ANALYSIS

print("\n" + "=" * 65)
print("PHASE 9 - CALIBRATION ANALYSIS")
print("=" * 65)

# Calibration: are predicted probabilities reliable?
# A well-calibrated model should show:
# predicted PD 20% --> actual default rate ~20%

def hosmer_lemeshow(y_true, y_proba, n_bins=10):
    """
    Hosmer-Lemeshow goodness-of-fit test.
    H0: model is well calibrated
    Low p-value --> poor calibration
    """
    df = pd.DataFrame({'prob': y_proba, 'actual': y_true})
    df['decile'] = pd.qcut(df['prob'], q=n_bins,
                            duplicates='drop', labels=False)
    grouped = df.groupby('decile').agg(
        n=('actual', 'count'),
        observed=('actual', 'sum'),
        expected=('prob', 'sum')
    )
    hl_stat = (((grouped['observed'] - grouped['expected'])**2) /
               (grouped['expected'] *
                (1 - grouped['expected']/grouped['n']))).sum()
    p_value = 1 - chi2.cdf(hl_stat, df=n_bins - 2)
    return hl_stat, p_value

def calibration_by_decile(y_true, y_proba, n_bins=10):
    """Predicted vs actual default rate by probability decile."""
    df = pd.DataFrame({'prob': y_proba, 'actual': y_true})
    df['decile'] = pd.qcut(df['prob'], q=n_bins,
                            duplicates='drop', labels=False)
    result = df.groupby('decile').agg(
        avg_predicted=('prob', 'mean'),
        actual_rate=('actual', 'mean'),
        count=('actual', 'count')
    ).reset_index()
    return result

# LR calibration
lr_hl_stat, lr_hl_p   = hosmer_lemeshow(
    y_test_woe.values, lr_probs)
lr_calib_df            = calibration_by_decile(
    y_test_woe.values, lr_probs)

# XGBoost calibration
xgb_hl_stat, xgb_hl_p = hosmer_lemeshow(
    y_test_ohe, xgb_probs)
xgb_calib_df           = calibration_by_decile(
    y_test_ohe, xgb_probs)

print(f"\nHosmer-Lemeshow Test:")
print(f"  LR:      stat={lr_hl_stat:.3f}, p={lr_hl_p:.4f} "
      f"({'PASS' if lr_hl_p > 0.05 else 'FAIL'})")
print(f"  XGBoost: stat={xgb_hl_stat:.3f}, p={xgb_hl_p:.4f} "
      f"({'PASS' if xgb_hl_p > 0.05 else 'FAIL'})")
print("\n(p > 0.05 = well calibrated, fail to reject H0)")

print(f"\nCalibration by decile (LR):")
print(lr_calib_df.to_string(index=False))

# PHASE 10: CREDIT SCORECARD SCALING (Logistic Regression)

print("\n" + "=" * 65)
print("PHASE 10 - CREDIT SCORECARD SCALING")
print("=" * 65)

pdo        = 20
base_score = 600
base_odds  = 1/19
factor     = pdo / np.log(2)
offset     = base_score - factor * np.log(base_odds)

log_odds = np.log(lr_probs / (1 - lr_probs + 1e-10))
scores   = offset + factor * log_odds
scores   = np.clip(scores, 300, 850)

print(f"Credit Score Distribution:")
print(f"  Min:  {scores.min():.0f}")
print(f"  Max:  {scores.max():.0f}")
print(f"  Mean: {scores.mean():.0f}")

# KS test on score distribution
from scipy import stats as scipy_stats
ks_stat, ks_p = scipy_stats.kstest(
    scores, 'norm', args=(scores.mean(), scores.std()))
print(f"\nKS Test on score distribution:")
print(f"  KS={ks_stat:.4f}, p={ks_p:.4f}")
if ks_p > 0.05:
    print("  Scores follow normal distribution")
else:
    print("  Scores do not follow normal distribution")

scores_df = pd.DataFrame({
    'score': scores, 'default': y_test_woe.values})
print(f"\nAverage score by default status:")
print(scores_df.groupby('default')['score'].mean())


# PHASE 11: PSI (MODEL STABILITY)

print("\n" + "=" * 65)
print("PHASE 11 - PSI (MODEL STABILITY)")
print("=" * 65)

# NOTE: PSI here compares train vs test predicted probabilities.
# In production, PSI would compare development sample vs
# a future scoring population to detect distribution drift.

def calculate_psi(expected, actual, bins=10):
    bp  = np.linspace(0, 1, bins + 1)
    e_c = np.histogram(expected, bins=bp)[0]
    a_c = np.histogram(actual,   bins=bp)[0]
    e_p = np.where(e_c/len(expected) == 0, 0.0001,
                   e_c/len(expected))
    a_p = np.where(a_c/len(actual)   == 0, 0.0001,
                   a_c/len(actual))
    psi_vals = (a_p - e_p) * np.log(a_p / e_p)
    return psi_vals.sum(), psi_vals

train_probs_lr = lr.predict_proba(X_train_woe)[:, 1]
psi_score, psi_bins = calculate_psi(train_probs_lr, lr_probs)

print(f"\nPSI (train vs test): {psi_score:.4f}")
if psi_score < 0.1:
    print("Stable - no significant distribution shift")
elif psi_score < 0.2:
    print("Moderate shift - monitor")
else:
    print("Significant shift - consider retraining")


# PHASE 12: SHAP ANALYSIS (XGBoost Explainability)

print("\n" + "=" * 65)
print("PHASE 12 - SHAP ANALYSIS")
print("=" * 65)

try:
    import shap
    print("Computing SHAP values...")
    explainer   = shap.TreeExplainer(xgb_best)
    shap_values = explainer.shap_values(X_test_ohe)
    SHAP_OK     = True
    print("SHAP computation complete.")
except ImportError:
    print("SHAP not installed. Run: pip install shap")
    SHAP_OK = False
except Exception as e:
    print(f"SHAP failed: {e}")
    SHAP_OK = False


# PHASE 13: PREDICTION FUNCTION

print("\n" + "=" * 60)
print("PHASE 13 - PREDICTION FUNCTION")
print("=" * 60)

def predict_new_customer(application, model, ohe_encoder,
                          ohe_cat_cols, ohe_num_cols,
                          offset, factor):
    """
    Predict default probability for a new loan application.
    Uses XGBoost with OneHot encoding.
    Unknown categories → handled by handle_unknown='ignore'
    in the OneHotEncoder (zero vector for unknown categories).
    """
    df = pd.DataFrame([application])

    # Feature engineering
    df['loan_to_income']  = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['revol_to_income'] = df['revol_bal'] / (df['annual_inc'] + 1)
    df['has_pub_rec']     = (df['pub_rec']        > 0).astype(int)
    df['has_delinq']      = (df['delinq_2yrs']    > 0).astype(int)
    df['high_inq']        = (df['inq_last_6mths'] > 3).astype(int)
    df['high_revol_util'] = (df['revol_util']     > 80).astype(int)

    if 'installment' in df.columns:
        df['payment_to_income'] = (df['installment'] /
                                    (df['annual_inc'] / 12 + 1))
        df.drop('installment', axis=1, inplace=True)

    # Encode - unknown categories produce zero vectors (safe)
    missing_cats = [c for c in ohe_cat_cols if c not in df.columns]
    for c in missing_cats:
        df[c] = 'Unknown'

    ohe_arr  = ohe_encoder.transform(df[ohe_cat_cols])
    ohe_cols = ohe_encoder.get_feature_names_out(ohe_cat_cols)
    ohe_df   = pd.DataFrame(ohe_arr, columns=ohe_cols)

    missing_nums = [c for c in ohe_num_cols if c not in df.columns]
    for c in missing_nums:
        df[c] = 0

    X_new = pd.concat([
        df[ohe_num_cols].reset_index(drop=True),
        ohe_df.reset_index(drop=True)
    ], axis=1).fillna(0)

    # Align columns
    X_new = X_new.reindex(columns=X_test_ohe.columns, fill_value=0)

    pd_prob      = model.predict_proba(X_new)[0][1]
    log_odds_new = np.log(pd_prob / (1 - pd_prob + 1e-10))
    credit_score = float(np.clip(offset + factor * log_odds_new,
                                  300, 850))

    if pd_prob < 0.20:
        decision = "APPROVE"
    elif pd_prob < 0.40:
        decision = "REVIEW"
    else:
        decision = "REJECT"

    print("\n" + "=" * 50)
    print("CREDIT DECISION - NEW CUSTOMER")
    print("=" * 50)
    print(f"Loan Amount:   £{application['loan_amnt']:,}")
    print(f"Annual Income: £{application['annual_inc']:,}")
    print(f"DTI:           {application['dti']}%")
    print(f"Grade:         {application['grade']}")
    print(f"\nPD (Probability of Default): {pd_prob:.2%}")
    print(f"Credit Score:                {credit_score:.0f}")
    print(f"Decision:            *** {decision} ***")
    print("=" * 50)
    return pd_prob, credit_score, decision

# Test on two borrower profiles
high_risk = {
    'loan_amnt': 35000, 'int_rate': 24.5, 'grade': 'F',
    'sub_grade': 'F3', 'emp_length': 2,
    'home_ownership': 'RENT', 'annual_inc': 30000,
    'verification_status': 'Not Verified',
    'purpose': 'debt_consolidation', 'dti': 35.0,
    'delinq_2yrs': 2, 'inq_last_6mths': 4, 'open_acc': 6,
    'pub_rec': 1, 'revol_bal': 18000, 'revol_util': 92.0,
    'total_acc': 10, 'issue_month': 6, 'issue_quarter': 2,
}
low_risk = {
    'loan_amnt': 8000, 'int_rate': 6.5, 'grade': 'A',
    'sub_grade': 'A1', 'emp_length': 10,
    'home_ownership': 'MORTGAGE', 'annual_inc': 120000,
    'verification_status': 'Verified',
    'purpose': 'home_improvement', 'dti': 5.0,
    'delinq_2yrs': 0, 'inq_last_6mths': 0, 'open_acc': 12,
    'pub_rec': 0, 'revol_bal': 5000, 'revol_util': 15.0,
    'total_acc': 20, 'issue_month': 3, 'issue_quarter': 1,
}

print("\nHigh Risk Borrower:")
predict_new_customer(high_risk, xgb_best, ohe, cat_cols_ohe,
                     num_cols_ohe, offset, factor)
print("\nLow Risk Borrower:")
predict_new_customer(low_risk, xgb_best, ohe, cat_cols_ohe,
                     num_cols_ohe, offset, factor)


# PHASE 14: VISUALIZATIONS (individual charts)
print("\n" + "=" * 65)
print("PHASE 14 - VISUALIZATIONS")
print("=" * 65)

#  1. ROC Curves
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(fpr_lr,  tpr_lr,  'b-', lw=2,
        label=f'Logistic Regression  (AUC={lr_auc:.4f}, KS={lr_ks:.3f})')
ax.plot(fpr_xgb, tpr_xgb, 'r-', lw=2,
        label=f'XGBoost              (AUC={xgb_auc:.4f}, KS={xgb_ks:.3f})')
ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
ax.set_title('ROC Curve — Logistic Regression vs XGBoost',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chart_01_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: chart_01_roc_curves.png")

# 2 Confusion Matrix - Logistic Regression
fig, ax = plt.subplots(figsize=(7, 6))
cm_lr = confusion_matrix(y_test_woe, lr_preds)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'],
            annot_kws={'size': 14})
ax.set_title('Confusion Matrix - Logistic Regression',
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Actual', fontsize=11)
ax.set_xlabel('Predicted', fontsize=11)
plt.tight_layout()
plt.savefig('chart_02_confusion_lr.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: chart_02_confusion_lr.png")

#3. Confusion Matrix - XGBoost
fig, ax = plt.subplots(figsize=(7, 6))
cm_xgb = confusion_matrix(y_test_ohe, xgb_preds)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'],
            annot_kws={'size': 14})
ax.set_title('Confusion Matrix - XGBoost (Tuned)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Actual', fontsize=11)
ax.set_xlabel('Predicted', fontsize=11)
plt.tight_layout()
plt.savefig('chart_03_confusion_xgb.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: chart_03_confusion_xgb.png")

# 4. Calibration Curves
fig, ax = plt.subplots(figsize=(9, 7))
frac_pos_lr,  mean_pred_lr  = calibration_curve(y_test_woe, lr_probs,  n_bins=10)
frac_pos_xgb, mean_pred_xgb = calibration_curve(y_test_ohe, xgb_probs, n_bins=10)
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(mean_pred_lr,  frac_pos_lr,  'b-o', lw=2,
        label=f'Logistic Regression (Brier={lr_brier:.4f})')
ax.plot(mean_pred_xgb, frac_pos_xgb, 'r-o', lw=2,
        label=f'XGBoost             (Brier={xgb_brier:.4f})')
ax.set_title('Calibration Curve - Predicted PD vs Actual Default Rate',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Mean Predicted Probability', fontsize=11)
ax.set_ylabel('Fraction of Positives', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chart_04_calibration_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: chart_04_calibration_curves.png")
# 5. Calibration by Decile 
fig, ax = plt.subplots(figsize=(11, 7))
x = np.arange(len(lr_calib_df))
w = 0.35
ax.bar(x - w/2, lr_calib_df['avg_predicted'], w,
       color='steelblue', alpha=0.85, label='LR — Predicted')
ax.bar(x - w/2, lr_calib_df['actual_rate'],   w,
       color='steelblue', alpha=0.40, label='LR — Actual',
       edgecolor='black', linewidth=0.8)
ax.bar(x + w/2, xgb_calib_df['avg_predicted'], w,
       color='firebrick', alpha=0.85, label='XGB — Predicted')
ax.bar(x + w/2, xgb_calib_df['actual_rate'],   w,
       color='firebrick', alpha=0.40, label='XGB — Actual',
       edgecolor='black', linewidth=0.8)
ax.set_title('Predicted vs Actual Default Rate by Probability Decile',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Probability Decile', fontsize=11)
ax.set_ylabel('Default Rate', fontsize=11)
ax.set_xticks(x)
ax.legend(fontsize=10, ncol=2)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('chart_05_calibration_decile.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: chart_05_calibration_decile.png")

# 6. Information Value 
fig, ax = plt.subplots(figsize=(9, 7))
iv_plot = iv_df[iv_df['IV'] > 0.02]['IV'].sort_values()
iv_plot.plot(kind='barh', color='steelblue', edgecolor='black',
             alpha=0.8, ax=ax)
ax.set_title('Information Value by Feature (Training Set)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('IV', fontsize=11)
ax.axvline(x=0.02, color='red', linestyle='--',
           linewidth=1.2, label='IV = 0.02 threshold')
ax.axvline(x=0.10, color='orange', linestyle='--',
           linewidth=1.2, label='IV = 0.10 (medium predictor)')
ax.axvline(x=0.30, color='green', linestyle='--',
           linewidth=1.2, label='IV = 0.30 (strong predictor)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('chart_06_information_value.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: chart_06_information_value.png")

# 7. Credit Score Distribution
fig, ax = plt.subplots(figsize=(9, 7))
ax.hist(scores, bins=60, color='steelblue', edgecolor='black',
        alpha=0.75, density=False)
ax.axvline(scores.mean(), color='red', linestyle='--', linewidth=1.5,
           label=f'Mean = {scores.mean():.0f}')
ax.axvline(np.percentile(scores, 25), color='orange', linestyle=':',
           linewidth=1.2, label=f'25th pct = {np.percentile(scores, 25):.0f}')
ax.axvline(np.percentile(scores, 75), color='orange', linestyle=':',
           linewidth=1.2, label=f'75th pct = {np.percentile(scores, 75):.0f}')
ax.set_title('Credit Score Distribution (300–850)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Credit Score', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chart_07_score_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: chart_07_score_distribution.png")

# 8. PSI by PD Bucket
fig, ax = plt.subplots(figsize=(9, 7))
bins_labels = [f"{i*10}–{(i+1)*10}%" for i in range(len(psi_bins))]
colors = ['firebrick' if v > 0.02 else 'steelblue' for v in psi_bins]
ax.bar(bins_labels, psi_bins, color=colors, edgecolor='black', alpha=0.8)
ax.axhline(0, color='black', linewidth=0.8)
ax.axhline(0.02, color='orange', linestyle='--', linewidth=1.2,
           label='PSI contribution = 0.02')
ax.set_title(f'PSI Contribution by PD Bucket  (Total PSI = {psi_score:.4f})',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('PD Bucket', fontsize=11)
ax.set_ylabel('PSI Contribution', fontsize=11)
ax.tick_params(axis='x', rotation=45)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('chart_08_psi.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: chart_08_psi.png")

# 9. SHAP (if available)
if SHAP_OK:
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_ohe, max_display=15, show=False)
    plt.title('SHAP Feature Importance - XGBoost',
              fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('chart_09_shap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: chart_09_shap.png")

print("\n" + "=" * 65)
print("All charts saved individually.")
print("=" * 65)
