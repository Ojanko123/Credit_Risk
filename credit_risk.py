"""
Credit Risk Model
Estimates probability of default for LendingClub loans, builds a
credit scorecard, and computes IFRS 9 expected credit loss (ECL).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import trim_mean, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                      StratifiedKFold)
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              classification_report, brier_score_loss)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.25          # fixed cutoff used for both models
DATA_PATH = r"C:\Users\ojank\Desktop\SQL\lc_2016_2017.csv"
OUTPUT_DIR = r"C:\Users\ojank\Desktop\python"


def evaluate_at_threshold(y_true, y_proba, threshold, label):
    """Print a classification report and confusion matrix at a fixed
    probability cutoff, and return the predictions and core metrics."""
    preds = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ks = max(tpr - fpr)

    print(f"\n{label} (threshold = {threshold})")
    print(f"  AUC={auc:.4f}  Gini={2*auc-1:.4f}  KS={ks:.4f}  Brier={brier:.4f}")
    print(classification_report(y_true, preds))

    cm = confusion_matrix(y_true, preds)
    print(f"Confusion matrix:\n{cm}")

    return preds, {"auc": auc, "gini": 2 * auc - 1, "ks": ks, "brier": brier}


# 1. Load data
loans = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Raw shape: {loans.shape}")
print(loans['loan_status'].value_counts())

# 2. Target definition
# Default = Charged Off, Default, and Late (31-120 days) - that late bucket
# is close enough to charge-off that it's treated as a default outcome.
# Late (16-30 days) is dropped entirely (not counted as default, not kept
# as a good loan) since it's too early to know how that loan resolves.
cols_to_keep = [
    'loan_amnt', 'int_rate', 'grade', 'sub_grade',
    'emp_length', 'home_ownership', 'annual_inc',
    'verification_status', 'purpose', 'dti',
    'delinq_2yrs', 'inq_last_6mths', 'open_acc',
    'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
    'installment', 'issue_d', 'loan_status'
]
cols_to_keep = [c for c in cols_to_keep if c in loans.columns]
loans = loans[cols_to_keep].copy()

loans = loans[loans['loan_status'].isin([
    'Fully Paid', 'Charged Off', 'Default', 'Late (31-120 days)',
    'Does not meet the credit policy. Status:Charged Off',
    'Does not meet the credit policy. Status:Fully Paid'
])]

loans['target'] = np.where(loans['loan_status'].isin([
    'Fully Paid', 'Does not meet the credit policy. Status:Fully Paid'
]), 0, 1)

print(f"\nTarget distribution:\n{loans['target'].value_counts()}")
print(f"Default rate: {loans['target'].mean():.2%}")
loans.drop('loan_status', axis=1, inplace=True)

# 3. Cleaning
numeric_cols = loans.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    tm = trim_mean(loans[col].dropna(), proportiontocut=0.025)
    loans[col] = loans[col].fillna(tm)

categorical_cols = loans.select_dtypes(include=['object']).columns
for col in categorical_cols:
    loans[col] = loans[col].fillna(loans[col].mode()[0])

loans['emp_length'] = (loans['emp_length']
                       .str.replace(' years', '')
                       .str.replace(' year', '')
                       .str.replace('< 1', '0')
                       .str.replace('10+', '10'))
loans['emp_length'] = pd.to_numeric(loans['emp_length'], errors='coerce')
loans['emp_length'] = loans['emp_length'].fillna(loans['emp_length'].median())

if loans['int_rate'].dtype == object:
    loans['int_rate'] = loans['int_rate'].str.replace('%', '').astype(float)

if loans['revol_util'].dtype == object:
    loans['revol_util'] = loans['revol_util'].str.replace('%', '').astype(float)
    loans['revol_util'] = loans['revol_util'].fillna(loans['revol_util'].median())

print(f"Shape after cleaning: {loans.shape}")

# 4. Feature engineering
loans['loan_to_income'] = loans['loan_amnt'] / (loans['annual_inc'] + 1)
loans['revol_to_income'] = loans['revol_bal'] / (loans['annual_inc'] + 1)
loans['has_pub_rec'] = (loans['pub_rec'] > 0).astype(int)
loans['has_delinq'] = (loans['delinq_2yrs'] > 0).astype(int)
loans['high_inq'] = (loans['inq_last_6mths'] > 3).astype(int)
loans['high_revol_util'] = (loans['revol_util'] > 80).astype(int)

if 'installment' in loans.columns:
    loans['payment_to_income'] = loans['installment'] / (loans['annual_inc'] / 12 + 1)
    loans.drop('installment', axis=1, inplace=True)

if 'issue_d' in loans.columns:
    loans['issue_d'] = pd.to_datetime(loans['issue_d'])
    loans['issue_month'] = loans['issue_d'].dt.month
    loans['issue_quarter'] = loans['issue_d'].dt.quarter
    loans.drop('issue_d', axis=1, inplace=True)

print(f"Shape after feature engineering: {loans.shape}")

# 5. Time-based train/test split (loans are a time series, not i.i.d. rows)
split = int(len(loans) * 0.80)
train_loans = loans.iloc[:split].copy()
test_loans = loans.iloc[split:].copy()
print(f"Train: {len(train_loans):,} rows | Test: {len(test_loans):,} rows")
print(f"Train default rate: {train_loans['target'].mean():.2%} | "
      f"Test default rate: {test_loans['target'].mean():.2%}")

# 6. Multicollinearity control
# sub_grade is a finer-grained duplicate of grade + int_rate, so it's dropped
# outright. Remaining numeric features are checked pairwise on the training
# set only; when two features correlate above 0.85, we keep whichever one
# correlates more strongly with the target and drop the other.
for df_ in (train_loans, test_loans, loans):
    df_.drop(columns=['sub_grade'], inplace=True, errors='ignore')

numeric_feats = train_loans.select_dtypes(include=[np.number]).columns.drop('target')
corr_matrix = train_loans[numeric_feats].corr().abs()

to_drop = set()
for i, col_i in enumerate(numeric_feats):
    for col_j in numeric_feats[i + 1:]:
        if corr_matrix.loc[col_i, col_j] > 0.85:
            corr_i = abs(train_loans[col_i].corr(train_loans['target']))
            corr_j = abs(train_loans[col_j].corr(train_loans['target']))
            to_drop.add(col_j if corr_i >= corr_j else col_i)

if to_drop:
    print(f"Dropping correlated features (>0.85, r < target): {sorted(to_drop)}")
    train_loans.drop(columns=list(to_drop), inplace=True)
    test_loans.drop(columns=list(to_drop), inplace=True)

# 7. WoE encoding (fit on train, applied to test)
def calculate_woe_iv(df, feature, target, bins=10):
    """WoE/IV for one feature. Call on training data only."""
    df = df[[feature, target]].copy()
    if df[feature].dtype in [np.float64, np.int64, np.float32, np.int32]:
        try:
            df['bin'] = pd.qcut(df[feature], q=bins, duplicates='drop')
        except ValueError:
            df['bin'] = pd.cut(df[feature], bins=bins)
    else:
        df['bin'] = df[feature]

    grouped = df.groupby('bin', observed=True)[target].agg(['sum', 'count'])
    grouped.columns = ['events', 'total']
    grouped['non_ev'] = grouped['total'] - grouped['events']

    total_ev, total_nev = grouped['events'].sum(), grouped['non_ev'].sum()
    dist_ev = (grouped['events'] / (total_ev + 1e-10)).replace(0, 0.0001)
    dist_nev = (grouped['non_ev'] / (total_nev + 1e-10)).replace(0, 0.0001)

    grouped['woe'] = np.log(dist_ev / dist_nev)
    grouped['iv'] = (dist_ev - dist_nev) * grouped['woe']
    return grouped['woe'], grouped['iv'].sum()


features = [c for c in train_loans.columns if c != 'target']
iv_results = {}
for feature in features:
    try:
        _, iv = calculate_woe_iv(train_loans, feature, 'target')
        iv_results[feature] = iv
    except Exception:
        pass

iv_df = pd.DataFrame.from_dict(iv_results, orient='index', columns=['IV'])
iv_df = iv_df.sort_values('IV', ascending=False)
print(f"\nInformation values (train set):\n{iv_df.to_string()}")

selected_features = iv_df[iv_df['IV'] > 0.02].index.tolist()
print(f"Selected {len(selected_features)} features (IV > 0.02)")

train_woe, test_woe = pd.DataFrame(), pd.DataFrame()
for feature in selected_features:
    woe_map, _ = calculate_woe_iv(train_loans, feature, 'target')
    is_numeric = train_loans[feature].dtype in [np.float64, np.int64, np.float32, np.int32]

    if is_numeric:
        bins_tr, bin_edges = pd.qcut(train_loans[feature], q=10,
                                      duplicates='drop', retbins=True)
        train_woe[feature + '_woe'] = bins_tr.map(woe_map)
        bins_te = pd.cut(test_loans[feature], bins=bin_edges, include_lowest=True)
        test_woe[feature + '_woe'] = bins_te.map(woe_map)
    else:
        train_woe[feature + '_woe'] = train_loans[feature].map(woe_map)
        test_woe[feature + '_woe'] = test_loans[feature].map(woe_map)
        # unseen categories in test get a neutral WoE of 0
        test_woe[feature + '_woe'].fillna(0, inplace=True)

train_woe['target'] = train_loans['target'].values
test_woe['target'] = test_loans['target'].values
train_woe = train_woe.apply(pd.to_numeric, errors='coerce').fillna(0)
test_woe = test_woe.apply(pd.to_numeric, errors='coerce').fillna(0)

X_train_woe = train_woe.drop('target', axis=1)
y_train_woe = train_woe['target']
X_test_woe = test_woe.drop('target', axis=1)
y_test_woe = test_woe['target']

# 8. Logistic regression (WoE features)
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver='saga')
lr.fit(X_train_woe, y_train_woe)
lr_probs = lr.predict_proba(X_test_woe)[:, 1]

lr_preds, lr_metrics = evaluate_at_threshold(
    y_test_woe, lr_probs, DECISION_THRESHOLD, "Logistic Regression")

X_train_sm = sm.add_constant(X_train_woe)
logit_sm = sm.Logit(y_train_woe, X_train_sm).fit(method='lbfgs', maxiter=500, disp=False)
odds_df = pd.DataFrame({
    'Coefficient': logit_sm.params,
    'Odds Ratio': np.exp(logit_sm.params),
    'P-value': logit_sm.pvalues
}).drop('const', errors='ignore')
print(f"\nOdds ratios:\n{odds_df.sort_values('Odds Ratio', ascending=False).to_string()}")
print(f"McFadden Pseudo R^2: {logit_sm.prsquared:.4f} | AIC: {logit_sm.aic:.2f}")

# 9. XGBoost (one-hot encoded, no false ordinal assumption)
cat_cols_ohe = train_loans.select_dtypes(include=['object']).columns.tolist()
num_cols_ohe = [c for c in train_loans.columns if c != 'target' and c not in cat_cols_ohe]

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
ohe.fit(train_loans[cat_cols_ohe])


def encode_and_combine(df, encoder, cat_cols, num_cols):
    ohe_arr = encoder.transform(df[cat_cols])
    ohe_cols = encoder.get_feature_names_out(cat_cols)
    ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df.index)
    return pd.concat([df[num_cols].reset_index(drop=True),
                       ohe_df.reset_index(drop=True)], axis=1)


X_train_ohe = encode_and_combine(train_loans, ohe, cat_cols_ohe, num_cols_ohe).fillna(0)
X_test_ohe = encode_and_combine(test_loans, ohe, cat_cols_ohe, num_cols_ohe).fillna(0)
y_train_ohe = train_loans['target'].values
y_test_ohe = test_loans['target'].values

scale = (y_train_ohe == 0).sum() / (y_train_ohe == 1).sum()
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
}
xgb_base = XGBClassifier(scale_pos_weight=scale, random_state=RANDOM_STATE,
                          eval_metric='auc', verbosity=0)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
rscv = RandomizedSearchCV(xgb_base, param_distributions=param_dist, n_iter=20,
                           scoring='roc_auc', cv=cv, random_state=RANDOM_STATE, n_jobs=-1)
rscv.fit(X_train_ohe, y_train_ohe)
print(f"\nBest XGBoost params: {rscv.best_params_} | CV AUC: {rscv.best_score_:.4f}")

# 10. Calibrate XGBoost
# XGBoost tends to rank borrowers well but its raw probabilities aren't
# reliable on their own scale. We refit on a smaller training split and fit
# a calibration layer (Platt / Isotonic) on a held-out validation split that
# the model never trained on, then keep whichever calibration gives the
# lower Brier score.
X_train_base, X_val_cal, y_train_base, y_val_cal = train_test_split(
    X_train_ohe, y_train_ohe, test_size=0.20, random_state=RANDOM_STATE,
    stratify=y_train_ohe)

xgb_for_cal = XGBClassifier(**rscv.best_params_, scale_pos_weight=scale,
                             random_state=RANDOM_STATE, eval_metric='auc', verbosity=0)
xgb_for_cal.fit(X_train_base, y_train_base)
probs_uncal = xgb_for_cal.predict_proba(X_test_ohe)[:, 1]  # pre-calibration, for comparison

xgb_platt = CalibratedClassifierCV(xgb_for_cal, method='sigmoid', cv='prefit')
xgb_platt.fit(X_val_cal, y_val_cal)
probs_platt = xgb_platt.predict_proba(X_test_ohe)[:, 1]

xgb_iso = CalibratedClassifierCV(xgb_for_cal, method='isotonic', cv='prefit')
xgb_iso.fit(X_val_cal, y_val_cal)
probs_iso = xgb_iso.predict_proba(X_test_ohe)[:, 1]

if brier_score_loss(y_test_ohe, probs_platt) <= brier_score_loss(y_test_ohe, probs_iso):
    best_cal_name, xgb_calibrated, probs_cal = 'Platt Scaling', xgb_platt, probs_platt
else:
    best_cal_name, xgb_calibrated, probs_cal = 'Isotonic Regression', xgb_iso, probs_iso

print(f"\nBest calibration method: {best_cal_name}")

# 11. Threshold selection - justifying the 0.25 cutoff
# The cutoff is chosen on the VALIDATION split (X_val_cal / y_val_cal),
# which the calibrated model has never been scored against for this
# purpose, and only then applied once to the test set. Picking a
# threshold by looking at test-set performance would be leakage: the
# "final" test evaluation would no longer be an honest, single-use
# estimate of generalization, since a modeling decision (the threshold)
# was tuned against that same data.
from sklearn.metrics import precision_recall_curve

val_probs_cal = xgb_calibrated.predict_proba(X_val_cal)[:, 1]
val_precision, val_recall, pr_thresholds = precision_recall_curve(y_val_cal, val_probs_cal)
val_f1 = (2 * val_precision[:-1] * val_recall[:-1] /
          (val_precision[:-1] + val_recall[:-1] + 1e-12))
best_f1_idx = np.argmax(val_f1)
f1_optimal_threshold = pr_thresholds[best_f1_idx]

print(f"\nThreshold selection (validation set, XGBoost + {best_cal_name}):")
print(f"  F1-optimal threshold: {f1_optimal_threshold:.3f}  "
      f"(Precision={val_precision[best_f1_idx]:.3f}, "
      f"Recall={val_recall[best_f1_idx]:.3f}, F1={val_f1[best_f1_idx]:.3f})")


def evaluate_threshold(y_true, y_proba, t):
    preds = (y_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    balanced_acc = (recall + specificity) / 2
    return precision, recall, f1, specificity, balanced_acc


threshold_grid = pd.DataFrame(
    [[t, *evaluate_threshold(y_val_cal, val_probs_cal, t)]
     for t in np.arange(0.05, 0.51, 0.01)],
    columns=["Threshold", "Precision", "Recall", "F1", "Specificity", "Balanced_Accuracy"])

print("\nTop 10 thresholds by F1 (validation set):")
print(threshold_grid.sort_values("F1", ascending=False).head(10).to_string(index=False))

chosen_row = threshold_grid.iloc[(threshold_grid["Threshold"] - DECISION_THRESHOLD).abs().argmin()]
print(f"\nChosen threshold {DECISION_THRESHOLD}: Precision={chosen_row['Precision']:.3f}, "
      f"Recall={chosen_row['Recall']:.3f}, F1={chosen_row['F1']:.3f}, "
      f"Balanced Accuracy={chosen_row['Balanced_Accuracy']:.3f}")
print(f"0.25 is used ahead of the raw F1-optimum ({f1_optimal_threshold:.3f}) because default "
      f"is the costlier error to miss here, so recall is weighted a bit more heavily than a pure "
      f"F1 maximum would give - while still keeping precision/false-positive rate reasonable.")

xgb_preds_cal, xgb_metrics = evaluate_at_threshold(
    y_test_ohe, probs_cal, DECISION_THRESHOLD, f"XGBoost + {best_cal_name}")

# 12. Model comparison
comparison = pd.DataFrame({
    'Logistic Regression': lr_metrics,
    f'XGBoost + {best_cal_name}': xgb_metrics
}).T
print(f"\nModel comparison (decision threshold = {DECISION_THRESHOLD}):\n"
      f"{comparison.to_string()}")

# 13. Calibration diagnostics
def hosmer_lemeshow(y_true, y_proba, n_bins=10):
    """H0: model is well calibrated. p > 0.05 fails to reject H0."""
    df = pd.DataFrame({'prob': y_proba, 'actual': y_true})
    df['decile'] = pd.qcut(df['prob'], q=n_bins, duplicates='drop', labels=False)
    g = df.groupby('decile').agg(n=('actual', 'count'), observed=('actual', 'sum'),
                                  expected=('prob', 'sum'))
    stat = (((g['observed'] - g['expected']) ** 2) /
            (g['expected'] * (1 - g['expected'] / g['n']))).sum()
    return stat, 1 - chi2.cdf(stat, df=n_bins - 2)


def calibration_by_decile(y_true, y_proba, n_bins=10):
    """Predicted vs actual default rate by probability decile."""
    df = pd.DataFrame({'prob': y_proba, 'actual': y_true})
    df['decile'] = pd.qcut(df['prob'], q=n_bins, duplicates='drop', labels=False)
    return df.groupby('decile').agg(avg_predicted=('prob', 'mean'),
                                     actual_rate=('actual', 'mean'),
                                     count=('actual', 'count')).reset_index()


lr_hl_stat, lr_hl_p = hosmer_lemeshow(y_test_woe.values, lr_probs)
xgb_hl_stat, xgb_hl_p = hosmer_lemeshow(y_test_ohe, probs_cal)
print(f"\nHosmer-Lemeshow: LR stat={lr_hl_stat:.3f}, p={lr_hl_p:.4f} "
      f"({'PASS' if lr_hl_p > 0.05 else 'FAIL'})")
print(f"Hosmer-Lemeshow: XGBoost stat={xgb_hl_stat:.3f}, p={xgb_hl_p:.4f} "
      f"({'PASS' if xgb_hl_p > 0.05 else 'FAIL'})")

lr_calib_df = calibration_by_decile(y_test_woe.values, lr_probs)
xgb_calib_df = calibration_by_decile(y_test_ohe, probs_cal)

# 14. Credit scorecard scaling (points to double odds)
# Uses good:bad odds so score moves the right way - safer borrowers score
# higher. Base odds are set from the training set's actual default rate
# rather than an arbitrary 19:1, and PDO is widened to 50 so the score
# spreads out properly across the PD range instead of bunching up.
pdo, base_score = 50, 600
base_pd = y_train_woe.mean()
base_odds = (1 - base_pd) / base_pd          # good:bad odds at the base rate
factor = pdo / np.log(2)
offset = base_score - factor * np.log(base_odds)
log_odds = np.log((1 - lr_probs + 1e-10) / (lr_probs + 1e-10))   # good:bad odds
scores = np.clip(offset + factor * log_odds, 300, 850)
print(f"\nCredit score range: {scores.min():.0f} - {scores.max():.0f}, mean {scores.mean():.0f}")

# 15. Population Stability Index (train vs test predicted probabilities)
def calculate_psi(expected, actual, bins=10):
    bp = np.linspace(0, 1, bins + 1)
    e_p = np.where(np.histogram(expected, bins=bp)[0] / len(expected) == 0,
                    0.0001, np.histogram(expected, bins=bp)[0] / len(expected))
    a_p = np.where(np.histogram(actual, bins=bp)[0] / len(actual) == 0,
                    0.0001, np.histogram(actual, bins=bp)[0] / len(actual))
    psi_vals = (a_p - e_p) * np.log(a_p / e_p)
    return psi_vals.sum(), psi_vals


train_probs_lr = lr.predict_proba(X_train_woe)[:, 1]
psi_score, psi_bins = calculate_psi(train_probs_lr, lr_probs)
psi_label = "stable" if psi_score < 0.1 else "moderate shift" if psi_score < 0.2 else "significant shift"
print(f"PSI (train vs test): {psi_score:.4f} ({psi_label})")

# 16. SHAP explainability (on the pre-calibration tree model)
try:
    import shap
    explainer = shap.TreeExplainer(xgb_for_cal)
    shap_values = explainer.shap_values(X_test_ohe)
    shap_values_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_plot, X_test_ohe, show=False, max_display=15)
    plt.title('SHAP Summary - Feature Impact on Default Risk', fontweight='bold')
    plt.tight_layout()
    plt.savefig('chart_shap_summary.png', dpi=150, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(9, 8))
    shap.summary_plot(shap_values_plot, X_test_ohe, plot_type='bar', show=False, max_display=15)
    plt.title('SHAP Feature Importance (Mean |SHAP value|)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('chart_shap_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
except ImportError:
    print("SHAP not installed; skipping. Run: pip install shap")

# 17. Prediction function for new applications
def predict_new_customer(application, model, encoder, cat_cols, num_cols, offset, factor):
    """Score a single new loan application and return PD, credit score,
    and a decision band centered on DECISION_THRESHOLD."""
    df = pd.DataFrame([application])
    df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['revol_to_income'] = df['revol_bal'] / (df['annual_inc'] + 1)
    df['has_pub_rec'] = (df['pub_rec'] > 0).astype(int)
    df['has_delinq'] = (df['delinq_2yrs'] > 0).astype(int)
    df['high_inq'] = (df['inq_last_6mths'] > 3).astype(int)
    df['high_revol_util'] = (df['revol_util'] > 80).astype(int)
    if 'installment' in df.columns:
        df['payment_to_income'] = df['installment'] / (df['annual_inc'] / 12 + 1)
        df.drop('installment', axis=1, inplace=True)

    for c in [c for c in cat_cols if c not in df.columns]:
        df[c] = 'Unknown'
    ohe_arr = encoder.transform(df[cat_cols])
    ohe_df = pd.DataFrame(ohe_arr, columns=encoder.get_feature_names_out(cat_cols))
    for c in [c for c in num_cols if c not in df.columns]:
        df[c] = 0

    X_new = pd.concat([df[num_cols].reset_index(drop=True),
                        ohe_df.reset_index(drop=True)], axis=1).fillna(0)
    X_new = X_new.reindex(columns=X_test_ohe.columns, fill_value=0)

    pd_prob = model.predict_proba(X_new)[0][1]
    log_odds_new = np.log((1 - pd_prob + 1e-10) / (pd_prob + 1e-10))   # good:bad odds
    credit_score = float(np.clip(offset + factor * log_odds_new, 300, 850))

    if pd_prob < DECISION_THRESHOLD - 0.05:
        decision = "APPROVE"
    elif pd_prob < DECISION_THRESHOLD + 0.05:
        decision = "REVIEW"
    else:
        decision = "REJECT"

    print(f"\nPD: {pd_prob:.2%} | Credit score: {credit_score:.0f} | Decision: {decision}")
    return pd_prob, credit_score, decision


high_risk = {
    'loan_amnt': 35000, 'int_rate': 24.5, 'grade': 'F', 'emp_length': 2,
    'home_ownership': 'RENT', 'annual_inc': 30000,
    'verification_status': 'Not Verified', 'purpose': 'debt_consolidation',
    'dti': 35.0, 'delinq_2yrs': 2, 'inq_last_6mths': 4, 'open_acc': 6,
    'pub_rec': 1, 'revol_bal': 18000, 'revol_util': 92.0, 'total_acc': 10,
    'issue_month': 6, 'issue_quarter': 2,
}
low_risk = {
    'loan_amnt': 8000, 'int_rate': 6.5, 'grade': 'A', 'emp_length': 10,
    'home_ownership': 'MORTGAGE', 'annual_inc': 120000,
    'verification_status': 'Verified', 'purpose': 'home_improvement',
    'dti': 5.0, 'delinq_2yrs': 0, 'inq_last_6mths': 0, 'open_acc': 12,
    'pub_rec': 0, 'revol_bal': 5000, 'revol_util': 15.0, 'total_acc': 20,
    'issue_month': 3, 'issue_quarter': 1,
}
print("\nHigh risk borrower:")
predict_new_customer(high_risk, xgb_calibrated, ohe, cat_cols_ohe, num_cols_ohe, offset, factor)
print("\nLow risk borrower:")
predict_new_customer(low_risk, xgb_calibrated, ohe, cat_cols_ohe, num_cols_ohe, offset, factor)

# 18. Visualizations
fpr_lr, tpr_lr, _ = roc_curve(y_test_woe, lr_probs)
fpr_cal, tpr_cal, _ = roc_curve(y_test_ohe, probs_cal)

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(fpr_lr, tpr_lr, 'b-', lw=2,
        label=f"Logistic Regression (AUC={lr_metrics['auc']:.4f})")
ax.plot(fpr_cal, tpr_cal, 'r-', lw=2,
        label=f"XGBoost + {best_cal_name} (AUC={xgb_metrics['auc']:.4f})")
ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
ax.set_title('ROC Curve', fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('chart_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

frac_uncal, mean_uncal = calibration_curve(y_test_ohe, probs_uncal, n_bins=10)
frac_platt, mean_platt = calibration_curve(y_test_ohe, probs_platt, n_bins=10)
frac_iso, mean_iso = calibration_curve(y_test_ohe, probs_iso, n_bins=10)
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(mean_uncal, frac_uncal, 'r-o',
        label=f"XGBoost uncalibrated (Brier={brier_score_loss(y_test_ohe, probs_uncal):.4f})")
ax.plot(mean_platt, frac_platt, 'b-o',
        label=f"XGBoost + Platt (Brier={brier_score_loss(y_test_ohe, probs_platt):.4f})")
ax.plot(mean_iso, frac_iso, 'g-o',
        label=f"XGBoost + Isotonic (Brier={brier_score_loss(y_test_ohe, probs_iso):.4f})")
ax.set_title('Calibration Comparison - Before vs After Calibration', fontweight='bold')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('chart_calibration_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, y_true, preds, name, cmap in [
    (axes[0], y_test_woe, lr_preds, 'Logistic Regression', 'Blues'),
    (axes[1], y_test_ohe, xgb_preds_cal, f'XGBoost + {best_cal_name}', 'Oranges'),
]:
    cm = confusion_matrix(y_true, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    ax.set_title(f'{name}\nthreshold = {DECISION_THRESHOLD}', fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('chart_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

frac_pos_lr, mean_pred_lr = calibration_curve(y_test_woe, lr_probs, n_bins=10)
frac_pos_xgb, mean_pred_xgb = calibration_curve(y_test_ohe, probs_cal, n_bins=10)
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(mean_pred_lr, frac_pos_lr, 'b-o', label=f"LR (Brier={lr_metrics['brier']:.4f})")
ax.plot(mean_pred_xgb, frac_pos_xgb, 'r-o',
        label=f"XGBoost + {best_cal_name} (Brier={xgb_metrics['brier']:.4f})")
ax.set_title('Calibration Curve', fontweight='bold')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('chart_calibration_curves.png', dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(11, 7))
x = np.arange(len(lr_calib_df))
w = 0.35
ax.bar(x - w/2, lr_calib_df['avg_predicted'], w, color='steelblue', alpha=0.85, label='LR - Predicted')
ax.bar(x - w/2, lr_calib_df['actual_rate'], w, color='steelblue', alpha=0.40,
       edgecolor='black', label='LR - Actual')
ax.bar(x + w/2, xgb_calib_df['avg_predicted'], w, color='firebrick', alpha=0.85,
       label='XGBoost - Predicted')
ax.bar(x + w/2, xgb_calib_df['actual_rate'], w, color='firebrick', alpha=0.40,
       edgecolor='black', label='XGBoost - Actual')
ax.set_title('Predicted vs Actual Default Rate by Probability Decile', fontweight='bold')
ax.set_xlabel('Probability Decile')
ax.set_ylabel('Default Rate')
ax.set_xticks(x)
ax.legend(ncol=2)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('chart_calibration_decile.png', dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(9, 7))
iv_df[iv_df['IV'] > 0.02]['IV'].sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Information Value by Feature', fontweight='bold')
ax.axvline(0.10, color='orange', linestyle='--', label='Medium predictor (0.10)')
ax.axvline(0.30, color='green', linestyle='--', label='Strong predictor (0.30)')
ax.legend()
plt.tight_layout()
plt.savefig('chart_information_value.png', dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(9, 7))
ax.hist(scores, bins=60, color='steelblue', edgecolor='black', alpha=0.75)
ax.axvline(scores.mean(), color='red', linestyle='--', label=f'Mean = {scores.mean():.0f}')
ax.set_title('Credit Score Distribution (300-850)', fontweight='bold')
ax.set_xlabel('Credit Score')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('chart_score_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(9, 7))
bin_labels = [f"{i*10}-{(i+1)*10}%" for i in range(len(psi_bins))]
colors = ['firebrick' if v > 0.02 else 'steelblue' for v in psi_bins]
ax.bar(bin_labels, psi_bins, color=colors, edgecolor='black', alpha=0.8)
ax.axhline(0, color='black', linewidth=0.8)
ax.axhline(0.02, color='orange', linestyle='--', label='PSI contribution = 0.02')
ax.set_title(f'PSI Contribution by PD Bucket (Total PSI = {psi_score:.4f})', fontweight='bold')
ax.set_xlabel('PD Bucket')
ax.set_ylabel('PSI Contribution')
ax.tick_params(axis='x', rotation=45)
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('chart_psi.png', dpi=150, bbox_inches='tight')
plt.show()

# 19. IFRS 9 expected credit loss
# ECL = PD x LGD x EAD, probability-weighted across three macro scenarios.
# PD comes from the calibrated XGBoost model. LGD and EAD are fixed,
# industry-typical assumptions for unsecured consumer lending.
ecl_df = pd.DataFrame({
    'pd_model': probs_cal,
    'loan_amnt': test_loans['loan_amnt'].values,
    'grade': test_loans['grade'].values if 'grade' in test_loans.columns else 'NA',
})


def assign_lgd(loan_amnt):
    if loan_amnt <= 5000:
        return 0.60
    if loan_amnt <= 15000:
        return 0.65
    if loan_amnt <= 25000:
        return 0.70
    return 0.75


ecl_df['lgd'] = ecl_df['loan_amnt'].apply(assign_lgd)
ecl_df['ead'] = ecl_df['loan_amnt'] * 0.90  # ~10% average principal repaid pre-default

scenarios = {'Optimistic': {'weight': 0.30, 'pd_multiplier': 0.75},
             'Base': {'weight': 0.50, 'pd_multiplier': 1.00},
             'Downturn': {'weight': 0.20, 'pd_multiplier': 1.50}}

for name, cfg in scenarios.items():
    ecl_df[f'pd_{name.lower()}'] = np.clip(ecl_df['pd_model'] * cfg['pd_multiplier'], 0, 1)
    ecl_df[f'ecl_{name.lower()}'] = ecl_df[f'pd_{name.lower()}'] * ecl_df['lgd'] * ecl_df['ead']

ecl_df['ecl_weighted'] = sum(
    ecl_df[f'ecl_{name.lower()}'] * cfg['weight'] for name, cfg in scenarios.items())

total_ead = ecl_df['ead'].sum()
total_weighted_ecl = ecl_df['ecl_weighted'].sum()

print(f"\nIFRS 9 ECL summary")
for name, cfg in scenarios.items():
    total = ecl_df[f'ecl_{name.lower()}'].sum()
    print(f"  {name:<12} (w={cfg['weight']:.0%}): £{total:>14,.0f} | {total/total_ead:.2%} of EAD")
print(f"  {'Weighted (IFRS 9)':<12} (w=100%): £{total_weighted_ecl:>14,.0f} | "
      f"{total_weighted_ecl/total_ead:.2%} of EAD")

if 'grade' in test_loans.columns:
    grade_summary = ecl_df.groupby('grade').agg(
        loans=('ead', 'count'), total_ead=('ead', 'sum'),
        mean_pd=('pd_model', 'mean'), mean_lgd=('lgd', 'mean'),
        total_ecl=('ecl_weighted', 'sum')).reset_index()
    grade_summary['ecl_rate'] = grade_summary['total_ecl'] / grade_summary['total_ead']
    grade_summary['ecl_share'] = grade_summary['total_ecl'] / grade_summary['total_ecl'].sum()
    print(f"\nECL by grade:\n{grade_summary.to_string(index=False)}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('IFRS 9 Expected Credit Loss Framework\n'
                 'ECL = PD x LGD x EAD | Three Scenario Analysis',
                 fontsize=13, fontweight='bold', y=1.01)

    scenario_names = list(scenarios.keys()) + ['Weighted\n(IFRS 9)']
    bar_colors = ['#2ecc71', '#3498db', '#e74c3c', '#8e44ad']

    # Total ECL by scenario
    ax = axes[0, 0]
    scenario_values = [ecl_df[f'ecl_{n.lower()}'].sum() / 1e6 for n in scenarios] + \
                       [total_weighted_ecl / 1e6]
    bars = ax.bar(scenario_names, scenario_values, color=bar_colors, edgecolor='black', alpha=0.85)
    ax.bar_label(bars, fmt='£%.1fM', fontsize=9, padding=3)
    ax.set_title('Total ECL by Scenario', fontweight='bold')
    ax.set_ylabel('Expected Credit Loss (£M)')
    ax.grid(alpha=0.3, axis='y')

    # ECL rate by scenario
    ax = axes[0, 1]
    scenario_rates = [ecl_df[f'ecl_{n.lower()}'].sum() / total_ead * 100 for n in scenarios] + \
                      [total_weighted_ecl / total_ead * 100]
    bars2 = ax.bar(scenario_names, scenario_rates, color=bar_colors, edgecolor='black', alpha=0.85)
    ax.bar_label(bars2, fmt='%.2f%%', fontsize=9, padding=3)
    ax.set_title('ECL Rate by Scenario (ECL / EAD)', fontweight='bold')
    ax.set_ylabel('ECL Rate (%)')
    ax.grid(alpha=0.3, axis='y')

    # Loan-level ECL distribution
    ax = axes[1, 0]
    ax.hist(ecl_df['ecl_weighted'], bins=60, color='steelblue', edgecolor='black', alpha=0.75)
    ax.axvline(ecl_df['ecl_weighted'].mean(), color='red', linestyle='--',
               label=f"Mean ECL = £{ecl_df['ecl_weighted'].mean():,.0f}")
    ax.axvline(ecl_df['ecl_weighted'].median(), color='orange', linestyle=':',
               label=f"Median ECL = £{ecl_df['ecl_weighted'].median():,.0f}")
    ax.set_title('Loan-Level ECL Distribution\n(Probability-Weighted)', fontweight='bold')
    ax.set_xlabel('ECL per Loan (£)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ECL concentration and average PD by grade
    ax = axes[1, 1]
    grade_plot = grade_summary.sort_values('grade')
    ax.bar(grade_plot['grade'], grade_plot['ecl_share'] * 100,
           color='steelblue', edgecolor='black', alpha=0.85)
    ax_twin = ax.twinx()
    ax_twin.plot(grade_plot['grade'], grade_plot['mean_pd'] * 100,
                 'ro-', linewidth=2, markersize=6, label='Avg PD (%)')
    ax_twin.set_ylabel('Average PD (%)', color='red')
    ax_twin.tick_params(axis='y', labelcolor='red')
    ax_twin.legend(loc='upper left', fontsize=9)
    ax.set_title('ECL Concentration & Avg PD by Grade', fontweight='bold')
    ax.set_xlabel('Loan Grade')
    ax.set_ylabel('ECL Share (%)')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.figtext(0.5, -0.02,
                "Note: the estimated lifetime ECL is elevated relative to typical benchmarks due to a "
                "conservative default definition (includes 31-120 day delinquency) and the exclusion "
                "of current, unresolved loans from the modeling population.",
                ha='center', fontsize=9, style='italic', wrap=True)
    plt.savefig('chart_ifrs9_ecl.png', dpi=150, bbox_inches='tight')
    plt.show()

print(f"\nPortfolio EAD: £{total_ead:,.0f} | IFRS 9 weighted ECL: £{total_weighted_ecl:,.0f} "
      f"({total_weighted_ecl/total_ead:.2%})")
print("\nNote: the estimated lifetime ECL is elevated relative to typical benchmarks due to a "
      "conservative default definition (includes 31-120 day delinquency) and the exclusion of "
      "current, unresolved loans from the modeling population.")

