"""
Student Mental Health Early-Screening Support
Dataset : Kaggle "Student Depression Dataset" (hopesb/student-depression-dataset) -> student_depression_dataset.csv
          27,901 rows x 18 cols; target `Depression` (0/1, self-reported survey label, NOT a clinical diagnosis)
Run     : python student_depression_capstone.py student_depression_dataset.csv
Needs   : pandas numpy scikit-learn>=1.2 matplotlib joblib

Unique contributions
 1. Leakage/ethics audit: `suicidal_thoughts` is a symptom of the outcome, not a risk factor. The deployable
    "risk-factor" model excludes it (symptom-augmented model reported as upper bound).
 2. Engineered indices: stress_load, lifestyle_risk, pressure_x_sleeploss, strain_ratio.
 3. Calibrated probabilities + Low/Moderate/High tiers from recall-driven thresholds (chosen on out-of-fold data).
 4. Fairness audit by gender and age band (flag rate, TPR, FPR).
 5. Permutation-importance explainability.
 6. What-if analysis on modifiable factors (associational, NOT causal).
 7. Rule-based safety escalation outside the model for self-reported suicidal thoughts.
 8. Saved model bundle + screen() inference function.
"""
import sys, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_curve, confusion_matrix, classification_report)

warnings.filterwarnings("ignore")
SEED = 42
DATA = sys.argv[1] if len(sys.argv) > 1 else "student_depression_dataset.csv"
OUT = Path("mh_outputs"); OUT.mkdir(exist_ok=True)
RECALL_FLAG, RECALL_HIGH = 0.90, 0.70   # tier thresholds: flag at >=90% recall, "High" at >=70% recall

RENAME = {"Gender": "gender", "Age": "age", "City": "city", "Profession": "profession",
          "Academic Pressure": "academic_pressure", "Work Pressure": "work_pressure", "CGPA": "cgpa",
          "Study Satisfaction": "study_satisfaction", "Job Satisfaction": "job_satisfaction",
          "Sleep Duration": "sleep_duration", "Dietary Habits": "diet", "Degree": "degree",
          "Have you ever had suicidal thoughts ?": "suicidal_thoughts", "Work/Study Hours": "study_hours",
          "Financial Stress": "financial_stress", "Family History of Mental Illness": "family_history",
          "Depression": "depression"}
NUMERIC = ["age", "academic_pressure", "study_satisfaction", "cgpa", "study_hours", "financial_stress"]


# ---------------------------------------------------------------- cleaning / features
def sleep_hours(s):
    s = str(s).lower()
    return (4.5 if "less than 5" in s else 5.5 if "5-6" in s else 7.5 if "7-8" in s
            else 9.0 if "more than 8" in s else np.nan)


def diet_score(s):
    return {"healthy": 0, "moderate": 1, "unhealthy": 2}.get(str(s).strip().lower(), np.nan)


def prep(df):
    """Type conversion + sanity filters (row-wise, safe to apply before the split)."""
    d = df.copy()
    for c in NUMERIC:
        if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")     # financial_stress has '?' in raw data
    d.loc[~d["age"].between(15, 60), "age"] = np.nan
    d.loc[~d["cgpa"].between(0.1, 10), "cgpa"] = np.nan             # assumes 0-10 CGPA scale; zeros = missing
    for c in ["family_history", "suicidal_thoughts"]:
        d[c] = d[c].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    d["sleep_hours"] = d["sleep_duration"].map(sleep_hours)
    d["diet_score"] = d["diet"].map(diet_score)
    d["gender"] = d["gender"].astype(str).str.strip().str.title()
    return d.drop(columns=["sleep_duration", "diet"])


def engineer(d):
    d = d.copy()
    d["stress_load"] = d["academic_pressure"] + d["financial_stress"]
    d["lifestyle_risk"] = ((d["sleep_hours"] < 6).astype(int) + (d["diet_score"] == 2).astype(int)
                           + (d["study_hours"] >= 10).astype(int) + (d["study_satisfaction"] <= 2).astype(int))
    d["pressure_x_sleeploss"] = d["academic_pressure"] * (8 - d["sleep_hours"]).clip(lower=0)
    d["strain_ratio"] = d["stress_load"] / (d["study_satisfaction"] + 1)
    return d


def build(model, X):
    num = list(X.select_dtypes("number").columns)
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                               min_frequency=100, sparse_output=False))]), cat)])
    return Pipeline([("pre", pre), ("clf", model)])


def thr_for_recall(y, p, r):
    _, rec, thr = precision_recall_curve(y, p)
    idx = np.where(rec[:-1] >= r)[0]
    return float(thr[idx[-1]]) if len(idx) else float(thr[0])


# ---------------------------------------------------------------- load
df = pd.read_csv(DATA)
df.columns = df.columns.str.strip()
df = df.rename(columns=RENAME).drop(columns=["id", "city"], errors="ignore")  # city is noisy + a bias proxy
df = df.drop_duplicates()
y_raw = df["depression"].astype(str).str.strip().str.lower().map({"1": 1, "0": 0, "yes": 1, "no": 0, "true": 1, "false": 0})
df = df[y_raw.notna()].copy(); y = y_raw[y_raw.notna()].astype(int)
df = df.drop(columns=["depression"])

near_const = [c for c in df.columns if df[c].astype(str).value_counts(normalize=True).iloc[0] > 0.99]
df = df.drop(columns=near_const)
print(f"Rows: {len(y)} | Depression rate: {y.mean():.2%} | Dropped near-constant cols: {near_const}")

RAW_COLS = list(df.columns)
X_full = engineer(prep(df))
X_risk = X_full.drop(columns=["suicidal_thoughts"])
print("Model features:", list(X_risk.columns))

tr_idx, te_idx = train_test_split(X_full.index, test_size=0.2, stratify=y, random_state=SEED)
ytr, yte = y.loc[tr_idx], y.loc[te_idx]

# ---------------------------------------------------------------- model comparison (5-fold CV, both scenarios)
models = {
    "LogReg": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced",
                                           random_state=SEED, n_jobs=-1),
    "HistGradBoost": HistGradientBoostingClassifier(random_state=SEED),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall", "f1": "f1"}
rows = []
for scen, X in [("symptom-augmented", X_full), ("risk-factor (deployable)", X_risk)]:
    for name, m in models.items():
        s = cross_validate(build(m, X), X.loc[tr_idx], ytr, cv=cv, scoring=scoring)
        rows.append({"scenario": scen, "model": name, **{k: s[f"test_{k}"].mean() for k in scoring}})
cmp_df = pd.DataFrame(rows).round(3)
print("\n5-fold CV on training split:\n", cmp_df.to_string(index=False))
cmp_df.to_csv(OUT / "model_comparison.csv", index=False)

best = cmp_df[cmp_df.scenario.str.startswith("risk")].sort_values("roc_auc", ascending=False).iloc[0]["model"]
print(f"\nSelected model (risk-factor scenario, best CV ROC-AUC): {best}")

# ---------------------------------------------------------------- final calibrated model
Xtr, Xte = X_risk.loc[tr_idx], X_risk.loc[te_idx]
final = CalibratedClassifierCV(build(clone(models[best]), X_risk), method="isotonic", cv=3).fit(Xtr, ytr)
p = final.predict_proba(Xte)[:, 1]

aug = build(clone(models[best]), X_full).fit(X_full.loc[tr_idx], ytr)
p_aug = aug.predict_proba(X_full.loc[te_idx])[:, 1]

print(f"\nTEST  ROC-AUC {roc_auc_score(yte, p):.3f} | PR-AUC {average_precision_score(yte, p):.3f} "
      f"| Brier {brier_score_loss(yte, p):.3f}")
print(f"Symptom audit: ROC-AUC with suicidal_thoughts = {roc_auc_score(yte, p_aug):.3f} "
      f"vs risk-factor only = {roc_auc_score(yte, p):.3f}")

oof = cross_val_predict(CalibratedClassifierCV(build(clone(models[best]), X_risk), method="isotonic", cv=3),
                        Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
t_flag, t_high = thr_for_recall(ytr, oof, RECALL_FLAG), thr_for_recall(ytr, oof, RECALL_HIGH)
t_high = max(t_high, t_flag)
print(f"Thresholds (from out-of-fold train preds): flag = {t_flag:.3f} | high-risk = {t_high:.3f}")

flag = (p >= t_flag).astype(int)
print("\nScreening-mode confusion matrix (rows=true, cols=pred):\n", confusion_matrix(yte, flag))
print(classification_report(yte, flag, target_names=["not depressed", "depressed"], digits=3))

tier = pd.cut(p, [-np.inf, t_flag, t_high, np.inf], labels=["Low", "Moderate", "High"], right=False)
tier_tbl = pd.DataFrame({"tier": tier, "y": yte.values}).groupby("tier", observed=False)["y"].agg(["count", "mean"])
tier_tbl.columns = ["n", "observed_depression_rate"]
print("Risk tiers on test set:\n", tier_tbl.round(3))

# ---------------------------------------------------------------- fairness audit
meta = pd.DataFrame({"gender": X_full.loc[te_idx, "gender"].values,
                     "age_band": pd.cut(X_full.loc[te_idx, "age"], [0, 21, 25, 100],
                                        labels=["<=21", "22-25", "26+"]).astype(str).values})
fair = []
for col in ["gender", "age_band"]:
    for grp in sorted(meta[col].unique()):
        m = (meta[col] == grp).values
        yt, fl = yte.values[m], flag[m]
        fair.append({"attribute": col, "group": grp, "n": int(m.sum()), "flag_rate": fl.mean(),
                     "TPR": fl[yt == 1].mean() if (yt == 1).any() else np.nan,
                     "FPR": fl[yt == 0].mean() if (yt == 0).any() else np.nan})
fair_df = pd.DataFrame(fair).round(3)
print("\nFairness audit:\n", fair_df.to_string(index=False))
fair_df.to_csv(OUT / "fairness_audit.csv", index=False)

# ---------------------------------------------------------------- explainability
pi = permutation_importance(final, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False)
print("\nTop 10 features (permutation importance, drop in ROC-AUC):\n", imp.head(10).round(4))
imp.to_csv(OUT / "feature_importance.csv", header=["importance"])

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(2, 2, figsize=(13, 10))
for pp, lab, ls in [(p, "Risk-factor (deployable)", "-"), (p_aug, "+ suicidal_thoughts", "--")]:
    fpr, tpr, _ = roc_curve(yte, pp)
    ax[0, 0].plot(fpr, tpr, ls, label=f"{lab} AUC={roc_auc_score(yte, pp):.2f}")
ax[0, 0].plot([0, 1], [0, 1], "k:"); ax[0, 0].set(title="ROC - symptom audit", xlabel="FPR", ylabel="TPR"); ax[0, 0].legend()
fr, mp = calibration_curve(yte, p, n_bins=10, strategy="quantile")
ax[0, 1].plot(mp, fr, "o-", label="Calibrated model"); ax[0, 1].plot([0, 1], [0, 1], "k:")
ax[0, 1].set(title="Calibration", xlabel="Predicted probability", ylabel="Observed rate"); ax[0, 1].legend()
imp.head(12)[::-1].plot.barh(ax=ax[1, 0], title="Permutation importance (top 12)")
tier_tbl["observed_depression_rate"].plot.bar(ax=ax[1, 1], title="Observed depression rate by risk tier", rot=0)
ax[1, 1].set_ylim(0, 1)
plt.tight_layout(); plt.savefig(OUT / "capstone_figures.png", dpi=150); plt.close()

# ---------------------------------------------------------------- save + inference
bundle = {"model": final, "t_flag": t_flag, "t_high": t_high, "raw_cols": RAW_COLS, "feat_cols": list(X_risk.columns)}
joblib.dump(bundle, OUT / "mh_screening_model.joblib")

WHAT_IF = {
    "sleep 7-8 hours": lambda r: {**r, "sleep_duration": "7-8 hours"},
    "healthy diet": lambda r: {**r, "diet": "Healthy"},
    "academic pressure -1": lambda r: {**r, "academic_pressure": max(0, float(r.get("academic_pressure", 0)) - 1)},
    "financial stress -1": lambda r: {**r, "financial_stress": max(1, float(r.get("financial_stress", 1)) - 1)},
}


def _prob(b, person):
    r = pd.DataFrame([person]).reindex(columns=b["raw_cols"])
    return float(b["model"].predict_proba(engineer(prep(r))[b["feat_cols"]])[:, 1][0])


def screen(b, person, what_if=True):
    """person: dict with the original column names in snake_case (see RAW_COLS). Missing answers allowed."""
    pr = _prob(b, person)
    out = {"risk_probability": round(pr, 3),
           "tier": "High" if pr >= b["t_high"] else "Moderate" if pr >= b["t_flag"] else "Low"}
    if str(person.get("suicidal_thoughts", "")).strip().lower() == "yes":      # rule-based safety override
        out["ESCALATE"] = ("Self-reported suicidal thoughts: recommend prompt follow-up with a qualified "
                           "professional or local crisis service, regardless of model score.")
    if what_if:
        out["what_if_prob"] = {k: round(_prob(b, f(person)), 3) for k, f in WHAT_IF.items()}
        out["note"] = "What-if values are associations in survey data, not causal effects. Screening aid only."
    return out


demo = df.loc[te_idx[0]].to_dict()
print("\nDemo screen() on one held-out student:", screen(bundle, demo))
print(f"\nSaved to {OUT}/: model, figures, comparison, fairness, importance.")
print("DISCLAIMER: screening/decision-support aid only; not a diagnosis. Label = self-reported survey indicator.")
