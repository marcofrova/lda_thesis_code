from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

#  Paths / Config 
BASE = Path("/Users/marcofrova/Desktop/TESI/Capitolo 3 Empirico")
DATA_XLSX = BASE / "results" / "meetings" / "meeting_dataset_K30_FIX_named.xlsx"
OUT_DIR   = BASE / "results" / "regressions_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Chosen policy-relevant topics (labels must match your renamed columns)
TOPIC_LABELS = [
    "Inflation and Prices",
    "ECB Strategy and Price Stability",
    "Monetary Policy Operations & Liquidity",
    "Unconventional Monetary Policy (QE & Forward Guidance)",
]

def find_topic_col(df: pd.DataFrame, keyword: str) -> str | None:
    """Return the first topic column whose label contains the keyword (case-insensitive)."""
    ks = keyword.lower()
    matches = [c for c in df.columns if c.startswith("Topic ") and ks in c.lower()]
    return matches[0] if matches else None

def main():
    #  Load data 
    assert DATA_XLSX.exists(), f"Missing input file: {DATA_XLSX}"
    df = pd.read_excel(DATA_XLSX, parse_dates=["meeting_date"])
    # Basic sanity
    assert "decision_+7d" in df.columns, "Missing column 'decision_+7d' in dataset."
    assert "period" in df.columns, "Missing column 'period' in dataset."

    #  Build predictors 
    pred_cols = []
    for lbl in TOPIC_LABELS:
        col = find_topic_col(df, lbl)
        if col is None:
            print(f"[WARN] Topic label not found: {lbl}")
        else:
            pred_cols.append(col)

    # period dummy
    df["post2015"] = (df["period"] == "2015–oggi").astype(int)

    X_cols = pred_cols + ["post2015"]
    if len(pred_cols) == 0:
        raise RuntimeError("No topic predictors found. Check labels and dataset columns.")

    print("Predictors used:")
    for c in X_cols:
        print(" -", c)

    # (1) Binary Logit: hike (1) vs cut (0), dropping hold
    print("\n=== (1) Binary Logit: hike vs cut (hold excluded) ===")
    df_bin = df[df["decision_+7d"].isin(["hike", "cut"])].copy()
    df_bin["y_hike"] = (df_bin["decision_+7d"] == "hike").astype(int)

    Xb = df_bin[X_cols].copy()
    yb = df_bin["y_hike"].copy()

    # Statsmodels (coeffs are scale-dependent; that's fine for ORs/AMEs)
    Xb_sm = sm.add_constant(Xb)
    logit_mod = sm.Logit(yb, Xb_sm)
    logit_res = logit_mod.fit(disp=False)

    # Odds Ratios (OR) with 95% CI
    params = logit_res.params
    conf   = logit_res.conf_int()
    or_table = pd.DataFrame({
        "feature": params.index,
        "OR": np.exp(params.values),
        "CI_low": np.exp(conf[0].values),
        "CI_high": np.exp(conf[1].values),
        "p_value": logit_res.pvalues.values
    })
    or_table.to_csv(OUT_DIR / "logit_hike_vs_cut_OR.csv", index=False)

    with open(OUT_DIR / "logit_hike_vs_cut_SUMMARY.txt", "w") as f:
        f.write(logit_res.summary().as_text())

    # Average marginal effects (AME) at the overall mean
    margins = logit_res.get_margeff(at="overall").summary_frame()
    margins.to_csv(OUT_DIR / "logit_hike_vs_cut_MARGINS.csv", index=False)

    # CV AUC (pipeline standardizes X for prediction)
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, solver="liblinear"))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(pipe, Xb, yb, cv=cv, scoring="roc_auc")
    pd.Series(auc_scores, name="AUC").to_csv(OUT_DIR / "logit_hike_vs_cut_AUC_folds.csv", index=False)
    print(f"AUC (5-fold CV): mean={auc_scores.mean():.3f}, sd={auc_scores.std():.3f}")


    # (2) Multinomial Logit: cut / hold / hike  (baseline = 'hold')

    print("\n=== (2) Multinomial Logit: cut / hold / hike (baseline = 'hold') ===")
    df_mn = df[df["decision_+7d"].isin(["cut","hold","hike"])].copy()

    # Force categorical ordering so that 'hold' is the reference (baseline)
    cat = pd.Categorical(df_mn["decision_+7d"], categories=["hold","cut","hike"], ordered=True)
    y_codes = cat.codes  # 0=hold (baseline), 1=cut, 2=hike

    Xm = sm.add_constant(df_mn[X_cols].copy())
    mn_mod = sm.MNLogit(y_codes, Xm)
    mn_res = mn_mod.fit(disp=False)

    # Relative Risk Ratios (RRR = exp(coef)) for 'cut vs hold' and 'hike vs hold'
    rrr = np.exp(mn_res.params.copy())
    # Label the two equations clearly:
    if rrr.shape[0] == 2:
        rrr.index = ["cut vs hold", "hike vs hold"]
    rrr.to_csv(OUT_DIR / "mnlogit_RRR.csv")
    with open(OUT_DIR / "mnlogit_SUMMARY.txt", "w") as f:
        f.write(mn_res.summary().as_text())

    #  Console snapshots 
    print("\n--- Binary Logit (hike vs cut) — top signals (by p-value) ---")
    print(or_table.sort_values("p_value")[["feature","OR","CI_low","CI_high","p_value"]].head(10))

    print("\n--- Multinomial Logit — RRR (all predictors) ---")
    print(rrr)

    print("\nSaved files in:", OUT_DIR)
    print(" - logit_hike_vs_cut_OR.csv")
    print(" - logit_hike_vs_cut_SUMMARY.txt")
    print(" - logit_hike_vs_cut_MARGINS.csv")
    print(" - logit_hike_vs_cut_AUC_folds.csv")
    print(" - mnlogit_RRR.csv")
    print(" - mnlogit_SUMMARY.txt")

if __name__ == "__main__":
    main()