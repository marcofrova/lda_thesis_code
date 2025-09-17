from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#  Paths 
BASE = Path("/Users/marcofrova/Desktop/TESI/Capitolo 3 Empirico")
DATA_XLSX = BASE / "results" / "meetings" / "meeting_dataset_K30_FIX_named.xlsx"

# Output folder (all results go here)
OUTDIR = BASE / "results" / "regressions_pca_hold"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Choose how many PCs (≈70% variance at ~11 PCs)
N_PC = 11

def main():
    #  Load dataset 
    assert DATA_XLSX.exists(), f"Missing input file: {DATA_XLSX}"
    df = pd.read_excel(DATA_XLSX, parse_dates=["meeting_date"])
    assert "decision_+7d" in df.columns and "period" in df.columns

    #  PCA on topics 
    topic_cols = [c for c in df.columns if c.startswith("Topic ")]
    if not topic_cols:
        raise RuntimeError("No topic columns found (expected names starting with 'Topic ').")
    X = df[topic_cols].fillna(0).values

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=N_PC)
    X_pca = pca.fit_transform(X_std)

    # (Optional) explained variance curve (full)
    pca_full = PCA().fit(X_std)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(cumvar)+1), cumvar, marker="o")
    plt.axhline(0.70, ls="--", alpha=0.5)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA — cumulative explained variance")
    plt.tight_layout()
    plt.savefig(OUTDIR / "pca_explained_variance.png", dpi=150)
    plt.close()

    # Attach PC scores
    for i in range(N_PC):
        df[f"PC{i+1}"] = X_pca[:, i]

    # Period dummy
    df["post2015"] = (df["period"] == "2015–oggi").astype(int)

    # Save dataset with PCs
    out_pca = OUTDIR / f"meeting_dataset_with_PC{N_PC}.xlsx"
    df.to_excel(out_pca, index=False)
    print("Saved:", out_pca)

    #  Regressions 
    X_cols = [f"PC{i+1}" for i in range(N_PC)] + ["post2015"]

    #  (1) Binary Logit: hike (1) vs cut (0) 
    df_bin = df[df["decision_+7d"].isin(["hike","cut"])].copy()
    df_bin["y_hike"] = (df_bin["decision_+7d"] == "hike").astype(int)

    Xb = sm.add_constant(df_bin[X_cols])
    yb = df_bin["y_hike"]
    logit_res = sm.Logit(yb, Xb).fit(disp=False)

    # Odds Ratios with 95% CI
    conf = logit_res.conf_int()
    or_table = pd.DataFrame({
        "feature": logit_res.params.index,
        "OR": np.exp(logit_res.params.values),
        "CI_low": np.exp(conf[0].values),
        "CI_high": np.exp(conf[1].values),
        "p_value": logit_res.pvalues.values
    })
    or_table.to_csv(OUTDIR / "pca_logit_hike_vs_cut_OR.csv", index=False)
    with open(OUTDIR / "pca_logit_hike_vs_cut_SUMMARY.txt","w") as f:
        f.write(logit_res.summary().as_text())

    # Cross-validated AUC (sklearn logistic inside a pipeline)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, solver="liblinear"))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(pipe, df_bin[X_cols], yb, cv=cv, scoring="roc_auc")
    pd.Series(auc_scores, name="AUC").to_csv(OUTDIR / "pca_logit_hike_vs_cut_AUC_folds.csv", index=False)
    print(f"AUC (5-fold CV, PCs): mean={auc_scores.mean():.3f}, sd={auc_scores.std():.3f}")

    #  (2) Multinomial Logit: cut / hold / hike  [BASELINE = 'hold'] 
    df_mn = df[df["decision_+7d"].isin(["cut","hold","hike"])].copy()
    Xm = sm.add_constant(df_mn[X_cols])

    # Explicit coding to force baseline = hold
    # hold=0 (baseline), cut=1, hike=2
    code_map = {"hold": 0, "cut": 1, "hike": 2}
    y_codes = df_mn["decision_+7d"].map(code_map).astype(int)
    print("Unique y codes (expect [0 1 2]):", np.unique(y_codes))

    mn_res = sm.MNLogit(y_codes, Xm).fit(disp=False)

    # Relative Risk Ratios (RRR = exp(coef)) vs baseline ('hold')
    rrr_raw = np.exp(mn_res.params)

    # Ensure equations are rows: (cut vs hold, hike vs hold)
    n_features = 1 + len(X_cols)  # const + predictors
    if rrr_raw.shape[0] == n_features:   # rows are features -> transpose
        rrr = rrr_raw.T.copy()
    else:
        rrr = rrr_raw.copy()

    # Label rows explicitly and columns as features
    if rrr.shape[0] == 2:
        rrr.index = ["cut vs hold", "hike vs hold"]
    else:
        rrr.index = [f"eq_{i}" for i in range(rrr.shape[0])]
    rrr.columns = ["const"] + X_cols

    # Save MNLogit outputs
    rrr.to_csv(OUTDIR / "pca_mnlogit_RRR.csv")
    with open(OUTDIR / "pca_mnlogit_SUMMARY.txt","w") as f:
        f.write(mn_res.summary().as_text())
    (OUTDIR / "mnlogit_baseline_mapping.txt").write_text(
        "MNLogit explicit coding:\n"
        "hold=0 (baseline), cut=1, hike=2\n"
        "Rows in RRR: [cut vs hold, hike vs hold]\n"
    )

    print("\nSaved to:", OUTDIR)
    print(" - pca_explained_variance.png")
    print(" - meeting_dataset_with_PC11.xlsx")
    print(" - pca_logit_hike_vs_cut_OR.csv")
    print(" - pca_logit_hike_vs_cut_SUMMARY.txt")
    print(" - pca_logit_hike_vs_cut_AUC_folds.csv")
    print(" - pca_mnlogit_RRR.csv")
    print(" - pca_mnlogit_SUMMARY.txt")
    print(" - mnlogit_baseline_mapping.txt")

if __name__ == "__main__":
    main()