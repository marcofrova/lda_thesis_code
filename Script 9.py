from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#  Paths 
BASE = Path("/Users/marcofrova/Desktop/TESI/Capitolo 3 Empirico")
IN_XLSX = BASE / "results" / "meetings" / "meeting_dataset_K30_FIX_named.xlsx"
OUT_DIR = BASE / "results" / "pca"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# How many PCs to keep in the augmented dataset
N_KEEP = 11
TOP_K  = 5   # top topics to display per PC

#  Load 
assert IN_XLSX.exists(), f"Missing input file: {IN_XLSX}"
m2 = pd.read_excel(IN_XLSX, parse_dates=["meeting_date"])

# Topic columns
topic_cols = [c for c in m2.columns if c.startswith("Topic ")]
if not topic_cols:
    raise RuntimeError("No topic columns found (expected names starting with 'Topic ').")

X = m2[topic_cols].fillna(0).values

# - PCA fit -
scaler = StandardScaler()
X_std  = scaler.fit_transform(X)

pca = PCA()                         # full spectrum fit
X_pca = pca.fit_transform(X_std)

# Explained variance plot
cumvar = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8,5))
plt.plot(range(1, len(cumvar)+1), cumvar, marker="o")
plt.axhline(0.70, ls="", alpha=0.5)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA — cumulative explained variance")
plt.grid(True, linestyle="", alpha=0.6)
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_explained_variance.png", dpi=150)
plt.close()
print("Saved:", OUT_DIR / "pca_explained_variance.png")

# Loadings: topics (rows) x PCs (cols)
pc_names = [f"PC{i+1}" for i in range(len(topic_cols))]
loadings = pd.DataFrame(pca.components_.T, index=topic_cols, columns=pc_names)
loadings.to_csv(OUT_DIR / "pca_topic_loadings.csv")
print("Saved:", OUT_DIR / "pca_topic_loadings.csv")

# Augment dataset with first N_KEEP PC scores
m2_pca = m2.copy()
for i in range(min(N_KEEP, X_pca.shape[1])):
    m2_pca[f"PC{i+1}"] = X_pca[:, i]

m2_pca.to_excel(OUT_DIR / f"meeting_dataset_PCA_PC{N_KEEP}.xlsx", index=False)
print("Saved:", OUT_DIR / f"meeting_dataset_PCA_PC{N_KEEP}.xlsx")

# Top-k topics per PC (1..N_KEEP) 
# Pretty table + console print to help labeling PCs
rows = []
for i in range(min(N_KEEP, loadings.shape[1])):
    pc = f"PC{i+1}"
    s = loadings[pc]
    top_idx = s.abs().sort_values(ascending=False).head(TOP_K).index
    for rank, topic in enumerate(top_idx, start=1):
        coef = float(s.loc[topic])
        rows.append({
            "PC": pc,
            "Rank": rank,
            "Topic": topic,
            "Loading": coef,
            "AbsLoading": abs(coef),
            "Sign": "positive" if coef >= 0 else "negative",
        })

top_table = pd.DataFrame(rows).sort_values(["PC","Rank"]).reset_index(drop=True)
top_path  = OUT_DIR / f"pca_top_loadings_PC1-{min(N_KEEP, loadings.shape[1])}.csv"
top_table.to_csv(top_path, index=False)
print("Saved:", top_path)

# Console pretty print
print(f"\nTop-{TOP_K} topics by absolute loading for PC1..PC{min(N_KEEP, loadings.shape[1])}")
print("-"*72)
for pc in [f"PC{i+1}" for i in range(min(N_KEEP, loadings.shape[1]))]:
    block = top_table[top_table["PC"] == pc]
    print(f"{pc}:")
    for _, r in block.iterrows():
        label = r["Topic"].split(": ", 1)[-1] if ": " in r["Topic"] else r["Topic"]
        print(f"  #{int(r['Rank'])}  {label:35s}  Loading={r['Loading']:+.3f} ({r['Sign']})")
    print("-"*72)