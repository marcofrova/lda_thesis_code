from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#  Paths
BASE    = Path("/Users/marcofrova/Desktop/TESI/Capitolo 3 Empirico")
LOADCSV = BASE / "results" / "pca" / "pca_topic_loadings.csv"
OUTDIR  = BASE / "results" / "pca"
OUTDIR.mkdir(parents=True, exist_ok=True)

#  Settings
N_PC         = 11     # columns PC1..PC11
TOP_N_TOPICS = 25     # heatmap trimmed to top N topics by max |loading| for readability

def order_pc_cols(cols):
    """Sort PC columns by numeric index (PC1, PC2, ...)."""
    pc_cols = [c for c in cols if c.lower().startswith("pc")]
    def pcnum(c): 
        return int(''.join([ch for ch in c if ch.isdigit()]) or 0)
    return sorted(pc_cols, key=pcnum)

def plot_heatmap(mat, row_labels, col_labels, title, outpath):
    plt.figure(figsize=(1.2*len(col_labels)+2, 0.35*len(row_labels)+2))
    im = plt.imshow(mat, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Loading (signed)")
    plt.xticks(range(len(col_labels)), col_labels, rotation=45, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    print("Saved:", outpath)

def main():
    assert LOADCSV.exists(), f"Missing loadings file: {LOADCSV}"
    load = pd.read_csv(LOADCSV, index_col=0)

    # keep PC1..PCN
    pc_cols = order_pc_cols(load.columns)[:N_PC]
    if not pc_cols:
        raise RuntimeError("No PC columns found in loadings CSV.")
    M = load[pc_cols].copy()

    # Full heatmap (may be tall) 
    plot_heatmap(
        M.values,
        row_labels=list(M.index),
        col_labels=pc_cols,
        title=f"PCA loadings (topics × PCs 1–{N_PC}) — full",
        outpath=OUTDIR / "pca_loadings_heatmap_full.png"
    )

    # Top-N topics by max |loading| across PCs 
    max_abs = M.abs().max(axis=1).sort_values(ascending=False)
    top_idx = list(max_abs.head(TOP_N_TOPICS).index)
    M_top   = M.loc[top_idx]

    # Order rows by their dominant PC (optional nicer structure)
    # sort by (argmax PC, then magnitude)
    argmax_pc = M_top.abs().values.argmax(axis=1)
    order = np.lexsort((-M_top.abs().values.max(axis=1), argmax_pc))
    M_top_ord = M_top.iloc[order]

    plot_heatmap(
        M_top_ord.values,
        row_labels=list(M_top_ord.index),
        col_labels=pc_cols,
        title=f"PCA loadings (topics × PCs 1–{N_PC}) — top {TOP_N_TOPICS} topics",
        outpath=OUTDIR / f"pca_loadings_heatmap_top{TOP_N_TOPICS}.png"
    )

if __name__ == "__main__":
    main()