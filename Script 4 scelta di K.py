
from pathlib import Path
import time, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import tomotopy as tp
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

#  Paths 
BASE        = Path("/Users/marcofrova/Desktop/TESI/Capitolo 3 Empirico")
INTERIM     = BASE / "interim"
OUTDIR      = BASE / "results" / "lda_alpha"
OUTDIR.mkdir(parents=True, exist_ok=True)

TOKENS_PATH = INTERIM / "tokens_big.pkl"
META_PATH   = INTERIM / "corpus_meta.csv"

assert TOKENS_PATH.exists(), f"Missing file: {TOKENS_PATH}"
assert META_PATH.exists(),   f"Missing file: {META_PATH}"

#  Grid 
K_VALUES    = [20, 25, 30]
# alpha set includes fixed values and K/50 (as requested)
def alpha_grid_for_K(K: int):
    return [0.1, 0.3, 0.5, K/50.0]

ITERATIONS  = 2000          # CGS iterations (adjust to 3000 for the final run)
RANDOM_SEED = 42
TOPN_WORDS  = 50            # for c_v coherence

#  Load 
texts = pd.read_pickle(TOKENS_PATH)["tokens_big"]   # Series of lists
meta  = pd.read_csv(META_PATH)

# Prepare inputs for gensim coherence (plain lists + dictionary)
cm_texts_full = [list(map(str, doc)) for doc in texts.tolist()]  # list[list[str]]
id2word_cm    = Dictionary(cm_texts_full)

print(f"Docs: {len(texts)} | Example doc tokens: {texts.iloc[0][:12]}")

#  Run grid 
rows = []

for K in tqdm(K_VALUES, desc="Grid over K"):
    alphas = alpha_grid_for_K(K)
    for alpha in tqdm(alphas, desc=f"K={K} | alpha loop", leave=False):
        t0 = time.perf_counter()

        # Build model with explicit symmetric alpha; keep eta default (tomotopy ~0.01)
        mdl = tp.LDAModel(k=K, alpha=alpha, seed=RANDOM_SEED)

        # Add documents
        for doc in cm_texts_full:
            mdl.add_doc(doc)

        # Train (CGS)
        for i in range(ITERATIONS):
            mdl.train(1)
            if (i+1) % 200 == 0:
                pass  # keep console quiet; add prints if you want

        # Compute coherence c_v
        topic_words = [[str(w) for (w, _p) in mdl.get_topic_words(k, top_n=TOPN_WORDS)]
                       for k in range(K)]
        cm = CoherenceModel(
            topics=topic_words,
            texts=cm_texts_full,
            dictionary=id2word_cm,
            coherence='c_v',
            processes=1
        )
        c_v = float(cm.get_coherence())

        # Training-set perplexity from ll_per_word
        llpw = float(mdl.ll_per_word)     # log-likelihood per word
        perp = float(np.exp(-llpw))       # perplexity = exp(-LL per word)

        t1 = time.perf_counter()

        rows.append({
            "K": K,
            "alpha": alpha,
            "iterations": ITERATIONS,
            "coherence_c_v": c_v,
            "ll_per_word": llpw,
            "perplexity": perp,
            "train_seconds": t1 - t0
        })

        # save a tiny preview of top words per topic for quick inspection
        preview_path = OUTDIR / f"topics_PREVIEW_K{K}_alpha{alpha:.3f}.csv"
        pd.DataFrame({
            "topic": list(range(K)),
            "words": [", ".join([w for w, _ in mdl.get_topic_words(t, top_n=12)]) for t in range(K)]
        }).to_csv(preview_path, index=False)

        # Free coherence objects
        del cm, topic_words
        gc.collect()

#  Save table 
df = pd.DataFrame(rows).sort_values(["K", "alpha"]).reset_index(drop=True)
out_csv = OUTDIR / "alpha_sensitivity_results.csv"
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)

#  Plots 
# 1) Coherence vs alpha (one line per K)
plt.figure(figsize=(8,5))
for K in K_VALUES:
    sub = df[df["K"] == K]
    plt.plot(sub["alpha"], sub["coherence_c_v"], marker="o", label=f"K={K}")
plt.xlabel("alpha")
plt.ylabel("Coherence (c_v)")
plt.title("Coherence c_v vs alpha (tomotopy CGS)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "coherence_by_alpha.png", dpi=150)
plt.close()
print("Saved:", OUTDIR / "coherence_by_alpha.png")

# 2) Perplexity vs alpha (lower is better)
plt.figure(figsize=(8,5))
for K in K_VALUES:
    sub = df[df["K"] == K]
    plt.plot(sub["alpha"], sub["perplexity"], marker="o", label=f"K={K}")
plt.xlabel("alpha")
plt.ylabel("Perplexity (training)")
plt.title("Perplexity vs alpha (tomotopy CGS)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "perplexity_by_alpha.png", dpi=150)
plt.close()
print("Saved:", OUTDIR / "perplexity_by_alpha.png")

# 3) Quick console summary: best c_v per K
print("\nBest coherence per K:")
best = (df.loc[df.groupby("K")["coherence_c_v"].idxmax(), ["K","alpha","coherence_c_v","perplexity"]]
          .sort_values("K"))
print(best.to_string(index=False))