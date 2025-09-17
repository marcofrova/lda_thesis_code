from pathlib import Path
import time, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import tomotopy as tp
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

#  Paths & folders 
BASE        = Path("/Users/marcofrova/Desktop/TESI/Capitolo 3 Empirico")
INTERIM     = BASE / "interim"
RESULTS_LDA = (BASE / "results" / "lda")
RESULTS_LDA.mkdir(parents=True, exist_ok=True)

TOKENS_PATH = INTERIM / "tokens_big.pkl"   # pandas pickle with column 'tokens_big'
META_PATH   = INTERIM / "corpus_meta.csv"

assert TOKENS_PATH.exists(), f"Missing file: {TOKENS_PATH}"
assert META_PATH.exists(),   f"Missing file: {META_PATH}"

#  Grid & hyperparams 
K_VALUES    = [15, 20, 25, 30, 35]   # center around circa 25
ITERATIONS  = 3000                   # CGS iterations
RANDOM_SEED = 42
TOPN_WORDS  = 50                     # words used for c_v

#  Load artifacts 
texts = pd.read_pickle(TOKENS_PATH)["tokens_big"]  # Series of lists
meta  = pd.read_csv(META_PATH)
print(f"Docs: {len(texts)} | Example doc tokens: {texts.iloc[0][:12]}")

# prepare texts/dictionary for coherence (pure python lists)
cm_texts_full = [list(map(str, doc)) for doc in texts.tolist()]  # list[list[str]]
id2word_cm    = Dictionary(cm_texts_full)

#  Full run — grid over K 
coh_rows = []
models   = {}

for K in tqdm(K_VALUES, desc="Grid over K (full run)"):
    t0 = time.perf_counter()
    print(f"\n=== K={K} | START ===")

    mdl = tp.LDAModel(k=K, seed=RANDOM_SEED, alpha=0.1, eta=0.01)

    # add documents
    for doc in cm_texts_full:
        mdl.add_doc(doc)

    # training
    for i in tqdm(range(ITERATIONS), desc=f"Training K={K}", leave=False):
        mdl.train(1)
        if (i + 1) % 100 == 0:
            print(f"K={K}, Iter={i+1}, LL/word={mdl.ll_per_word:.4f}")

    t1 = time.perf_counter()
    print(f"K={K} | Training done in {t1 - t0:.1f}s → computing coherence...")

    # coherence (single-process for IDE stability)
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
    t2 = time.perf_counter()
    print(f"K={K} | c_v={c_v:.4f} | Coherence time {t2 - t1:.1f}s")

    # accumulate & save incrementally
    coh_rows.append({"K": K, "coherence": c_v})
    models[K] = mdl
    pd.DataFrame(coh_rows).sort_values("K").to_csv(RESULTS_LDA / "lda_results.csv", index=False)
    pd.DataFrame({
        "topic": list(range(K)),
        "words": [", ".join([w for w, _ in mdl.get_topic_words(t, top_n=12)]) for t in range(K)]
    }).to_csv(RESULTS_LDA / f"topics_K{K}_PREVIEW.csv", index=False)

    del cm, topic_words
    gc.collect()
    print(f"=== K={K} | DONE (total {t2 - t0:.1f}s) ===")

#  Save table & plot coherence 
df_results = pd.DataFrame(coh_rows).sort_values("K")
df_results.to_csv(RESULTS_LDA / "lda_results.csv", index=False)
print("Saved:", RESULTS_LDA / "lda_results.csv")

plt.figure(figsize=(7,4))
plt.plot(df_results["K"], df_results["coherence"], marker="o")
plt.title("c_v Coherence vs K (tomotopy CGS) — FULL")
plt.xlabel("K (number of topics)")
plt.ylabel("Coherence (c_v)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(RESULTS_LDA / "coherence_vs_K.png", dpi=150)
plt.close()
print("Saved:", RESULTS_LDA / "coherence_vs_K.png")

#  Pick best K and export final artifacts 
best_row = df_results.loc[df_results["coherence"].idxmax()]
best_K = int(best_row["K"]); best_c = float(best_row["coherence"])
print(f"\nBest K by coherence: {best_K} (c_v={best_c:.4f})")

mdl = models[best_K]

# topics (top-12 words)
pd.DataFrame({
    "topic": list(range(best_K)),
    "words": [", ".join([w for w, _ in mdl.get_topic_words(t, top_n=12)]) for t in range(best_K)]
}).to_csv(RESULTS_LDA / f"topics_K{best_K}.csv", index=False)
print("Saved:", RESULTS_LDA / f"topics_K{best_K}.csv")

# θ (doc-topic) dense
theta = np.vstack([mdl.docs[d].get_topic_dist() for d in range(len(mdl.docs))])
df_theta = pd.DataFrame(theta, columns=[f"topic_{k}" for k in range(best_K)])
df_theta = pd.concat([meta.reset_index(drop=True), df_theta], axis=1)
df_theta.to_csv(RESULTS_LDA / f"doc_topics_K{best_K}.csv", index=False)
print("Saved:", RESULTS_LDA / f"doc_topics_K{best_K}.csv")

#  Perplexity (training set) + plot 
perp_rows = []
for K, mdl in models.items():
    llpw = float(mdl.ll_per_word)        # log-likelihood per word
    perp = float(np.exp(-llpw))          # perplexity = exp(-LL per word)
    perp_rows.append({"K": int(K), "ll_per_word": llpw, "perplexity": perp})

df_perp = pd.DataFrame(perp_rows).sort_values("K")
df_perp.to_csv(RESULTS_LDA / "lda_perplexity_train.csv", index=False)
print("Saved:", RESULTS_LDA / "lda_perplexity_train.csv")

plt.figure(figsize=(7,4))
plt.plot(df_perp["K"], df_perp["perplexity"], marker="o")
plt.title("Training Perplexity vs K (tomotopy CGS)")
plt.xlabel("K (number of topics)")
plt.ylabel("Perplexity (lower is better)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(RESULTS_LDA / "perplexity_vs_K.png", dpi=150)
plt.close()
print("Saved:", RESULTS_LDA / "perplexity_vs_K.png")

print("\nDone. Check your outputs in:", RESULTS_LDA)

#  (Optional) Save artifacts for all K 
for K, mdl in models.items():
    # topics
    pd.DataFrame({
        "topic": list(range(K)),
        "words": [", ".join([w for w,_ in mdl.get_topic_words(t, top_n=12)]) for t in range(K)]
    }).to_csv(RESULTS_LDA / f"topics_K{K}.csv", index=False)

    # θ
    theta = np.vstack([mdl.docs[d].get_topic_dist() for d in range(len(mdl.docs))])
    df_theta = pd.DataFrame(theta, columns=[f"topic_{k}" for k in range(K)])
    df_theta = pd.concat([meta.reset_index(drop=True), df_theta], axis=1)
    df_theta.to_csv(RESULTS_LDA / f"doc_topics_K{K}.csv", index=False)

print("Saved topics_ and doc_topics_ for all K in models.")