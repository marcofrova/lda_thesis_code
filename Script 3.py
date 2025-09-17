from pathlib import Path
import re
import pickle
import pandas as pd
from tqdm import tqdm
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary

#  Paths 
BASE     = Path("/Users/marcofrova/Desktop/TESI/Capitolo 3 Empirico")
INTERIM  = BASE / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

IN_PATH  = INTERIM / "speeches_en.csv"
assert IN_PATH.exists(), f"Input not found: {IN_PATH}\nRun Step 1 first."

OUT_DICT   = INTERIM / "dictionary.pkl"
OUT_CORPUS = INTERIM / "corpus.pkl"
OUT_TOKPKL = INTERIM / "tokens_big.pkl"
OUT_META   = INTERIM / "corpus_meta.csv"
OUT_LONG   = INTERIM / "tokens_long.csv"

#  Load
df = pd.read_csv(IN_PATH)
df = df.rename(columns={"contents": "text"})
# keep date as datetime for metadata
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("Total speeches before preprocessing:", len(df))

#  Cleaning
def basic_clean(s: str) -> str:
    s = s.lower()
    s = re.sub(r"https?://\S+", " ", s)         # URLs
    s = re.sub(r"\d+(\.\d+)?", " ", s)          # numbers
    s = re.sub(r"[^a-z\s-]", " ", s)            # punctuation/symbols
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["text_clean"] = df["text"].astype(str).apply(basic_clean)

#  Stopwords 
stopwords = {
    # common english
    "the","and","for","that","with","from","this","have","has","had","are","is","was","were",
    "of","to","in","on","as","by","at","be","or","an","it","its","we","our","they","their",
    # structural ECB words (uninformative for topics)
    "ecb","euro","area","european","central","bank","banks","monetary","policy","policies",
    "eurosystem","press","conference","statement","introductory","speech",
    # generic discourse fillers
    "say","think","see","make","need","more","only","even","also","today","new",
    # generic economic/time words
    "time","year","years","continue","remain","high","increase"
}

#  Lemmatization
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except Exception as e:
    raise SystemExit(
        "spaCy model 'en_core_web_sm' not found. Install it first:\n"
        "    python -m spacy download en_core_web_sm\n"
        f"Original error: {e}"
    )

tqdm.pandas()

def lemmatize_doc(s: str):
    doc = nlp(s)
    lemmas = [t.lemma_ for t in doc if t.pos_ in {"NOUN","ADJ","VERB","ADV"}]
    important_two = {"eu","us","uk","qe"}
    lemmas = [w for w in lemmas if (len(w) >= 3 or w in important_two) and w not in stopwords]
    return lemmas

df["tokens"] = df["text_clean"].progress_apply(lemmatize_doc)

#  Filter very short docs 
df = df[df["tokens"].apply(len) >= 40].copy()
print("Remaining speeches after length filter:", len(df))

#  Bigrams 
phrases = Phrases(df["tokens"], min_count=10, threshold=10.0)
bigram  = Phraser(phrases)

def apply_bigrams(tokens):
    return bigram[tokens]

df["tokens_big"] = df["tokens"].apply(apply_bigrams)

# Force key domain phrases
def force_phrases(tokens):
    joined = " ".join(tokens)
    joined = joined.replace("forward guidance", "forward_guidance")
    joined = joined.replace("asset purchase", "asset_purchase")
    joined = joined.replace("deposit facility", "deposit_facility")
    joined = joined.replace("interest rate", "interest_rate")
    return joined.split()

df["tokens_big"] = df["tokens_big"].apply(force_phrases)

#  Dictionary & corpus 
dictionary = Dictionary(df["tokens_big"])
dictionary.filter_extremes(no_below=5)  # keep words appearing in >= 5 speeches
dictionary.compactify()

corpus = [dictionary.doc2bow(toks) for toks in df["tokens_big"]]

print("Vocabulary size:", len(dictionary))

#  Save
(df[["date","speakers","title","subtitle"]]
 .assign(n_tokens=df["tokens_big"].apply(len))
 .to_csv(OUT_META, index=False))

dictionary.save(str(OUT_DICT))

with open(OUT_CORPUS, "wb") as f:
    pickle.dump(corpus, f)

df[["tokens_big"]].to_pickle(OUT_TOKPKL)

print("Saved for Python:")
print(" -", OUT_DICT.name)
print(" -", OUT_CORPUS.name)
print(" -", OUT_TOKPKL.name)
print(" -", OUT_META.name)

#  Save (R-friendly long format in case of using R)
tokens_long = (
    df[["date","speakers","title","subtitle","tokens_big"]]
    .reset_index(names="doc_id")          # numeric doc id
    .explode("tokens_big")
    .rename(columns={"tokens_big":"token"})
)

tokens_long.to_csv(OUT_LONG, index=False)
print("Saved for R:")
print(" -", OUT_LONG.name)