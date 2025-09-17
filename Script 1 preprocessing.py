from pathlib import Path
import pandas as pd
from langdetect import detect, DetectorFactory
from tqdm import tqdm

BASE = Path("/Users/marcofrova/Desktop/TESI/Capitolo 3 Empirico")
DATA_DIR = BASE / "data"
INTERIM_DIR = BASE / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

IN_PATH = DATA_DIR / "all_ECB_speeches.csv"
assert IN_PATH.exists(), f"Input file not found: {IN_PATH}"

OUT_ALL = INTERIM_DIR / "speeches_with_lang.csv"
OUT_EN  = INTERIM_DIR / "speeches_en.csv"

df = pd.read_csv(
    IN_PATH,
    sep="|",
    encoding="utf-8",
    na_values=["", " ", "nan", "NaN", "NULL", "null"]
)
print("Speeches before cleaning:", len(df))

# Cleaning
df = df.dropna(subset=["contents"]).copy()
df["contents"] = (
    df["contents"]
    .astype(str)
    .str.replace("\xa0", " ", regex=False)
    .str.strip()
)
df = df[df["contents"].str.len() > 0].copy()

print("Speeches after cleaning:", len(df))
print("NaN in contents:", df["contents"].isna().sum())
print("Literal 'nan' left:", (df["contents"].str.lower() == "nan").sum())

#  Language detection 
DetectorFactory.seed = 0  # reproducibility

def detect_lang_safe(text: str, min_chars: int = 60) -> str:
    try:
        if len(text) < min_chars:
            return "short"
        return detect(text)
    except Exception:
        return "unk"

def head_for_detection(text: str, max_chars: int = 1500) -> str:
    return text[:max_chars]

tqdm.pandas(desc="Detecting language")
df["lang"] = df["contents"].progress_apply(lambda x: detect_lang_safe(head_for_detection(x)))

print("\nLanguage distribution (top):")
print(df["lang"].value_counts().head(10))

#  Save outputs 
df_en = df[df["lang"] == "en"].copy()

df.to_csv(OUT_ALL, index=False, encoding="utf-8")
df_en.to_csv(OUT_EN,  index=False, encoding="utf-8")

print("\nSaved:")
print(" -", OUT_ALL)
print(" -", OUT_EN)