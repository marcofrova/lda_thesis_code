
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Paths 
BASE     = Path("/Users/marcofrova/Desktop/TESI/Capitolo 3 Empirico")
INTERIM  = BASE / "interim"
RESULTS  = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

IN_PATH  = INTERIM / "speeches_en.csv"
assert IN_PATH.exists(), f"Input not found: {IN_PATH}\nRun Step 1 first."

#  Load 
df = pd.read_csv(IN_PATH)
print("Total speeches (English only):", len(df))

# Parse dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).copy()
df["year"] = df["date"].dt.year

#  EDA 

# 1) Count per year
speeches_per_year = df.groupby("year", dropna=True).size()

plt.figure(figsize=(10,5))
speeches_per_year.plot(kind="bar", color="steelblue")
plt.title("Number of ECB speeches per year (English only)")
plt.ylabel("Count")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig(RESULTS / "eda_speeches_per_year.png", dpi=150)
plt.close()
print("Saved:", RESULTS / "eda_speeches_per_year.png")

# 2) Top speakers
top_speakers = (df["speakers"]
                .fillna("Unknown")
                .value_counts()
                .head(10))

plt.figure(figsize=(9,5))
sns.barplot(y=top_speakers.index, x=top_speakers.values, color="darkorange")
plt.title("Top 10 speakers by number of speeches (English only)")
plt.xlabel("Number of speeches")
plt.ylabel("Speaker")
plt.tight_layout()
plt.savefig(RESULTS / "eda_top_speakers.png", dpi=150)
plt.close()
print("Saved:", RESULTS / "eda_top_speakers.png")

# 3) Length stats & histogram
df["n_chars"] = df["contents"].astype(str).str.len()
df["n_words"] = df["contents"].astype(str).str.split().str.len()

print("Average length (characters):", round(df["n_chars"].mean(), 1))
print("Average length (words):     ", round(df["n_words"].mean(), 1))

plt.figure(figsize=(9,5))
sns.histplot(df["n_words"], bins=60, kde=False, color="green")
plt.xlim(0, 20000)  # adjust if needed
plt.title("Distribution of speech lengths (words)")
plt.xlabel("Number of words")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(RESULTS / "eda_length_hist_words.png", dpi=150)
plt.close()
print("Saved:", RESULTS / "eda_length_hist_words.png")