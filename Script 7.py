from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  Paths 
BASE         = Path("/Users/marcofrova/Desktop/TESI/Capitolo 3 Empirico")
DATA_DIR     = BASE / "data"
RESULTS_DIR  = BASE / "results"
RESULTS_LDA  = RESULTS_DIR / "lda"
RESULTS_MTG  = RESULTS_DIR / "meetings"
RESULTS_MTG.mkdir(parents=True, exist_ok=True)

# Inputs
MEET_PATH    = RESULTS_LDA / "meeting_dataset_K30.csv"   # meeting-level topics (preferred)
EA_MPD_XLSX  = DATA_DIR / "Dataset_EA-MPD.xlsx"
DFR_PATH     = DATA_DIR / "ECBDFR_clean.csv"

assert EA_MPD_XLSX.exists(), f"Missing: {EA_MPD_XLSX}"
assert DFR_PATH.exists(),    f"Missing: {DFR_PATH}"
assert MEET_PATH.exists(),   f"Missing: {MEET_PATH} (ensure you exported meeting-level topics)"

# Outputs
OUT_CSV      = RESULTS_MTG / "meeting_dataset_K30_FIX.csv"
OUT_XLSX     = RESULTS_MTG / "meeting_dataset_K30_FIX.xlsx"
OUT_XLSX_NM  = RESULTS_MTG / "meeting_dataset_K30_FIX_named.xlsx"

#  Load meeting-level topics 
meet = pd.read_csv(MEET_PATH, parse_dates=["meeting_date"])
# allinea il formato data
meet["meeting_date"] = pd.to_datetime(meet["meeting_date"], errors="coerce").dt.normalize()

#  EA-MPD meeting dates 
ea_dates = pd.read_excel(EA_MPD_XLSX, sheet_name="DATES")
ea_dates["meeting_date"] = pd.to_datetime(ea_dates["date-ordered"], errors="coerce").dt.normalize()
meet = meet.merge(ea_dates[["meeting_date"]], on="meeting_date", how="inner")

print("Meeting after EA merge — min/max:", meet["meeting_date"].min(), meet["meeting_date"].max())
print("N meetings kept:", meet["meeting_date"].nunique())

ea_dates = (ea_dates.dropna(subset=["meeting_date"])
                     .drop_duplicates(subset=["meeting_date"])
                     .sort_values("meeting_date")
                     .reset_index(drop=True))

#  DFR daily (deposit facility rate) 
dfr = pd.read_csv(DFR_PATH, parse_dates=["date_correct"])
dfr = dfr.rename(columns={"date_correct": "date", "ECBDFR": "dfr"})
dfr["date"] = dfr["date"].dt.normalize()
dfr["dfr"]  = pd.to_numeric(dfr["dfr"], errors="coerce")
dfr = dfr.dropna(subset=["date","dfr"]).sort_values("date").reset_index(drop=True)
dfr["dfr_lag"]    = dfr["dfr"].shift(1)
dfr["delta_dfr"]  = dfr["dfr"] - dfr["dfr_lag"]
dfr["delta_sign"] = np.where(dfr["delta_dfr"] > 1e-12, 1,
                      np.where(dfr["delta_dfr"] < -1e-12, -1, 0))

#  Keep only real policy meetings 
meet = meet.merge(ea_dates[["meeting_date"]], on="meeting_date", how="inner")

#  Merge DFR on meeting day 
m2 = meet.drop(columns=["dfr","delta_dfr","decision"], errors="ignore").merge(
    dfr[["date","dfr","delta_dfr"]],
    left_on="meeting_date", right_on="date", how="left"
).drop(columns=["date"])

#  Labels: decision on the day 
def label_day(delta):
    if pd.isna(delta): return "no_data"
    if delta > 1e-12:  return "hike"
    if delta < -1e-12: return "cut"
    return "hold"

m2["decision_day"] = m2["delta_dfr"].apply(label_day)

#  Decision within +7 days 
def decision_forward7(meet_date):
    win = dfr[(dfr["date"] >= meet_date) & (dfr["date"] <= meet_date + pd.Timedelta(days=7))]
    if win.empty:
        return pd.NA
    chg = win[win["delta_sign"] != 0]
    if chg.empty:
        return "hold"
    return "hike" if chg.iloc[0]["delta_sign"] > 0 else "cut"

m2["decision_+7d"] = m2["meeting_date"].apply(decision_forward7)

#  Period split 
m2["period"] = np.where(
    m2["meeting_date"] < pd.Timestamp(2015, 1, 1),
    "1999–2014",   # en dash U+2013
    "2015–oggi"
)

#  Save clean (unlabeled topics) 
m2.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
m2.to_excel(OUT_XLSX, index=False)
print("Saved:", OUT_CSV)
print("Saved:", OUT_XLSX)

print("Decision counts (day): ", m2["decision_day"].value_counts(dropna=False).to_dict())
print("Decision counts (+7d): ", m2["decision_+7d"].value_counts(dropna=False).to_dict())

#  Rename topic columns with human-readable labels 
labels = {
    1: "Climate and Risk",
    2: "Credit and Lending",
    3: "Cash and Banknotes",
    4: "European Integration and Convergence",
    5: "Fiscal Policy",
    6: "Banking Union and Supervision",
    7: "Currency and Exchange Rates",
    8: "Digital Money and Fintech",
    9: "Payment Systems and Infrastructure",
    10: "European Integration and Institutions",
    11: "Inflation and Prices",
    12: "Financial Stability and Risk",
    13: "Communication and Rhetoric",
    14: "ECB Strategy and Price Stability",
    15: "Fiscal and Sovereign Debt Policy",
    16: "Economic Research & Modelling",
    17: "Economic Outlook & Global Environment",
    18: "Household Consumption and Savings",
    19: "Crisis Management & Recovery",
    20: "Structural Reforms & Competitiveness",
    21: "Prudential Supervision and Regulation",
    22: "Formal and Protocol Discourse",
    23: "Monetary Policy Operations & Liquidity",
    24: "Economic Outlook",
    25: "Unconventional Monetary Policy (QE & Forward Guidance)",
    26: "Statistics and Data",
    27: "Generic Discourse / Residual",
    28: "ECB Governance and Independence",
    29: "Pandemic and PEPP",
    30: "Financial Stability and Macroprudential Policy",
}

topic_cols = [c for c in m2.columns if c.startswith("topic_")]

def topic_id_from_col(col: str) -> int:
    # es: "topic_0" -> 0
    try:
        return int(col.split("_", 1)[1])
    except Exception:
        return None

rename_map = {}
missing_labels = []
for col in topic_cols:
    tid = topic_id_from_col(col)  # 0-based id dal file
    if tid is None:
        continue
    human_id = tid + 1            # passa a 1..30
    if human_id in labels:
        rename_map[col] = f"Topic {human_id}: {labels[human_id]}"
    else:
        missing_labels.append(human_id)

if missing_labels:
    print("[WARN] Mancano etichette per i topic:", sorted(set(missing_labels)))

m2 = m2.rename(columns=rename_map)

m2.to_excel(OUT_XLSX_NM, index=False)
print("Saved with renamed topic labels:", OUT_XLSX_NM)

#  Exploratory plots (saved to results/meetings) 

# Helper: find first topic column by keyword
def find_topic_col(keyword):
    ks = keyword.lower()
    matches = [c for c in m2.columns if c.startswith("Topic ") and ks in c.lower()]
    return matches[0] if matches else None

# Select 4 policy-relevant topics
topics_wanted = {
    "Inflation and Prices":                find_topic_col("Inflation and Prices"),
    "Unconventional Monetary Policy (QE & Forward Guidance)": find_topic_col("Unconventional Monetary Policy"),
    "Economic Outlook & Global Environment": find_topic_col("Economic Outlook & Global Environment"),
    "Fiscal and Sovereign Debt Policy":    find_topic_col("Fiscal and Sovereign Debt Policy"),
}

dec_col = "decision_+7d"
order   = ["cut", "hold", "hike"]

# Boxplots per topic
for label, col in topics_wanted.items():
    if col is None:
        continue
    data = [m2.loc[m2[dec_col]==cat, col].dropna().values for cat in order]
    plt.figure(figsize=(7,5))
    plt.boxplot(data, labels=order, showmeans=True)
    plt.title(f"{label} by policy decision (+7d)")
    plt.xlabel("Policy decision (+7d)")
    plt.ylabel("Topic share")
    plt.tight_layout()
    plt.savefig(RESULTS_MTG / f"boxplot_{label.replace(' ','_')}.png", dpi=150)
    plt.close()
    print(f"Saved boxplot for {label}")

# Time series for Inflation and Prices with decision markers
focal_col = topics_wanted.get("Inflation and Prices")
if focal_col:
    plt.figure(figsize=(10,5))
    plt.plot(m2["meeting_date"], m2[focal_col], label="Inflation and Prices", color="steelblue")
    for d in m2.loc[m2[dec_col]=="hike", "meeting_date"]:
        plt.axvline(d, linestyle="--", color="green", alpha=0.4)
    for d in m2.loc[m2[dec_col]=="cut", "meeting_date"]:
        plt.axvline(d, linestyle=":", color="red", alpha=0.6)
    plt.title("Inflation and Prices topic share over time with ECB decisions")
    plt.xlabel("Meeting date")
    plt.ylabel("Topic share")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_MTG / "timeseries_Inflation_Prices.png", dpi=150)
    plt.close()
    print("Saved: timeseries_Inflation_Prices.png")

# Focus set
focus = {k:v for k,v in topics_wanted.items() if v is not None}
print("Selected focus topics:", focus)

# Sort by date
m2 = m2.sort_values("meeting_date").reset_index(drop=True)
m2["year"] = m2["meeting_date"].dt.year

# Yearly averages (trends)
yr = (m2[["year"] + list(focus.values())]
      .groupby("year", as_index=False)
      .mean())

plt.figure(figsize=(10,5))
for label, col in focus.items():
    plt.plot(yr["year"], yr[col], marker="o", label=label)
plt.title("Yearly average topic share")
plt.xlabel("Year")
plt.ylabel("Average share")
plt.tight_layout()
plt.legend()
plt.savefig(RESULTS_MTG / "topic_trends_yearly.png", dpi=150)
plt.close()
print("Saved: topic_trends_yearly.png")

# Rolling average (4 meetings)
m2_rolling = m2.copy()
for label, col in focus.items():
    m2_rolling[f"roll_{label}"] = m2_rolling[col].rolling(window=4, min_periods=1).mean()

plt.figure(figsize=(10,5))
for label, col in focus.items():
    plt.plot(m2_rolling["meeting_date"], m2_rolling[f"roll_{label}"], label=label)
plt.title("Rolling (4 meetings) average topic share")
plt.xlabel("Meeting date")
plt.ylabel("Rolling average share")
plt.tight_layout()
plt.legend()
plt.savefig(RESULTS_MTG / "topic_trends_rolling.png", dpi=150)
plt.close()
print("Saved: topic_trends_rolling.png")

# Period comparison bars
per_mean = (m2.groupby("period")[list(focus.values())]
              .mean()
              .T.reset_index()
              .rename(columns={"index":"Topic"}))
per_mean["Topic"] = per_mean["Topic"].map({v:k for k,v in focus.items()})

plt.figure(figsize=(9,5))
x = np.arange(len(per_mean))
w = 0.35
plt.bar(x - w/2, per_mean["1999–2014"], width=w, label="1999–2014")
plt.bar(x + w/2, per_mean["2015–oggi"], width=w, label="2015–today")
plt.xticks(x, per_mean["Topic"], rotation=15, ha="right")
plt.ylabel("Mean share")
plt.title("Topic share by period")
plt.tight_layout()
plt.legend()
plt.savefig(RESULTS_MTG / "topic_period_comparison.png", dpi=150)
plt.close()
print("Saved: topic_period_comparison.png")

# Topic–topic correlation (heatmap)
topic_cols_all = [c for c in m2.columns if c.startswith("Topic ")]
corr = m2[topic_cols_all].corr()

plt.figure(figsize=(10,8))
im = plt.imshow(corr, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04, label="Correlation")
plt.title("Correlation between topics (meeting-level shares)")
tick_every = max(1, len(topic_cols_all)//20)
plt.xticks(range(0,len(topic_cols_all),tick_every),
           [c for i,c in enumerate(topic_cols_all) if i%tick_every==0],
           rotation=90)
plt.yticks(range(0,len(topic_cols_all),tick_every),
           [c for i,c in enumerate(topic_cols_all) if i%tick_every==0])
plt.tight_layout()
plt.savefig(RESULTS_MTG / "topic_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: topic_correlation_heatmap.png")

# Topic entropy (over all topics)
eps = 1e-12
P = m2[topic_cols_all].clip(lower=0) + eps
row_sum = P.sum(axis=1).values.reshape(-1,1)
P = P.values / row_sum
entropy = -(P * np.log(P)).sum(axis=1)
m2["topic_entropy"] = entropy

plt.figure(figsize=(10,4))
plt.plot(m2["meeting_date"], m2["topic_entropy"])
plt.title("Topic entropy over time (all topics)")
plt.xlabel("Meeting date")
plt.ylabel("Entropy")
plt.tight_layout()
plt.savefig(RESULTS_MTG / "topic_entropy_timeseries.png", dpi=150)
plt.close()
print("Saved: topic_entropy_timeseries.png")

plt.figure(figsize=(6,5))
data = [m2.loc[m2["period"]=="1999–2014", "topic_entropy"],
        m2.loc[m2["period"]=="2015–oggi", "topic_entropy"]]
plt.boxplot(data, labels=["1999–2014", "2015–today"], showmeans=True)
plt.title("Topic entropy by period")
plt.ylabel("Entropy")
plt.tight_layout()
plt.savefig(RESULTS_MTG / "topic_entropy_by_period.png", dpi=150)
plt.close()
print("Saved: topic_entropy_by_period.png")

# Emerging topics (optional block kept as-is)
def find_topic_col2(keyword: str):
    ks = keyword.lower()
    m = [c for c in m2.columns if c.startswith("Topic ") and ks in c.lower()]
    return m[0] if m else None

# Emerging topics
emerging = {
    "Climate and Risk":              find_topic_col2("Climate and Risk"),
    "Digital Money and Fintech":     find_topic_col2("Digital Money and Fintech"),
    "Banking Union and Supervision": find_topic_col2("Banking Union and Supervision"),
    "Pandemic and PEPP":             find_topic_col2("Pandemic and PEPP"),
}

emerging = {k: v for k, v in emerging.items() if v is not None}
print("Emerging topics found:", emerging)

if emerging:
    m2 = m2.sort_values("meeting_date").reset_index(drop=True)
    m2["year"] = m2["meeting_date"].dt.year

    yr = (m2[["year"] + list(emerging.values())]
          .groupby("year", as_index=False)
          .mean())
    plt.figure(figsize=(10,5))
    for label, col in emerging.items():
        plt.plot(yr["year"], yr[col], marker="o", label=label)
    plt.title("Yearly average topic share — emerging themes")
    plt.xlabel("Year")
    plt.ylabel("Average share")
    plt.tight_layout()
    plt.legend()
    plt.savefig(RESULTS_MTG / "emerging_topic_trends_yearly.png", dpi=150)
    plt.close()
    print("Saved: emerging_topic_trends_yearly.png")

    m2_roll = m2.copy()
    for label, col in emerging.items():
        m2_roll[f"roll_{label}"] = m2_roll[col].rolling(window=4, min_periods=1).mean()

    plt.figure(figsize=(10,5))
    for label, col in emerging.items():
        plt.plot(m2_roll["meeting_date"], m2_roll[f"roll_{label}"], label=label)
    plt.title("Rolling (4 meetings) average — emerging themes")
    plt.xlabel("Meeting date")
    plt.ylabel("Rolling average share")
    plt.tight_layout()
    plt.legend()
    plt.savefig(RESULTS_MTG / "emerging_topic_trends_rolling.png", dpi=150)
    plt.close()
    print("Saved: emerging_topic_trends_rolling.png")

    per_mean = (m2.groupby("period")[list(emerging.values())]
                  .mean()
                  .T.reset_index()
                  .rename(columns={"index":"Column"}))
    col2lab = {v: k for k, v in emerging.items()}
    per_mean["Topic"] = per_mean["Column"].map(col2lab)
    per_mean = per_mean.sort_values("2015–oggi", ascending=False)

    x = np.arange(len(per_mean)); w = 0.38
    plt.figure(figsize=(10,6))
    plt.bar(x - w/2, per_mean["1999–2014"], width=w, label="1999–2014")
    plt.bar(x + w/2, per_mean["2015–oggi"], width=w, label="2015–today")
    plt.xticks(x, per_mean["Topic"], rotation=15, ha="right")
    plt.ylabel("Mean share")
    plt.title("Emerging topics — average share by period")
    plt.tight_layout()
    plt.legend()
    plt.savefig(RESULTS_MTG / "emerging_topic_period_comparison.png", dpi=150)
    plt.close()
    print("Saved: emerging_topic_period_comparison.png")
    
    
