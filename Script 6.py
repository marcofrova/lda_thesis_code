# %% Step 6 — Build meeting-level dataset (K=30)
import pandas as pd
import numpy as np
from pathlib import Path

#  Paths 
BASE          = Path("/Users/marcofrova/Desktop/TESI/Capitolo 3 Empirico")
THETA_PATH    = BASE / "results" / "lda" / "doc_topics_K30.csv"       
EA_MPD_XLSX   = BASE / "data" / "Dataset_EA-MPD.xlsx"                 
OUT_PATH      = BASE / "results" / "lda" / "meeting_dataset_K30.csv"  # output final
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 45        
DELTA_TOL   = 1e-12     

def label_decision(delta):
    if pd.isna(delta): return "no_data"
    if delta >  DELTA_TOL: return "hike"
    if delta < -DELTA_TOL: return "cut"
    return "hold"

df = pd.read_csv(THETA_PATH)

if "meeting_date" in df.columns:
    df["date"] = pd.to_datetime(df["meeting_date"], errors="coerce").dt.normalize()
else:
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

topic_cols = [c for c in df.columns if str(c).startswith("topic_")]
assert len(topic_cols) == 30, f"Expected 30 topic columns, found {len(topic_cols)}"

ea = pd.read_excel(EA_MPD_XLSX, sheet_name="DATES")
meetings = (
    pd.to_datetime(ea["date-ordered"], errors="coerce")
      .dropna()
      .drop_duplicates()
      .to_frame(name="meeting_date")
)
meetings["meeting_date"] = meetings["meeting_date"].dt.normalize()
meetings = meetings.sort_values("meeting_date").reset_index(drop=True)

meetings_sorted = meetings.sort_values("meeting_date").reset_index(drop=True)
merged = pd.merge_asof(
    df.sort_values("date"), meetings_sorted,
    left_on="date", right_on="meeting_date",
    direction="forward", tolerance=pd.Timedelta(days=WINDOW_DAYS)
).dropna(subset=["meeting_date"])

agg = merged.groupby("meeting_date")[topic_cols].mean().reset_index()
agg["n_speeches"] = merged.groupby("meeting_date").size().values

meeting_out = agg.copy()
meeting_out["period"] = np.where(
    meeting_out["meeting_date"] < pd.Timestamp(2015,1,1),
    "1999–2014", "2015–oggi"
)

meeting_out = meeting_out.sort_values("meeting_date").reset_index(drop=True)
meeting_out.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
print("Rows:", len(meeting_out), "| date min/max:", meeting_out["meeting_date"].min(), meeting_out["meeting_date"].max())


# %% Step 6 fix
from pathlib import Path
import pandas as pd
import numpy as np

#  Paths 
BASE     = Path("/Users/marcofrova/Desktop/TESI/Capitolo 3 Empirico")
MEET_IN  = BASE / "results" / "lda" / "meeting_dataset_K30.csv"              
DFR_CSV  = BASE / "data" / "ECBDFR_clean.csv"                                
OUT_CSV  = BASE / "results" / "meetings" / "meeting_dataset_K30_FIX.csv"     
OUT_XLSX = BASE / "results" / "meetings" / "meeting_dataset_K30_FIX.xlsx"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

meet = pd.read_csv(MEET_IN, parse_dates=["meeting_date"])

dfr_raw = pd.read_csv(DFR_CSV)

if {"DATE", "VALUE"}.issubset(dfr_raw.columns):
    dfr = dfr_raw.rename(columns={"DATE": "date", "VALUE": "dfr"})
elif {"date_correct", "ECBDFR"}.issubset(dfr_raw.columns):
    dfr = dfr_raw.rename(columns={"date_correct": "date", "ECBDFR": "dfr"})
else:
    first_two = dfr_raw.columns[:2]
    dfr = dfr_raw.rename(columns={first_two[0]: "date", first_two[1]: "dfr"})

dfr["date"] = pd.to_datetime(dfr["date"], errors="coerce").dt.normalize()
dfr["dfr"]  = pd.to_numeric(dfr["dfr"], errors="coerce")
dfr = dfr.dropna(subset=["date","dfr"]).sort_values("date").reset_index(drop=True)
dfr["dfr_lag"]   = dfr["dfr"].shift(1)
dfr["delta_dfr"] = dfr["dfr"] - dfr["dfr_lag"]
dfr["delta_sign"] = np.where(dfr["delta_dfr"] > 1e-12, 1,
                      np.where(dfr["delta_dfr"] < -1e-12, -1, 0))

m2 = meet.merge(dfr[["date","dfr","delta_dfr","delta_sign"]],
                left_on="meeting_date", right_on="date", how="left").drop(columns=["date"])

def label_day(delta):
    if pd.isna(delta): return "no_data"
    if delta > 1e-12:  return "hike"
    if delta < -1e-12: return "cut"
    return "hold"

m2["decision_day"] = m2["delta_dfr"].apply(label_day)

def decision_forward7(meet_date):
    win = dfr[(dfr["date"] >= meet_date) & (dfr["date"] <= meet_date + pd.Timedelta(days=7))]
    if win.empty:
        return pd.NA
    chg = win[win["delta_sign"] != 0]
    if chg.empty:
        return "hold"
    return "hike" if chg.iloc[0]["delta_sign"] > 0 else "cut"

m2["decision_+7d"] = m2["meeting_date"].apply(decision_forward7)

if "period" not in m2.columns:
    m2["period"] = np.where(m2["meeting_date"] < pd.Timestamp(2015,1,1),
                            "1999–2014", "2015–oggi")

m2.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
m2.to_excel(OUT_XLSX, index=False)
print("Saved:", OUT_CSV)
print("Saved:", OUT_XLSX)
print("Decision counts (day): ", m2["decision_day"].value_counts(dropna=False).to_dict())
print("Decision counts (+7d): ", m2["decision_+7d"].value_counts(dropna=False).to_dict())