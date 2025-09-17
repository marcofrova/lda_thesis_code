from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

#  Paths 
BASE   = Path("/Users/marcofrova/Desktop/TESI/Capitolo 3 Empirico")
IN_XLS = BASE / "results" / "regressions_pca_hold" / "meeting_dataset_with_PC11.xlsx"
OUTDIR = BASE / "results" / "visuals" / "pc_timeseries_named"
OUTDIR.mkdir(parents=True, exist_ok=True)

DEC_COL = "decision_+7d"

#  PC labels
PC_NAME = {
    "PC2":  "Inflation & Prices + | Climate & Risk + | Liquidity & Stability −",
    "PC3":  "Fiscal Policy + | Sovereign Debt + | Financial Stability −",
    "PC4":  "Pandemic & PEPP + | Credit & Lending − | Banking Union −",
    "PC6":  "Interest Rates + | Monetary Policy Ops − | Structural Reforms −",
    "PC7":  "Global Outlook + | Trade & Integration − | Household Consumption −",
}

def main():
    df = pd.read_excel(IN_XLS, parse_dates=["meeting_date"])
    df = df.sort_values("meeting_date").reset_index(drop=True)

    # date lists per decision (per le linee verticali)
    hike_dates = df.loc[df[DEC_COL] == "hike", "meeting_date"].tolist()
    cut_dates  = df.loc[df[DEC_COL] == "cut",  "meeting_date"].tolist()

    for pc_col, pretty_name in PC_NAME.items():
        if pc_col not in df.columns:
            print(f"[WARN] {pc_col} not found in {IN_XLS.name}")
            continue

        plt.figure(figsize=(11, 5))

        # Serie PC
        plt.plot(df["meeting_date"], df[pc_col],
                 label=f"{pc_col} — {pretty_name}",
                 color="steelblue", linewidth=1.6)

        # Overlay: hikes (verde tratteggiato) e cuts (rosso puntinato)
        first_h, first_c = True, True
        for d in hike_dates:
            plt.axvline(d, linestyle="--", color="green", alpha=0.6,
                        label="Hike" if first_h else None)
            first_h = False
        for d in cut_dates:
            plt.axvline(d, linestyle=":", color="red", alpha=0.6,
                        label="Cut" if first_c else None)
            first_c = False

        plt.title(f"{pc_col} — time series with ECB policy decisions")
        plt.xlabel("Meeting date")
        plt.ylabel("PC score (standardized units)")
        plt.legend()
        plt.tight_layout()

        outpath = OUTDIR / f"{pc_col}_timeseries_named.png"
        plt.savefig(outpath, dpi=150)
        plt.close()
        print("Saved:", outpath)

if __name__ == "__main__":
    main()