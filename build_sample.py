"""
Build a sampled flight dataset from the monthly BTS files.

Reads each monthly CSV in Data/, keeps only the selected columns, removes
cancelled and diverted flights, takes a random sample of ROWS_PER_FILE rows
from each month, and writes the combined sample to a single CSV.

Source: U.S. DOT Bureau of Transportation Statistics,
Reporting Carrier On-Time Performance (1987-present):
https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ
"""

from pathlib import Path
import pandas as pd

# constants for configuration
ROWS_PER_FILE = 5000
RANDOM_STATE = 42

DATA_DIR = Path(__file__).parent / "Data"
MONTH_FILES = [
    "January.csv",
    "February.csv",
    "March.csv",
    "April.csv",
    "May.csv",
    "June.csv",
]
OUTPUT_FILE = DATA_DIR / "flights_sample_2025H1.csv"

# Features known before departure (Tail_Number kept for aircraft-history features)
FEATURE_COLS = [
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Origin",
    "Dest",
    "OriginState",
    "DestState",
    "CRSDepTime",
    "DepTimeBlk",
    "CRSArrTime",
    "ArrTimeBlk",
    "CRSElapsedTime",
    "Distance",
    "DistanceGroup",
]
TARGET_COLS = ["ArrDelayMinutes", "ArrDel15"]
FILTER_COLS = ["Cancelled", "Diverted"]  # used for filtering, then dropped


def load_and_sample(path: Path) -> pd.DataFrame:
    """Load one monthly file, drop cancelled/diverted flights, and sample it."""
    df = pd.read_csv(
        path,
        usecols=FEATURE_COLS + TARGET_COLS + FILTER_COLS,
        dtype={"Tail_Number": str},
    )
    total_rows = len(df)

    df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)]
    df = df.drop(columns=FILTER_COLS)
    completed_rows = len(df)

    n = min(ROWS_PER_FILE, completed_rows)
    sample = df.sample(n=n, random_state=RANDOM_STATE)

    print(
        f"{path.name} total={total_rows}  "
        f"completed={completed_rows}  sampled={n}  "
        f"delay rate={sample['ArrDel15'].mean():.1%}"
    )
    return sample


def main() -> None:
    samples = [load_and_sample(DATA_DIR / name) for name in MONTH_FILES]

    combined = (
        pd.concat(samples)
        .sort_values(["FlightDate", "CRSDepTime"])
        .reset_index(drop=True)
    )
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"Wrote {OUTPUT_FILE} with shape {combined.shape}")
    print(f"Overall delay rate (ArrDel15): {combined['ArrDel15'].mean():.1%}")


if __name__ == "__main__":
    main()
