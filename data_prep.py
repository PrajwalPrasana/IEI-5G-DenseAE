"""
data_prep.py -- load 5G-NIDD Combined.csv, clean, split, scale.

Verified facts about the real file (1,215,890 rows):
  * Label = Benign (477,737) / Malicious (738,153); per-class info is in "Attack Type".
  * sVid is blank in ~90.6% of rows, dTtl/dHops ~77.6%, sTtl/sHops <0.02%.
    Missing values are filled with 0 (documented choice, applied identically to all data).
  * Proto is text; mapped with a FIXED alphabetical table so codes never change between runs.

Split (per seed):
  * Benign flows: 70% train / 10% validation / 20% test.
  * ALL malicious flows go to the test set (the model never trains on attacks,
    so there is nothing to hold back).
  * MinMaxScaler is fitted on the training benign flows only.

Usage: python data_prep.py --input Combined.csv --seeds 42 123 256 789 1024
"""
import argparse, os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import config as C

KEEP = C.FEATURE_COLUMNS + [C.LABEL_COLUMN, C.ATTACK_TYPE_COLUMN]


def load_and_clean(path):
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    df = df[KEEP].copy()
    if not pd.api.types.is_numeric_dtype(df["Proto"]):
        m = {p: i for i, p in enumerate(sorted(C.PROTO_VALUES))}
        df["Proto"] = df["Proto"].map(m)
        if df["Proto"].isna().any():
            raise ValueError("Unexpected Proto value in CSV")
    df[C.FEATURE_COLUMNS] = df[C.FEATURE_COLUMNS].fillna(0.0).astype("float32")
    return df


def split(df, seed):
    ben = df[df[C.LABEL_COLUMN] == C.BENIGN_LABEL]
    mal = df[df[C.LABEL_COLUMN] == C.MALICIOUS_LABEL]
    train, rest = train_test_split(ben, test_size=0.30, random_state=seed)
    val, test_ben = train_test_split(rest, test_size=2 / 3, random_state=seed)
    holdout = pd.concat([test_ben, mal]).sample(frac=1.0, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), holdout.reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    a = ap.parse_args()
    df = load_and_clean(a.input)
    print(f"Loaded {len(df)} rows")
    for s in a.seeds:
        tr, va, ho = split(df, s)
        sc = MinMaxScaler().fit(tr[C.FEATURE_COLUMNS].values)
        tr.to_parquet(os.path.join(C.DATA_DIR, f"train_benign_seed{s}.parquet"))
        va.to_parquet(os.path.join(C.DATA_DIR, f"val_benign_seed{s}.parquet"))
        ho.to_parquet(os.path.join(C.DATA_DIR, f"holdout_test_seed{s}.parquet"))
        joblib.dump(sc, os.path.join(C.DATA_DIR, f"scaler_seed{s}.joblib"))
        print(f"[seed {s}] train={len(tr)} val={len(va)} holdout={len(ho)} "
              f"(benign {int((ho[C.LABEL_COLUMN]==C.BENIGN_LABEL).sum())}, "
              f"malicious {int((ho[C.LABEL_COLUMN]==C.MALICIOUS_LABEL).sum())})")
    print("Holdout attack-type counts (last seed):")
    print(ho[C.ATTACK_TYPE_COLUMN].value_counts().to_string())
