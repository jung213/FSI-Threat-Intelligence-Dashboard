import numpy as np
import pandas as pd

WINDOWS = [10, 30, 60]  # minutes

def add_time_index(df):
    df = df.sort_values(["account_id", "ts"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df

def add_slot_baseline(df):
    df["dow"] = df["ts"].dt.dayofweek
    df["hour"] = df["ts"].dt.hour
    df["slot"] = df["dow"].astype(str) + "-" + df["hour"].astype(str)
    med = df.groupby(["account_id","slot"])["amount"].median().rename("base_amount")
    df = df.join(med, on=["account_id","slot"])
    df["base_amount"] = df["base_amount"].fillna(df["amount"].median())
    df["amount_resid_slot"] = df["amount"] - df["base_amount"]
    return df

def add_ewma_baseline(df, alpha=0.1):
    df["base_amount_ewma"] = df.groupby("account_id")["amount"]\
        .transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    df["amount_resid_ewma"] = df["amount"] - df["base_amount_ewma"].fillna(df["amount"].median())
    return df

def add_cyclic_time(df):
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7.0)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7.0)
    return df

def _roll(df, minutes):
    # helper: 같은 계정 내에서 최근 minutes 윈도우
    df = df.sort_values(["account_id","ts"])
    grp = df.groupby("account_id", group_keys=False)
    return grp.apply(lambda g: g.set_index("ts")
                     .assign(
                         amt_sum = g.set_index("ts")["amount"].rolling(f"{minutes}min").sum(),
                         amt_mean= g.set_index("ts")["amount"].rolling(f"{minutes}min").mean(),
                         amt_std = g.set_index("ts")["amount"].rolling(f"{minutes}min").std(),
                         tx_cnt  = g.set_index("ts")["amount"].rolling(f"{minutes}min").count(),
                     ).reset_index())

def add_window_stats(df, windows=WINDOWS):
    out = df.copy()
    for w in windows:
        tmp = _roll(out[["ts","account_id","amount"]].copy(), w)
        out[f"amt_sum_{w}m"]  = tmp["amt_sum"].values
        out[f"amt_mean_{w}m"] = tmp["amt_mean"].values
        out[f"amt_std_{w}m"]  = tmp["amt_std"].values
        out[f"tx_cnt_{w}m"]   = tmp["tx_cnt"].values
    return out

def add_robust_stats(df):
    med = df.groupby("account_id")["amount"].transform("median")
    iqr = (df.groupby("account_id")["amount"].transform(lambda s: s.quantile(0.75)) -
           df.groupby("account_id")["amount"].transform(lambda s: s.quantile(0.25)))
    df["zscore"] = (df["amount"] - df.groupby("account_id")["amount"].transform("mean")) / \
                   (df.groupby("account_id")["amount"].transform("std") + 1e-6)
    df["rzscore"] = (df["amount"] - med) / (iqr + 1e-6)
    return df

def add_context_flags(df):
    # 간단한 컨텍스트 플래그/파생
    df["speed_kmh_clip"] = df["speed_kmh"].clip(upper=1200).fillna(0)
    # 첫 옥텟만 사용(ASN 대체): 0~255
    df["ip_oct1"] = df["ip"].str.split(".").str[0].astype(int, errors="ignore")
    df["device_change"] = (df["device_id"] != df.groupby("account_id")["device_id"].shift(1)).astype(int)
    return df

def build_features(df):
    df = add_time_index(df)
    df = add_slot_baseline(df)
    df = add_ewma_baseline(df)
    df = add_cyclic_time(df)
    df = add_window_stats(df)
    df = add_robust_stats(df)
    df = add_context_flags(df)
    return df
