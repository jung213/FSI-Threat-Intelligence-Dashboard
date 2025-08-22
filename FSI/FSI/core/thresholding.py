import numpy as np
import pandas as pd

def by_account_quantile(df, score_col="score_ens", q=0.97):
    thr_map = df.groupby("account_id")[score_col].quantile(q)
    # thr_map이 NaN인 계정은 전체분포 q 사용
    global_thr = df[score_col].quantile(q)
    return thr_map, global_thr

def apply_account_threshold(df, thr_map, global_thr, score_col="score_ens"):
    def _flag(row):
        thr = thr_map.get(row["account_id"], global_thr)
        return int(row[score_col] >= thr)
    return df.apply(_flag, axis=1)

def alert_budget(df, score_col="score_ens", top_ratio=0.03):
    k = max(1, int(len(df) * top_ratio))
    thr = df[score_col].nlargest(k).min()
    return (df[score_col] >= thr).astype(int), thr

def rolling_quantile(prev_thr, new_thr, alpha=0.3):
    # 지수평활(레이트 변화 완화)
    return alpha*new_thr + (1-alpha)*prev_thr
