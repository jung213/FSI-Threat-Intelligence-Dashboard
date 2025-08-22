import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from typing import Dict, Tuple, List

def fit_isoforest(X, contamination=0.03, seed=7):
    iso = IsolationForest(n_estimators=300, contamination=contamination,
                          random_state=seed, n_jobs=-1)
    iso.fit(X)
    return iso

def score_isoforest(iso, X):
    return -iso.score_samples(X)  # 높을수록 이상

def fit_lof(X, n_neighbors=20):
    # LOF는 fit_predict에서 -1/1만 주는 버전 많음 → novelty=True 불가 시 score_samples 대체로 negative_outlier_factor_
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
    y = lof.fit_predict(X)
    # score: -negative_outlier_factor_ (높을수록 이상)
    score = -lof.negative_outlier_factor_
    return lof, score

def zscore_minmax(s):
    s = (s - np.nanmean(s)) / (np.nanstd(s) + 1e-6)
    s = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-6)
    return s

def ensemble_scores(df_feat: pd.DataFrame, features: List[str], contamination=0.03) -> pd.DataFrame:
    X = df_feat[features].fillna(0).values
    # IForest
    iso = fit_isoforest(X, contamination=contamination)
    s_iso = score_isoforest(iso, X)
    # LOF
    _, s_lof = fit_lof(X, n_neighbors=min(20, max(5, int(np.sqrt(len(df_feat))))))
    # 정규화 후 가중 평균
    s_iso_n = zscore_minmax(s_iso)
    s_lof_n = zscore_minmax(s_lof)
    s_ens = 0.7 * s_iso_n + 0.3 * s_lof_n
    out = df_feat.copy()
    out["score_iso"] = s_iso
    out["score_lof"] = s_lof
    out["score_ens"] = s_ens
    return out, iso
