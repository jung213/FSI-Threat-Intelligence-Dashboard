
from __future__ import annotations
import os, math, time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple, Dict, Set
from collections import deque

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sklearn.ensemble import IsolationForest

import streamlit as st
import plotly.express as px
import plotly.io as pio
import requests_cache

# SHAP surrogate용
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt

# ============================
# Page / Plotly theme
# ============================
st.set_page_config(page_title="FSI Threat Intel Dashboard", page_icon="💳", layout="wide")
pio.templates.default = "simple_white"

# ============================
# Global Config & Caching
# ============================
SESSION = None
CACHE = requests_cache.CachedSession(
    cache_name="/mnt/data/fsi_ti_cache",
    backend="sqlite",
    expire_after=timedelta(minutes=30),
)

ANOMALY_ALERT_THRESHOLD = int(os.getenv("ANOMALY_ALERT_THRESHOLD", "10"))

# ============================
# Queue & Webhook Functions
# ============================
def _init_queue_state():
    if "ioc_queue" not in st.session_state:
        st.session_state["ioc_queue"] = deque()
    if "ioc_results" not in st.session_state:
        st.session_state["ioc_results"] = {}

def send_webhook(text: str, webhook_url: str) -> bool:
    if not webhook_url:
        return False
    try:
        r = requests.post(webhook_url, json={"text": text}, timeout=10)
        return r.status_code in (200, 204)
    except Exception:
        return False

def _requests_session() -> requests.Session:
    global SESSION
    if SESSION is not None:
        return SESSION
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=0.4, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    SESSION = s
    return s

# ============================
# [모듈1] 이상거래 탐지 — 데이터 & 피처
# ============================
@dataclass
class Txn:
    ts: datetime
    account_id: str
    amount: float
    country: str
    device_id: str
    ip: str
    lat: float
    lon: float

COUNTRIES = [
    ("KR", 37.5665, 126.9780),
    ("US", 40.7128, -74.0060),
    ("JP", 35.6762, 139.6503),
    ("GB", 51.5074, -0.1278),
    ("DE", 52.5200, 13.4050),
    ("IN", 28.6139, 77.2090),
]

def haversine(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def haversine_vec(lat1, lon1, lat2, lon2):
    lat1 = lat1.astype(float).ffill().fillna(0)
    lon1 = lon1.astype(float).ffill().fillna(0)
    lat2 = lat2.astype(float).fillna(0)
    lon2 = lon2.astype(float).fillna(0)
    res = []
    for a,b,c,d in zip(lat1, lon1, lat2, lon2):
        res.append(haversine(a,b,c,d))
    return pd.Series(res)

def synth_txn_stream(n: int = 500, accounts: int = 5, seed: int = 42, use_kst: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = datetime.utcnow() - timedelta(hours=1)
    if use_kst:
        base = base.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=9)))
    acct_ids = [f"ACCT{1000+i}" for i in range(accounts)]
    devices = [f"dev-{i}" for i in range(accounts*2)]

    for i in range(n):
        ts = base + timedelta(seconds=i*6)
        acct = rng.choice(acct_ids)
        country, lat, lon = COUNTRIES[rng.integers(0, len(COUNTRIES))]
        amount = max(0.0, float(rng.normal(85, 40)))
        if rng.random() < 0.03:
            amount *= float(rng.uniform(5, 20))
        device = rng.choice(devices)
        ip = f"{rng.integers(3, 223)}.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,255)}"
        rows.append((ts, acct, amount, country, device, ip, lat, lon))

    df = pd.DataFrame(rows, columns=["ts","account_id","amount","country","device_id","ip","lat","lon"])
    df = df.sort_values(["account_id","ts"]).reset_index(drop=True)
    df[["prev_ts","prev_lat","prev_lon"]] = df.groupby("account_id")[['ts','lat','lon']].shift(1)
    df["gap_min"] = (df["ts"] - df["prev_ts"]).dt.total_seconds() / 60
    df["dist_km"] = haversine_vec(df["prev_lat"], df["prev_lon"], df["lat"], df["lon"])  # km
    df["speed_kmh"] = df["dist_km"] / (df["gap_min"] / 60.0)
    df.loc[df["gap_min"].isna(), ["gap_min","dist_km","speed_kmh"]] = 0
    return df

def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'])
    df['dow'] = df['ts'].dt.dayofweek  # 0~6
    df['hour'] = df['ts'].dt.hour
    df['slot'] = df['dow'].astype(str) + '-' + df['hour'].astype(str)

    # 계정×슬롯별 중앙값 baseline
    median_map = df.groupby(['account_id','slot'])['amount'].median().rename('base_amount')
    df = df.join(median_map, on=['account_id','slot'])
    df['base_amount'] = df['base_amount'].fillna(df['amount'].median())
    df['amount_resid'] = df['amount'] - df['base_amount']

    # 사이클릭 인코딩
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24.0)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24.0)
    df['dow_sin']  = np.sin(2*np.pi*df['dow']/7.0)
    df['dow_cos']  = np.cos(2*np.pi*df['dow']/7.0)
    return df

# ============================
# [모듈1] 세분화 모델 (상위 N계정 전용 + 공용)
# ============================
FEATURES_IFOREST = ['amount_resid','gap_min','dist_km','speed_kmh','hour_sin','hour_cos','dow_sin','dow_cos']

def top_accounts(df: pd.DataFrame, top_n: int = 3):
    counts = df['account_id'].value_counts()
    return set(counts.head(top_n).index)

def fit_isoforest_segmented(df: pd.DataFrame, contamination: float = 0.03, top_n: int = 3):
    models: dict[tuple, IsolationForest] = {}
    tops = top_accounts(df, top_n)
    # 계정 전용
    for acct in tops:
        sub = df[df['account_id'] == acct]
        X = sub[FEATURES_IFOREST].fillna(0)
        iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=7, n_jobs=-1)
        iso.fit(X)
        models[('acct', acct)] = iso
    # 나머지 공용
    rest = df[~df['account_id'].isin(tops)]
    Xr = rest[FEATURES_IFOREST].fillna(0)
    iso_rest = IsolationForest(n_estimators=200, contamination=contamination, random_state=7, n_jobs=-1)
    iso_rest.fit(Xr)
    models[('global','rest')] = iso_rest
    return models, tops

def score_segmented(models: dict, tops: set, row: pd.Series) -> float:
    key = ('acct', row['account_id']) if row['account_id'] in tops else ('global','rest')
    iso = models[key]
    # DataFrame으로 넣어 feature name 일치 → sklearn 경고 제거
    x_df = pd.DataFrame([[row[c] for c in FEATURES_IFOREST]], columns=FEATURES_IFOREST)
    return float(-iso.score_samples(x_df)[0])  # 높을수록 이상치

# ============================
# [모듈2] 위협 인텔 (VirusTotal만)
# ============================
VT_BASE = "https://www.virustotal.com/api/v3"

def get_secret(name: str, env_name: str | None = None, default: str | None = None) -> str | None:
    try:
        if name in st.secrets:
            return st.secrets.get(name)
    except Exception:
        pass
    if env_name:
        val = os.getenv(env_name)
        if val:
            return val
    return default

def vt_scan_url(url: str, api_key: str) -> str | None:
    if not api_key:
        return None
    s = _requests_session()
    resp = s.post(
        f"{VT_BASE}/urls",
        headers={"x-apikey": api_key},
        data={"url": url},
        timeout=20,
    )
    if resp.status_code in (200, 202):
        try:
            return resp.json()["data"]["id"]
        except Exception:
            return None
    return None

def vt_get_analysis(analysis_id: str, api_key: str) -> dict | None:
    if not api_key or not analysis_id:
        return None
    s = _requests_session()
    r = s.get(f"{VT_BASE}/analyses/{analysis_id}", headers={"x-apikey": api_key}, timeout=20)
    if r.status_code == 200:
        return r.json()
    return None

# ============================
# [모듈3] SHAP surrogate (XGBoost)
# ============================
SURR_FEATURES = FEATURES_IFOREST

def train_shap_surrogate(df: pd.DataFrame, target_col: str = 'seg_score'):
    # 표본 추출(속도 위해)
    sample = df.sample(n=min(len(df), 500), random_state=7)
    X = sample[SURR_FEATURES].fillna(0)
    y = sample[target_col].values

    reg = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=7,
        n_jobs=-1,
    )
    reg.fit(X, y)
    explainer = shap.TreeExplainer(reg)
    shap_values = explainer(X)
    return reg, explainer, shap_values, sample

# ============================
# Labeling UI (save to data/labels.csv)
# ============================
def labeling_ui(df_top_alerts: pd.DataFrame):
    here = Path(__file__).resolve()
    project_root = (here.parent.parent if here.parent.name.lower()=="app" else here.parent)
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    label_path = data_dir / "labels.csv"

    base = df_top_alerts[['ts','account_id','amount','seg_score']].copy()
    base["label"] = False
    if "label_editor_df" not in st.session_state:
        st.session_state["label_editor_df"] = base

    edited = st.data_editor(
        st.session_state["label_editor_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="label_editor",
    )
    st.session_state["label_editor_df"] = edited

    c1,c2 = st.columns([1,1])
    with c1:
        if st.button("라벨 저장", type="primary", use_container_width=True):
            try:
                cols = [c for c in ["ts","account_id","amount","seg_score","label"] if c in edited.columns]
                if not cols:
                    st.error("저장할 컬럼이 없습니다.")
                    return
                save = edited[cols].copy()
                if label_path.exists():
                    prev = pd.read_csv(label_path)
                    save = pd.concat([prev, save], ignore_index=True)
                save.to_csv(label_path, index=False, encoding="utf-8-sig")
                st.success(f"Saved to {label_path}")
                csv_bytes = save.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button("labels.csv 다운로드", data=csv_bytes, file_name="labels.csv", mime="text/csv", use_container_width=True)
            except Exception as e:
                st.exception(e)
    with c2:
        with st.expander("경로/환경 디버그"):
            st.write({"cwd": str(Path.cwd()), "label_path": str(label_path)})

# ============================
# Live Mode Functions
# ============================
def _init_live_state(init_rows: int, use_kst: bool):
    if "live_df" not in st.session_state:
        base_df = synth_txn_stream(init_rows, use_kst=use_kst)
        st.session_state.live_df = base_df.copy()
        st.session_state.last_ts = base_df["ts"].max()
        st.session_state.tick = 0

def _append_new_tx(n_new: int):
    df = st.session_state.live_df
    last_ts = st.session_state.last_ts
    rng = np.random.default_rng(int(time.time()))
    acct_ids = df["account_id"].unique().tolist()
    devices = df["device_id"].unique().tolist()
    rows = []
    for i in range(n_new):
        ts = last_ts + timedelta(seconds=6*(i+1))
        acct = rng.choice(acct_ids)
        country, lat, lon = COUNTRIES[rng.integers(0, len(COUNTRIES))]
        amount = max(0.0, float(rng.normal(85, 40)))
        if rng.random() < 0.03:
            amount *= float(rng.uniform(5, 20))
        device = rng.choice(devices)
        ip = f"{rng.integers(3,223)}.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,255)}"
        rows.append((ts, acct, amount, country, device, ip, lat, lon))
    new_df = pd.DataFrame(rows, columns=["ts","account_id","amount","country","device_id","ip","lat","lon"])
    out = pd.concat([df, new_df], ignore_index=True).sort_values(["account_id","ts"]).reset_index(drop=True)
    out[["prev_ts","prev_lat","prev_lon"]] = out.groupby("account_id")[['ts','lat','lon']].shift(1)
    out["gap_min"] = (out["ts"] - out["prev_ts"]).dt.total_seconds() / 60
    out["dist_km"] = haversine_vec(out["prev_lat"], out["prev_lon"], out["lat"], out["lon"])
    out["speed_kmh"] = out["dist_km"] / (out["gap_min"] / 60.0)
    out.loc[out["gap_min"].isna(), ["gap_min","dist_km","speed_kmh"]] = 0
    st.session_state.live_df = out
    st.session_state.last_ts = out["ts"].max()
    st.session_state.tick = st.session_state.get("tick", 0) + 1

# ============================
# Sidebar (Combined from test1.py + test2.py)
# ============================
with st.sidebar:
    st.subheader("⚙️ 설정")
    
    # 시간대 옵션 (test2.py에서)
    use_kst = st.toggle("🕒 KST(UTC+9) 타임스탬프 사용", value=False, key="use_kst")
    
    # Live 모드 설정 (test2.py에서)
    live_mode = st.toggle("🔴 Live 모드", value=False, key="live_mode")
    refresh_sec = st.number_input("갱신 주기(초)", 1, 60, 3, key="refresh_sec")
    new_per_tick = st.number_input("틱당 신규 거래 수", 1, 200, 20, key="new_per_tick")
    window_rows = st.number_input("학습용 최근 샘플 수", 200, 5000, 1200, key="window_rows")
    
    st.markdown("---")
    
    # API Key 및 기본 설정
    VT_KEY = st.text_input("VirusTotal API Key", 
                           value=os.getenv("VIRUSTOTAL_API_KEY", st.secrets.get("VT_API_KEY", "") if "VT_API_KEY" in st.secrets else ""),
                           type="password", key="vt_api_key")
    n_rows = st.slider("거래 생성량", 200, 2000, 600, step=100, key="init_rows")
    contam = st.slider("IF 이상치 비율", 0.01, 0.10, 0.03, step=0.01, key="contamination")
    top_n = st.slider("전용 모델 계정 수", 1, 5, 3, key="topn_accounts")
    
    st.caption("API 키가 없으면 VT 조회는 건너뜁니다.")

    # 챗봇 카드 (test1.py에서)
    st.markdown("---")
    st.subheader("🤖 챗봇")
    try:
        st.page_link(
            "pages/fsi_chat_bot.py",
            label="보안 챗봇 열기",
            icon=":material/smart_toy:"
        )
    except Exception:
        if st.button("🤖 보안 챗봇 열기", use_container_width=True):
            st.switch_page("pages/fsi_chat_bot.py")

    # 사용 팁
    with st.expander("ℹ️ 사용 팁"):
        st.markdown("""
        - **Live 모드 ON** → 아래 Tick 카운트가 증가해야 정상
        - **갱신 주기**: 너무 빠르면 3~5초로, 느리면 1~2초로 조정
        - **API 키**: VirusTotal API 키가 없으면 VT 조회는 건너뜁니다
        - **Slack 알림**: ALERT_WEBHOOK_URL 환경변수 설정 시 이상거래 알림
        """)

# ============================
# Header
# ============================
st.title("💳 FSI Threat Intelligence Dashboard")
st.caption("이상거래 탐지 · VirusTotal · SHAP · 라벨링 · Live 모드 · Slack 알림")

# ============================
# Pipeline (data → features → model)
# ============================
# 데이터 준비 (라이브 모드면 세션 누적 사용)
if live_mode:
    _init_live_state(n_rows, use_kst)
    _append_new_tx(int(new_per_tick))
    raw = st.session_state.live_df.tail(int(window_rows)).copy()  # 슬라이딩 윈도우
    feat = add_seasonality_features(raw)  # 캐시 우회 (항상 최신 반영)
else:
    raw = synth_txn_stream(n_rows, use_kst=use_kst)
    feat = add_seasonality_features(raw)

models, tops = fit_isoforest_segmented(feat, contam, top_n)

# 세분화 스코어 계산
feat['seg_score'] = feat.apply(lambda r: score_segmented(models, tops, r), axis=1)
thr_global = float(np.quantile(feat['seg_score'], 0.97))
feat['is_anomaly'] = (feat['seg_score'] >= thr_global).astype(int)

# ============================
# KPI
# ============================
total = len(feat)
anoms = int(feat['is_anomaly'].sum())
avg_amt = float(feat['amount'].mean())
fast_mv = int((feat['speed_kmh'] > 900).sum())
tick_count = st.session_state.get("tick", 0) if live_mode else 0

k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("거래 총수", f"{total:,}")
with k2: st.metric("이상 탐지 건수", anoms)
with k3: st.metric("평균 거래금액", f"₩ {avg_amt:,.0f}")
with k4: st.metric("고속 이동 이벤트(>900km/h)", fast_mv)
with k5: st.metric("Live Tick", tick_count)

# ============================
# 이상거래 대시보드
# ============================
st.subheader("📊 이상거래 탐지 ")
fig = px.line(feat.sort_values('ts'), x='ts', y='amount', color='is_anomaly', markers=False)
st.plotly_chart(fig, use_container_width=True)
st.caption("빨간 범례(1)는 이상치로 간주된 거래 — 모델: 상위 계정 전용 + 공용")

topN = st.slider("라벨링 대상 상위 N", 10, 200, 50)
top_alerts = feat.sort_values('seg_score', ascending=False).head(topN)[[
    'ts','account_id','amount','amount_resid','country','ip','gap_min','dist_km','speed_kmh','seg_score','is_anomaly'
]]
st.dataframe(top_alerts, use_container_width=True, height=360)

# ============================
# VirusTotal + SHAP (two columns)
# ============================
left, right = st.columns([1.15, 1.0])

with left:
    st.subheader("🛰️ VirusTotal URL 분석")
    url_ioc = st.text_input("IOC URL", placeholder="https://...", key="ioc_url")
    if st.button("VirusTotal 조회", type="primary"):
        if not VT_KEY:
            st.warning("API Key가 없어 VT 조회를 건너뜁니다.")
        else:
            with st.spinner("VirusTotal 조회 중..."):
                vt_id = vt_scan_url(url_ioc, VT_KEY)
                vt_res = vt_get_analysis(vt_id, VT_KEY) if vt_id else None
            if vt_res:
                try:
                    attr = vt_res.get("data", {}).get("attributes", {})
                    stats = attr.get("stats", {}) or attr.get("last_analysis_stats", {})
                    st.json(stats)
                except Exception:
                    st.json(vt_res)
            else:
                st.warning("VT 결과 없음 (키 누락/레이트리밋/분석 대기 가능)")

with right:
    st.subheader("🔍 SHAP 분석")
    st.caption("IF 점수를 XGBoost로 근사하여 SHAP로 설명")
    if st.button("SHAP 실행"):
        with st.spinner("SHAP 계산 중..."):
            reg, explainer, shap_values, sample = train_shap_surrogate(feat, target_col='seg_score')
        # 요약 바
        fig1 = plt.figure(figsize=(6, 4))
        try:
            shap.summary_plot(shap_values, features=sample[SURR_FEATURES], feature_names=SURR_FEATURES, plot_type="bar", show=False)
        except Exception:
            shap.summary_plot(shap_values.values, features=sample[SURR_FEATURES], feature_names=SURR_FEATURES, plot_type="bar", show=False)
        st.pyplot(fig1, bbox_inches='tight', pad_inches=0.1)

# ============================
# Slack 알림 (이상 건수 임계)
# ============================
_init_queue_state()  # st.session_state 사용 시 안전

# 중복 알림 방지용: 15분 쿨다운
COOLDOWN_SEC = 15 * 60
now = datetime.utcnow()
last_sent = st.session_state.get("last_anomaly_alert_at")
ALERT_WEBHOOK = get_secret("ALERT_WEBHOOK_URL", "ALERT_WEBHOOK_URL", None)

if ALERT_WEBHOOK and anoms >= ANOMALY_ALERT_THRESHOLD:
    should_send = (last_sent is None) or ((now - last_sent).total_seconds() > COOLDOWN_SEC)
    if should_send:
        msg = (
            f":rotating_light: 이상거래 경보\n"
            f"- 탐지 건수: *{anoms}* (임계 {ANOMALY_ALERT_THRESHOLD}+)\n"
            f"- 샘플: 상위 3건 계정/금액\n"
        )
        try:
            top3 = (
                feat.loc[feat["is_anomaly"] == 1, ["account_id", "amount", "seg_score"]]
                .sort_values("seg_score", ascending=False)
                .head(3)
            )
            lines = [
                f"  • {r.account_id} — ₩{r.amount:,.0f} (score={r.seg_score:.3f})" for r in top3.itertuples()
            ]
            msg += "\n".join(lines) if lines else "  • (표시할 샘플 없음)"
        except Exception:
            pass

        if send_webhook(msg, ALERT_WEBHOOK):
            st.session_state["last_anomaly_alert_at"] = now
            st.success(f"Slack 알림 전송 완료! (이상거래 {anoms}건)")

# ============================
# 라벨링 UI
# ============================
st.subheader("🏷️ 라벨링")
st.caption("탐지 상위 건을 수동 라벨링하여 지도학습에 활용")
labeling_ui(top_alerts)

# ============================
# Footer
# ============================
st.markdown("---")
st.markdown("ⓒ 2025 금융보안아카데미 • FDS/SIEM 데모 • 통합 버전 (test1 + test2)")

# ============================
# Live auto refresh (sleep + rerun)
# ============================
if live_mode:
    tz = timezone(timedelta(hours=9)) if use_kst else timezone.utc
    st.caption(f"⏱️ Live Tick: {st.session_state.get('tick',0)} · 최근 샘플: {len(feat):,} 건 · TZ={'KST' if use_kst else 'UTC'} · 마지막 갱신: {datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}")
    time.sleep(float(refresh_sec))
    st.rerun()