import os
import re
import pandas as pd
import streamlit as st
from datetime import datetime

# LangChain / LangGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing import TypedDict

# ----------------------------
# Streamlit Page
# ----------------------------
st.set_page_config(page_title="보안 라벨링 CSV 챗봇", page_icon="🤖", layout="wide")
st.title("🤖 보안 라벨링 CSV 챗봇")

with st.expander("사용 방법", expanded=False):
    st.markdown("""
- `labels.csv`를 업로드하면, 모델이 데이터를 요약하고 설명해줍니다.
- 질문 예시:
  - `seg_score가 뭐고, 라벨과 어떻게 같이 써요?`
  - `라벨 True 비율이랑, 계정별 평균 seg_score 상위 5개 알려줘`
  - `시간대별 라벨 True 비율 경향 설명해줘`
- 모델: **gpt-4o-mini** (OPENAI_API_KEY 필요)
    """)

# ----------------------------
# Helpers: 데이터 요약/통계
# ----------------------------
RE_DATA = re.compile(r"(비율|상위|평균|분포|추이|랭킹|순위|count|건수|퍼센트|퍼센티지|증가|감소)")

def _ensure_columns(df: pd.DataFrame):
    """필수 컬럼이 없을 경우 안전하게 보정"""
    for c in ["ts","account_id","amount","seg_score","label"]:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # ts 시간 변환
    if "ts" in out.columns:
        try:
            out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
        except Exception:
            pass
    # label bool 변환
    if "label" in out.columns:
        out["label"] = out["label"].astype(str).str.lower().isin(["true","1","t","y","yes"])
    # 수치형 변환
    for c in ["amount","seg_score"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def summarize_df_for_prompt(df: pd.DataFrame, topn: int = 5) -> str:
    """LLM 컨텍스트로 넣을 요약 텍스트 생성 (보안담당자 친화)"""
    dff = _coerce_types(_ensure_columns(df))
    n = len(dff)
    true_rate = float(dff["label"].mean()) if n else 0.0
    avg_amt   = float(dff["amount"].mean()) if "amount" in dff else 0.0
    avg_seg   = float(dff["seg_score"].mean()) if "seg_score" in dff else 0.0

    lines = []
    lines.append(f"[스키마] columns={list(df.columns)}")
    lines.append(f"[기본] 건수={n:,}, 라벨 True 비율={true_rate:.2%}, 평균 금액={avg_amt:,.1f}, 평균 seg_score={avg_seg:.3f}")

    # 계정별 통계
    if "account_id" in dff.columns:
        grp = dff.groupby("account_id").agg(
            cnt=("account_id","size"),
            true_rate=("label","mean"),
            avg_seg=("seg_score","mean"),
            avg_amt=("amount","mean"),
        ).reset_index()

        top_seg = grp.sort_values("avg_seg", ascending=False).head(topn)
        lines.append("[계정별 평균 seg_score 상위]")
        for _, r in top_seg.iterrows():
            lines.append(f"- {r['account_id']}: avg_seg={r['avg_seg']:.3f}, cnt={int(r['cnt'])}, true_rate={r['true_rate']:.2%}")

        top_tr = grp.sort_values("true_rate", ascending=False).head(topn)
        lines.append("[계정별 라벨 True 비율 상위]")
        for _, r in top_tr.iterrows():
            lines.append(f"- {r['account_id']}: true_rate={r['true_rate']:.2%}, cnt={int(r['cnt'])}, avg_seg={r['avg_seg']:.3f}")

    # 시간대 경향
    if "ts" in dff.columns and pd.api.types.is_datetime64_any_dtype(dff["ts"]):
        dff["hour"] = dff["ts"].dt.hour
        hour_agg = dff.groupby("hour")["label"].mean().sort_index()
        if len(hour_agg) > 0:
            lines.append("[시간대별 라벨 True 비율(간이)] " + ", ".join([f"{int(h)}시={v:.1%}" for h, v in hour_agg.items()]))

    return "\n".join(lines)

def build_system_prompt(context_summary: str) -> str:
    return (
        "당신은 금융 보안담당자에게 설명하는 어시스턴트입니다. "
        "전문용어는 쉬운 말로 풀고, 항상 '결론 → 이유 → 권고' 순서로 5~7줄 핵심 요약을 먼저 제시하세요. "
        "가능하면 간단한 수치(%) 예시를 포함하세요. "
        "아래는 사용자가 업로드한 라벨 CSV의 요약입니다.\n\n"
        f"{context_summary}\n\n"
        "설명/요약을 요청하면 위 데이터를 근거로 답변하고, "
        "특정 통계/상위/비율/평균 등 데이터형 질문이면 반드시 이 요약 수치들을 근거로 삼아 답하세요."
    )

# ----------------------------
# LangGraph 상태 & 노드
# ----------------------------
class BotState(TypedDict):
    question: str
    route: str
    answer: str

def router_node(state: BotState) -> BotState:
    """질문이 데이터형인지(비율/상위/평균 등) 또는 설명형인지 라우팅"""
    q = state["question"]
    route = "data" if RE_DATA.search(q) else "plain"
    return {"question": q, "route": route, "answer": ""}

def answer_plain_node(state: BotState, llm: ChatOpenAI, sys_prompt: str) -> BotState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "{q}")
    ])
    chain = prompt | llm | StrOutputParser()
    out = chain.invoke({"q": state["question"]})
    return {"question": state["question"], "route": "plain", "answer": out}

def answer_data_node(state: BotState, llm: ChatOpenAI, sys_prompt: str) -> BotState:
    """데이터 질문: 컨텍스트 요약을 바탕으로 깔끔한 운영 관점 답변"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt + "\n\n데이터형 질문이므로 수치/순위/비율을 우선 설명하고, 실무적 권고를 덧붙이세요."),
        ("human", "{q}")
    ])
    chain = prompt | llm | StrOutputParser()
    out = chain.invoke({"q": state["question"]})
    return {"question": state["question"], "route": "data", "answer": out}

def build_graph(llm: ChatOpenAI, sys_prompt: str):
    graph = StateGraph(BotState)
    graph.add_node("router", router_node)
    # 함수형 노드를 감싸는 래퍼
    graph.add_node("plain", lambda s: answer_plain_node(s, llm, sys_prompt))
    graph.add_node("data",  lambda s: answer_data_node(s,  llm, sys_prompt))
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {"plain": "plain", "data": "data"},
    )
    graph.add_edge("plain", END)
    graph.add_edge("data", END)
    return graph.compile()

# ----------------------------
# UI: 파일 업로드 & 챗
# ----------------------------
uploaded = st.file_uploader("라벨 CSV 업로드", type=["csv"])
if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

if uploaded is None:
    st.info("labels.csv를 업로드하면 챗봇이 활성화됩니다. (권장 열: ts, account_id, amount, seg_score, label)")
else:
    # CSV 읽기 (인코딩 자동 대응)
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, encoding="utf-8-sig")

    # 컨텍스트 요약
    context_summary = summarize_df_for_prompt(df)
    st.markdown("#### 데이터 요약(자동)")
    st.code(context_summary, language="text")

    # 모델 준비
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY가 설정되지 않았습니다. 환경변수 설정 후 다시 시도하세요.")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

    # LangGraph 빌드
    sys_prompt = build_system_prompt(context_summary)
    graph = build_graph(llm, sys_prompt)

    # 이전 대화 렌더
    for role, content in st.session_state.chat_hist:
        with st.chat_message(role):
            st.write(content)

    # 입력
    q = st.chat_input("무엇이 궁금하신가요? (예: 라벨 True 비율과 상위 계정은?)")
    if q:
        st.session_state.chat_hist.append(("user", q))
        with st.chat_message("user"):
            st.write(q)

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                result = graph.invoke({"question": q, "route": "", "answer": ""})
                st.write(result["answer"])
        st.session_state.chat_hist.append(("assistant", result["answer"]))
