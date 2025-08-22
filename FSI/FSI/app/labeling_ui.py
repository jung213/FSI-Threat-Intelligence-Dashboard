import pandas as pd
import streamlit as st
from pathlib import Path
import io

def labeling_ui(df_top_alerts: pd.DataFrame):
    st.subheader("📝 라벨링")
    st.caption("탐지 상위 N건을 수동 라벨링하여 지도학습에 활용")

    # --- 프로젝트 루트/data/labels.csv로 고정 ---
    here = Path(__file__).resolve()
    project_root = (here.parent.parent if here.parent.name.lower() == "app" else here.parent)
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    label_path = data_dir / "labels.csv"

    # 디버그 정보(혹시 안 저장될 때 확인)
    with st.expander("경로/환경 디버그"):
        st.write({"__file__": str(here), "project_root": str(project_root), "label_path": str(label_path)})
        st.write({"cwd": str(Path.cwd())})

    # --- 에디터 (세션 보존) ---
    base = df_top_alerts.copy()
    base["label"] = False
    # 에디터 상태 보존
    if "label_editor_df" not in st.session_state:
        st.session_state["label_editor_df"] = base

    edited = st.data_editor(
        st.session_state["label_editor_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="label_editor",
    )
    # 최신 값 세션에 유지
    st.session_state["label_editor_df"] = edited

    # --- 저장 버튼 ---
    if st.button("라벨 저장", type="primary", use_container_width=True):
        try:
            # 필요한 컬럼만 안전하게 선택
            cols = [c for c in ["ts","account_id","amount","seg_score","label"] if c in edited.columns]
            if not cols:
                st.error("저장할 컬럼이 없습니다. (ts/account_id/amount/seg_score/label)")
                return
            save = edited[cols].copy()

            # 파일에 append 저장
            if label_path.exists():
                prev = pd.read_csv(label_path)
                save = pd.concat([prev, save], ignore_index=True)

            # 디스크 저장
            save.to_csv(label_path, index=False, encoding="utf-8-sig")

            # 디스크 검증 + 다운로드(메모리 버퍼)
            ok = label_path.exists() and label_path.stat().st_size > 0
            st.dataframe(save.tail(10), use_container_width=True, height=200)
            csv_bytes = save.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("labels.csv 다운로드", data=csv_bytes, file_name="labels.csv", mime="text/csv",
                               use_container_width=True)

            if ok:
                st.success(f"Saved to {label_path}")
            else:
                st.error("파일이 생성되지 않았습니다. 권한/경로를 확인하세요.")

        except Exception as e:
            st.exception(e)
