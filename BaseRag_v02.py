import traceback
import streamlit as st

from trag.news_daemon import ensure_daemon_started

# 앱 시작 시 뉴스 데몬을 백그라운드로 띄움(이미 떠 있으면 아무것도 안 함)
try:
    ensure_daemon_started()
except Exception as e:
    st.warning(f"뉴스 데몬 시작 실패: {e}")

    

st.set_page_config(page_title="TRAG", layout="wide")

try:
    from trag.config import UI_TITLE, AVAILABLE_LLM_MODELS
    from trag.rag import build_conversational_rag_chain

    st.header(UI_TITLE)
    selected_model = st.selectbox("Select Ollama Model", AVAILABLE_LLM_MODELS)

    chain = build_conversational_rag_chain(selected_model)
    if chain is None:
        st.error("초기화 실패: PDF_PATH 또는 Chroma/임베딩 설정을 확인해주세요.")
    else:
        from trag.ui import render_chat
        render_chat(chain)

except Exception:
    st.error("앱 초기화 중 예외가 발생했습니다. 아래 Traceback을 확인해주세요.")
    st.code(traceback.format_exc())