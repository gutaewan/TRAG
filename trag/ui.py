import streamlit as st

from .config import DATA_DIR
from .vectorstore import (
    sync_pdf_dir,
    save_uploaded_pdf_to_dir,
    list_ingested_pdfs,
)


def render_chat(conversational_chain):
    # ====== (선택) 상단 상태 ======
    st.caption(f"📁 데이터 폴더: {DATA_DIR}  (이 폴더의 PDF 전체를 대상으로 신규만 임베딩합니다)")

    # ====== 채팅 세션 상태 ======
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "📎 PDF를 첨부하려면 아래에 드래그앤드롭 해주세요. 그리고 질문을 입력해 주세요 🙂"}
        ]

    # ====== 기존 채팅 메시지 렌더링 ======
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # ====== (2번 방식) 업로더를 채팅 흐름 안에 삽입 ======
    with st.chat_message("assistant"):
        st.write("여기에 PDF를 드래그앤드롭 하시면 `./data`에 저장되고, **새로 추가된 PDF만** 임베딩됩니다. 📚")

        uploaded_files = st.file_uploader(
            "PDF 업로드",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    # 업로드 처리 (업로드되면 바로 저장 + 동기화)
    if uploaded_files:
        # 사용자가 업로드한 파일들을 '채팅 메시지'처럼 표시
        for uf in uploaded_files:
            st.chat_message("human").write(f"📎 업로드됨: {uf.name}")

        # 실제 저장/임베딩
        with st.spinner("업로드 파일 저장 및 신규 PDF 임베딩 중..."):
            # 1) ./data에 저장
            for uf in uploaded_files:
                save_uploaded_pdf_to_dir(uf, DATA_DIR)

            # 2) ./data 전체 스캔 → 신규 PDF만 임베딩
            result = sync_pdf_dir(DATA_DIR)

        # 결과를 assistant 메시지처럼 표시
        summary_lines = [
            f"✅ 동기화 완료!",
            f"- 총 PDF: {result.get('total_pdf', 0)}개",
            f"- 신규 임베딩: {len(result.get('added', []))}개",
            f"- 기존 스킵: {len(result.get('skipped', []))}개",
        ]
        if result.get("failed"):
            summary_lines.append(f"- 실패: {len(result['failed'])}개 (아래 참고)")

        st.chat_message("assistant").write("\n".join(summary_lines))

        # 실패가 있으면 상세 출력
        if result.get("failed"):
            with st.expander("실패 상세 보기", expanded=False):
                for fn, err in result["failed"][:20]:
                    st.code(f"{fn}\n{err}")

        # 세션 메시지에도 남기기(새로고침/리렌더 대비)
        st.session_state["messages"].append({"role": "assistant", "content": "\n".join(summary_lines)})

    # ====== 참고용: 임베딩된 PDF 목록 ======
    with st.expander("📚 임베딩된 PDF 목록(매니페스트 기준)", expanded=False):
        items = list_ingested_pdfs()
        if not items:
            st.write("아직 임베딩된 PDF가 없습니다.")
        else:
            for it in items[:100]:
                st.write(f"- {it.get('original_name')} | {it.get('ingested_at')} | {it.get('sha256','')[:12]}")

    # ====== 채팅 입력/응답 ======
    if prompt_message := st.chat_input("질문을 입력하세요"):
        st.session_state["messages"].append({"role": "human", "content": prompt_message})
        st.chat_message("human").write(prompt_message)

        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                config = {"configurable": {"session_id": "any"}}
                try:
                    response = conversational_chain.invoke({"input": prompt_message}, config)
                except Exception as e:
                    st.error("질문 처리 중 오류가 발생했습니다.")
                    st.code(str(e))
                    st.info(
                        "Ollama/임베딩 모델 문제일 수 있습니다.\n\n"
                        "1) `ollama list`\n"
                        "2) `ollama pull qwen3-embedding` 또는 `ollama pull nomic-embed-text`\n"
                        "3) `streamlit cache clear` 후 재실행"
                    )
                    return

                answer = response.get("answer", "")
                st.write(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})

                with st.expander("참고 문서 확인"):
                    for doc in response.get("context", []) or []:
                        src = (doc.metadata or {}).get("source", "Unknown")
                        st.markdown(src, help=getattr(doc, "page_content", ""))