import os
import streamlit as st
import chromadb
import langchain

# [변경] 최신 패키지 임포트 경로 적용
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# [변경] Ollama 관련 임포트
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

# LangChain 버전에 따라 import 경로가 달라질 수 있어 호환 처리
try:
    from langchain.chains import create_history_aware_retriever
except ImportError:
    try:
        from langchain.chains.history_aware_retriever import create_history_aware_retriever
    except ImportError:
        create_history_aware_retriever = None

# ChromaDB Tenant 오류 방지 (Streamlit 리로드 시 필수)
chromadb.api.client.SharedSystemClient.clear_system_cache()

# [설정] 임베딩 모델과 벡터 DB 저장 경로 설정
# 주의: OpenAI와 Ollama 임베딩은 호환되지 않으므로 경로를 분리했습니다.
#CHROMA_PATH = "./chroma_db_ollama"
#EMBEDDING_MODEL = "nomic-embed-text" # Ollama용 고성능 임베딩 모델

EMBEDDING_MODEL = "qwen2.5-embedding" # Ollama용 QWEN 임베딩 모델
CHROMA_PATH = "./chroma_db_ollama_{EMBEDDING_MODEL}" 

# cache_resource로 한번 실행한 결과 캐싱해두기
@st.cache_resource
def load_and_split_pdf(file_path):
    # 파일이 존재하는지 확인
    if not os.path.exists(file_path):
        st.error(f"파일을 찾을 수 없습니다: {file_path}")
        return []
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# 텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    if not _docs:
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(_docs)
    
    # [변경] OllamaEmbeddings 사용
    vectorstore = Chroma.from_documents(
        split_docs, 
        OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_PATH
    )
    return vectorstore

# 만약 기존에 저장해둔 ChromaDB가 있는 경우, 이를 로드
@st.cache_resource
def get_vectorstore(_docs):
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL)
        )
    else:
        return create_vector_store(_docs)
    
# PDF 문서 로드-벡터 DB 저장-검색기-히스토리 모두 합친 Chain 구축
@st.cache_resource
def initialize_components(selected_model):
    # [주의] 실제 파일 경로에 맞게 수정해주세요.
    file_path = r"./data/대한민국헌법(헌법)(제00010호)(19880225).pdf"
    
    pages = load_and_split_pdf(file_path)
    if not pages:
        return None
        
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # 채팅 히스토리 요약 시스템 프롬프트
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 질문-답변 시스템 프롬프트
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. please use imogi with the answer.
    대답은 한국어로 하고, 존댓말을 써줘.\

    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # [변경] ChatOllama 사용
    llm = ChatOllama(model=selected_model)
    
    # 히스토리 기반 질의 재구성 Retriever (버전에 없으면 일반 retriever 사용)
    # ⚠️ 일반 retriever는 "문자열 질문"만 받아야 합니다. (dict가 들어가면 EmbedRequest ValidationError 발생)
    if create_history_aware_retriever is not None:
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    else:
        # RunnableWithMessageHistory가 넘기는 입력은 {"input": 질문, "history": ...} 형태이므로
        # retriever에는 질문 문자열만 전달하도록 변환합니다.
        history_aware_retriever = RunnableLambda(lambda x: x["input"]) | retriever

    def _format_docs(docs):
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in (docs or []))

    # LangChain 버전에 따라 create_stuff_documents_chain API가 없을 수 있어
    # 동일한 동작(문서들을 한 덩어리로 'stuff'하여 프롬프트에 넣기)을 Runnable로 직접 구성
    rag_chain = (
        RunnablePassthrough
        .assign(context_docs=history_aware_retriever)
        .assign(context=RunnableLambda(lambda x: _format_docs(x["context_docs"])))
        .assign(
            answer=(
                qa_prompt
                | llm
                | RunnableLambda(lambda m: getattr(m, "content", str(m)))
            )
        )
        | RunnableLambda(lambda x: {"answer": x["answer"], "context": x["context_docs"]})
    )

    return rag_chain

# Streamlit UI
st.header("Taewan's RAG 챗봇 (Ollama Ver) 💬 📚")

# [변경] Ollama 모델 선택지로 변경
option = st.selectbox("Select Ollama Model", ("llama3.2", "mistral", "gemma2"))

# 체인 초기화
rag_chain = initialize_components(option)

chat_history = StreamlitChatMessageHistory(key="chat_messages")

if rag_chain:
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", 
                                         "content": "헌법에 대해 무엇이든 물어보세요!"}]

    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt_message := st.chat_input("Your question"):
        st.chat_message("human").write(prompt_message)
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                config = {"configurable": {"session_id": "any"}}
                response = conversational_rag_chain.invoke(
                    {"input": prompt_message},
                    config)
                
                answer = response['answer']
                st.write(answer)
                with st.expander("참고 문서 확인"):
                    for doc in response['context']:
                        st.markdown(doc.metadata.get('source', 'Unknown'), help=doc.page_content)
else:
    st.error("PDF 파일을 로드하는 데 실패했습니다. 경로를 확인해주세요.")