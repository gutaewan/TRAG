import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from .config import TOP_K
from .vectorstore import get_vectorstore


def _format_docs(docs):
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in (docs or []))


@st.cache_resource
def _build_rag_chain(selected_model: str):
    # ✅ 여기서는 절대 sync/임베딩/폴더스캔을 하지 않습니다.
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved context to answer the question. "
        "If you don't know, say you don't know. "
        "대답은 한국어로 존댓말로 해주세요. 이모지를 적절히 사용해 주세요.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatOllama(model=selected_model)

    retriever_runnable = RunnableLambda(lambda x: x["input"]) | retriever

    rag_chain = (
        RunnablePassthrough
        .assign(context_docs=retriever_runnable)
        .assign(context=RunnableLambda(lambda x: _format_docs(x["context_docs"])))
        .assign(answer=(qa_prompt | llm | RunnableLambda(lambda m: getattr(m, "content", str(m)))))
        | RunnableLambda(lambda x: {"answer": x["answer"], "context": x["context_docs"]})
    )
    return rag_chain


def build_conversational_rag_chain(selected_model: str):
    rag_chain = _build_rag_chain(selected_model)

    chat_history = StreamlitChatMessageHistory(key="chat_messages")

    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
    )