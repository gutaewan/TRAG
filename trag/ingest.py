import os
from langchain_community.document_loaders import PyPDFLoader

def load_pdf_pages(pdf_path: str):
    if not os.path.exists(pdf_path):
        return []

    loader = PyPDFLoader(pdf_path)
    # PyPDFLoader는 문서 페이지 단위로 Document 리스트를 반환
    return loader.load()