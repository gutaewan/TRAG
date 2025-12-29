import os
import json
import glob
import hashlib
from datetime import datetime
from typing import Dict, Any, Tuple, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain_core.documents import Document

from .config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    MANIFEST_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> Dict[str, Any]:
    if not os.path.exists(MANIFEST_PATH):
        return {"version": 1, "items": {}}
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "items" not in data:
            data["items"] = {}
        return data
    except Exception:
        return {"version": 1, "items": {}}


def _save_manifest(data: Dict[str, Any]) -> None:
    _ensure_dir(CHROMA_PATH)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_embedding_function() -> OllamaEmbeddings:
    """Ollama 임베딩 모델을 반환하되, 없으면 fallback으로 자동 전환."""
    try:
        emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
        emb.embed_query("ping")  # 모델 없으면 여기서 실패
        return emb
    except Exception:
        emb_fb = OllamaEmbeddings(model=FALLBACK_EMBEDDING_MODEL)
        emb_fb.embed_query("ping")
        return emb_fb


def get_vectorstore() -> Chroma:
    _ensure_dir(CHROMA_PATH)
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
        collection_name=COLLECTION_NAME,
    )


def _split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def ingest_pdf_path_if_new(pdf_path: str, vs: Chroma = None, manifest: Dict[str, Any] = None) -> Tuple[bool, str]:
    """단일 PDF를 새 파일일 때만 임베딩."""
    if not pdf_path or not os.path.exists(pdf_path):
        return False, ""

    sha = _sha256_file(pdf_path)

    if manifest is None:
        manifest = _load_manifest()
    if sha in manifest["items"]:
        return False, sha

    if vs is None:
        vs = get_vectorstore()

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    split_docs = _split_docs(docs)

    base = os.path.basename(pdf_path)
    abs_path = os.path.abspath(pdf_path)

    for d in split_docs:
        d.metadata = dict(d.metadata or {})
        d.metadata.update({"source": base, "path": abs_path, "sha256": sha})

    vs.add_documents(split_docs)

    manifest["items"][sha] = {
        "original_name": base,
        "stored_path": abs_path,
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
    }

    return True, sha


def sync_pdf_dir(data_dir: str) -> Dict[str, Any]:
    """✅ 폴더 내 PDF 전체를 스캔하여 '신규 PDF만' 추가 임베딩."""
    _ensure_dir(data_dir)
    vs = get_vectorstore()
    manifest = _load_manifest()

    pdf_paths = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    added: List[str] = []
    skipped: List[str] = []
    failed: List[Tuple[str, str]] = []

    for p in pdf_paths:
        try:
            ingested, sha = ingest_pdf_path_if_new(p, vs=vs, manifest=manifest)
            if ingested:
                added.append(os.path.basename(p))
            else:
                skipped.append(os.path.basename(p))
        except Exception as e:
            failed.append((os.path.basename(p), str(e)))

    # persist (가능한 경우)
    try:
        client = getattr(vs, "_client", None)
        if client is not None and hasattr(client, "persist"):
            client.persist()
    except Exception:
        pass

    _save_manifest(manifest)

    return {
        "data_dir": os.path.abspath(data_dir),
        "total_pdf": len(pdf_paths),
        "added": added,
        "skipped": skipped,
        "failed": failed,
    }


def save_uploaded_pdf_to_dir(uploaded_file, target_dir: str) -> str:
    """업로드 파일을 지정 폴더에 저장(파일명 충돌 시 자동 suffix)."""
    _ensure_dir(target_dir)
    name = os.path.basename(getattr(uploaded_file, "name", "uploaded.pdf"))
    base, ext = os.path.splitext(name)
    ext = ext or ".pdf"

    out = os.path.join(target_dir, name)
    idx = 1
    while os.path.exists(out):
        out = os.path.join(target_dir, f"{base}_{idx}{ext}")
        idx += 1

    with open(out, "wb") as f:
        f.write(uploaded_file.getvalue())

    return os.path.abspath(out)


def list_ingested_pdfs() -> List[Dict[str, Any]]:
    manifest = _load_manifest()
    items = []
    for sha, info in manifest.get("items", {}).items():
        items.append({"sha256": sha, **info})
    items.sort(key=lambda x: x.get("ingested_at", ""), reverse=True)
    return items

def add_news_documents_to_vectorstore(news_docs, vs=None):
    """
    news_docs: List[Document]
    """
    if not news_docs:
        return 0

    if vs is None:
        vs = get_vectorstore()

    vs.add_documents(news_docs)

    # persist 가능한 경우 마지막에 1번만
    try:
        client = getattr(vs, "_client", None)
        if client is not None and hasattr(client, "persist"):
            client.persist()
    except Exception:
        pass

    return len(news_docs)