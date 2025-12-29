import os
import chromadb

# Streamlit hot-reload에서 Chroma 공유 클라이언트 꼬임 방지
chromadb.api.client.SharedSystemClient.clear_system_cache()

# --- Data directory (Single source of truth) ---
DATA_DIR = r"./data"  # ✅ ./data 폴더 내 PDF 전체를 임베딩 대상으로 사용

# (선택) 기본 PDF 경로가 필요하면 유지
#PDF_PATH = os.path.join(DATA_DIR, "대한민국헌법(헌법)(제00010호)(19880225).pdf")

# 업로드 PDF 저장 폴더도 DATA_DIR과 동일하게 고정
UPLOAD_DIR = DATA_DIR

# --- Embedding / Chroma ---
EMBEDDING_MODEL = "qwen3-embedding"
FALLBACK_EMBEDDING_MODEL = "nomic-embed-text"

CHROMA_PATH = f"./chroma_db_ollama_{EMBEDDING_MODEL}"
COLLECTION_NAME = "rag_collection"

# 임베딩 완료된 PDF를 기록(새 파일만 추가 임베딩하기 위함)
MANIFEST_PATH = os.path.join(CHROMA_PATH, "ingested_manifest.json")

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# --- Retriever ---
TOP_K = 4

# --- UI ---
UI_TITLE = "TG RAG 챗봇 (Ollama Ver) 💬 📚"
AVAILABLE_LLM_MODELS = ("llama3.2", "mistral", "gemma2")

# =========================
# News ingestion (RSS)
# =========================
NEWS_ENABLED = True

# 사용자가 추가/수정 가능한 뉴스 검색 키워드
NEWS_KEYWORDS = [
    "소프트웨어 공학",
    "AI 안전",
    "자동차 기능안전",
]

# 검색 주기(초) - 기본 10분
NEWS_POLL_INTERVAL_SEC = 600

# 뉴스 대표문장 저장 폴더 (PDF ./data 와 분리)
NEWS_TEXT_DIR = r"./news_texts"
NEWS_SENTENCE_FILENAME = "./Representative.txt"
NEWS_SENTENCE_PATH = os.path.join(NEWS_TEXT_DIR, NEWS_SENTENCE_FILENAME)

# Google News RSS 지역/언어(한국)
NEWS_RSS_HL = "ko"
NEWS_RSS_GL = "KR"
NEWS_RSS_CEID = "KR:ko"

# 키워드당 가져올 최대 기사 수
NEWS_MAX_ITEMS_PER_KEYWORD = 20

# 유사 뉴스 제외 임계값(Chroma distance 기준, 작을수록 더 엄격)
# 보통 0.10~0.25 사이에서 튜닝합니다.
NEWS_DUP_DISTANCE_THRESHOLD = 0.15

# 데몬/로그/매니페스트
NEWS_MANIFEST_PATH = os.path.join(CHROMA_PATH, "news_manifest.json")
NEWS_LOG_PATH = r"./logs/news_daemon.log"
NEWS_PID_PATH = r"./run/news_daemon.pid"