전체 구조 개요

이 프로젝트는 크게 3개의 흐름이 동시에 돌아갑니다.
	1.	Streamlit UI 앱 (BaseRag_v02.py)
	•	사용자가 질문을 입력하고 답변을 받는 화면(웹앱)
	•	앱 시작 시 뉴스 데몬을 띄우고, RAG 체인을 구성해서 채팅을 렌더링합니다.
	2.	RAG 파이프라인 (trag/rag.py, trag/vectorstore.py 등)
	•	PDF/텍스트들을 읽어서 임베딩 → Chroma에 저장
	•	질의가 들어오면 retriever로 관련 문서를 찾아 LLM에 넣어 답변 생성
	3.	뉴스 수집 + 임베딩 데몬 (trag/news_daemon.py, trag/news_fetcher.py)
	•	주기적으로 뉴스 RSS를 가져와서 텍스트 파일로 저장
	•	신규만 임베딩(중복/유사 제외)하여 같은 Chroma에 누적

즉, UI(질문/응답)와 백그라운드(뉴스 수집/임베딩)가 분리되어 있고, 둘 다 **같은 벡터DB(Chroma)**를 업데이트/조회할 수 있는 구조입니다.

⸻

1) BaseRag_v02.py — Streamlit “메인 엔트리(실행 파일)”

역할
	•	Streamlit 앱의 시작점입니다.
	•	앱이 켜질 때 뉴스 데몬을 백그라운드로 실행(이미 떠 있으면 스킵)합니다.
	•	UI에서 모델 선택 → RAG 체인 생성 → 채팅 UI 렌더링을 수행합니다.

핵심 코드 흐름
	1.	뉴스 데몬 시작 시도
        from trag.news_daemon import ensure_daemon_started
        ensure_daemon_started()
    2.	페이지 설정
        st.set_page_config(page_title="TRAG", layout="wide")
    3.	설정/체인 로딩
        from trag.config import UI_TITLE, AVAILABLE_LLM_MODELS
        from trag.rag import build_conversational_rag_chain
    4.	UI 렌더링
        selected_model = st.selectbox(...)
        chain = build_conversational_rag_chain(selected_model)
        render_chat(chain)


입력/출력
	•	입력: 사용자 질문(채팅), 모델 선택
	•	출력: 답변(LLM 생성), 참고 문서(컨텍스트/출처)

자주 발생하는 문제 포인트
	•	build_conversational_rag_chain()가 느릴 수 있음(초기 임베딩/로딩)
	•	chain is None이면 config에서 PDF 경로나 Chroma/임베딩 설정 문제 가능

⸻

2) trag/config.py — 모든 설정(“한 곳에서 조절하는 변수들”)

역할
	•	프로젝트의 단일 설정 소스입니다.
	•	UI 제목, 사용 가능한 LLM 모델 목록, PDF 폴더 경로, Chroma 저장 경로, 임베딩 모델명, 뉴스 키워드/주기, 로그/매니페스트 경로 등을 정의합니다.

여기에 들어있어야 하는 대표 항목(예)
	•	UI
	•	UI_TITLE
	•	AVAILABLE_LLM_MODELS
	•	벡터스토어/임베딩
	•	CHROMA_PATH (또는 PDF/NEWS 공용/분리 여부)
	•	EMBEDDING_MODEL (Ollama 임베딩 모델명)
	•	데이터 폴더
	•	DATA_DIR = "./data" (PDF들이 들어있는 폴더)
	•	NEWS_TEXT_DIR = "./data_news" (뉴스 txt 저장 폴더)
	•	뉴스 데몬
	•	NEWS_ENABLED
	•	NEWS_KEYWORDS (사용자가 키워드 추가)
	•	NEWS_POLL_INTERVAL_SEC (10분 = 600)
	•	NEWS_SENTENCE_PATH (합본 파일 경로)
	•	NEWS_MANIFEST_PATH (중복/처리 상태 기록)
	•	NEWS_LOG_PATH, NEWS_PID_PATH

운영 팁
	•	“처음 실행이 느리다” → chunk size, 중복 체크 방식, 최초 임베딩 대상 범위를 config에서 줄이는 게 가장 효과적입니다.
	•	뉴스 키워드/주기는 config에서 바꾸고 데몬을 재시작하면 반영됩니다.

⸻

3) trag/rag.py — RAG 체인 구성(“검색 + 프롬프트 + LLM”)

역할
	•	대화형 RAG 체인을 빌드하는 상위 레벨 모듈입니다.
	•	대개 다음을 조합합니다:
	1.	벡터스토어 로드/빌드
	2.	retriever 구성
	3.	프롬프트 템플릿(시스템/히스토리/컨텍스트) 구성
	4.	ChatOllama 모델 연결
	5.	RunnableWithMessageHistory로 대화 히스토리 연동

주요 함수(예상/전형)
	•	build_conversational_rag_chain(selected_model: str) -> Runnable | None
	•	selected_model을 받아서 ChatOllama를 만들고
	•	vectorstore retriever를 붙여서
	•	최종적으로 UI가 호출할 “invoke 가능한 체인”을 반환

입력/출력
	•	입력: 사용자가 선택한 LLM 모델명(예: llama3.2)
	•	출력: chain.invoke({"input": 질문}, config) 형태로 호출 가능한 객체

문제 포인트
	•	초기 체인 구성 시 벡터스토어 로딩/빌드(특히 PDF 임베딩)가 포함되면 느려집니다.
	•	“히스토리 aware retriever”가 버전마다 import 경로가 달라서 호환 처리 필요(이미 과거 코드에서 처리했었음)

⸻

4) trag/vectorstore.py — Chroma 벡터DB 관리(“저장/추가/증분 임베딩”)

역할
	•	Chroma의 생성/로드를 전담합니다.
	•	“./data 폴더의 PDF 전부 임베딩, 신규만 추가” 같은 정책이 여기서 구현되는 게 가장 좋습니다.
	•	뉴스도 결국 Document로 만들어 “추가(add)”만 수행하면 되므로 이 파일이 중심이 됩니다.

흔히 포함되는 기능
	•	get_vectorstore()
	•	persist 디렉터리에 DB가 있으면 로드
	•	없으면 새로 생성
	•	load_or_build_vectorstore()
	•	데이터 디렉터리를 스캔해서 파일 목록/해시를 manifest로 관리
	•	새로 추가된 파일만 로더 → split → embed → upsert
	•	add_news_documents_to_vectorstore(docs, vs=None)
	•	뉴스 문서 리스트를 받아서 Chroma에 추가(신규만)
	•	반환: 실제 추가된 문서 수

입력/출력
	•	입력: 파일 경로/Document 리스트
	•	출력: vectorstore 객체, 추가된 문서 수

문제 포인트
	•	“임베딩 모델을 바꾸면 기존 DB와 호환 안됨” → Chroma 경로를 분리하거나 재생성 필요
	•	너무 큰 PDF/엑셀 → chunk/전처리 정책이 중요(속도/품질)

⸻

5) trag/ui.py — Streamlit 채팅 UI 렌더링

역할
	•	Streamlit의 st.chat_input, st.chat_message 등을 이용해서 채팅 UX를 담당합니다.
	•	체인을 받아서 invoke하고 결과를 화면에 표시합니다.
	•	문서 근거(컨텍스트)를 expander로 보여주는 기능도 여기서 처리합니다.

흔한 함수
	•	render_chat(chain)
	•	세션 히스토리 관리
	•	사용자 입력 받기
	•	spinner로 처리 중 표시
	•	response를 받아 답변 출력
	•	참고 문서/출처 출력

문제 포인트
	•	체인의 output key(answer, context, etc.)가 바뀌면 UI도 같이 맞춰야 함
	•	“브라우저에서 아무것도 안 뜸”은 보통:
	•	앱이 에러로 죽었는데 Streamlit에 표시가 안 됨(try/except로 감싸기)
	•	또는 render_chat이 호출되기 전에 chain이 None

⸻

6) trag/news_fetcher.py — 뉴스 수집 + 텍스트 정리 유틸

역할
	•	Google News RSS(또는 다른 RSS/API)를 사용해 뉴스 엔트리를 가져옵니다.
	•	엔트리에서 대표문장/요약을 만들어줍니다.
	•	중복 방지를 위한 stable_id() 같은 키 생성 함수가 있습니다.

주요 함수(대화에서 확인된 것)
	•	fetch_google_news(keyword, hl, gl, ceid, max_items)
	•	RSS URL을 만들고 requests로 가져온 뒤 feedparser로 파싱
	•	반환: entry 리스트(dict)
	•	대표문장_추출(title, summary_html)
	•	HTML 제거 후 첫 문장 추출
	•	여기서 정규식 look-behind 문제가 났었음
	•	stable_id(title, link)
	•	기사 식별자 생성(보통 hash)

문제 포인트(핵심)
	•	Python re의 look-behind 제약: (?<=A|BC) 같은 가변 길이는 실패
	•	RSS는 종종 User-Agent 없으면 빈 응답/차단 → headers 지정 권장

⸻

7) trag/news_daemon.py — 뉴스 데몬(주기 실행 + 신규만 임베딩)

역할
	•	설정된 주기(NEWS_POLL_INTERVAL_SEC)마다 실행되는 백그라운드 프로세스입니다.
	•	매 회차마다:
	1.	키워드별 뉴스 fetch
	2.	신규 기사인지 확인(manifest)
	3.	(선택) 유사 기사 제외(semantic dedup)
	4.	파일 저장(기사별 개별 파일 또는 합본 파일)
	5.	신규만 Chroma에 추가 임베딩
	6.	로그 출력

주요 함수(대화에서 등장)
	•	run_once()
	•	한 번의 수집/저장/임베딩 사이클
	•	run_loop()
	•	무한 루프(주기적으로 run_once 호출)
	•	ensure_daemon_started()
	•	PID 파일/프로세스 체크 후 이미 떠 있으면 스킵
	•	아니면 python -m trag.news_daemon --run 형태로 백그라운드 실행

최근 발생했던 대표 이슈들
	•	FileNotFoundError
	•	합본 txt 경로의 디렉터리를 만들지 않고 open(“a”)해서 실패
	•	NameError: title is not defined
	•	title = e.get("title")를 선언하기 전에 metadata에서 title 사용
	•	look-behind 문제
	•	대표문장_추출() 구현이 여전히 구버전으로 남아있던 상태(실제 실행 파일 확인으로 추적)

⸻

8) (추정) trag/rag 관련 보조 파일들

프로젝트가 RAG로 나뉘어 있을 때 자주 생기는 보조 모듈도 설명해두면 좋습니다.

trag/loaders.py (있다면)
	•	PDFLoader, TextLoader, ExcelLoader 등 파일 타입별 로딩 전략을 모아둡니다.
	•	“./data 폴더에 있는 모든 파일 임베딩” 정책을 깔끔히 분리할 수 있습니다.

trag/utils.py (있다면)
	•	해시 계산, 경로 정규화, 파일 변경 탐지, 로그 유틸 등을 둡니다.

trag/manifest.py (있다면)
	•	“이미 처리한 파일 목록/해시/mtime”을 json으로 저장하고,
	•	신규만 임베딩하는 로직을 담당합니다.