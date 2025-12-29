import os
import json
import time
import sys
import subprocess
from datetime import datetime

# 프로젝트 루트(= TRAG 폴더) 기준으로 모든 상대경로를 고정하기 위한 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _abs_path(p: str) -> str:
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(PROJECT_ROOT, p))


def _log(msg: str):
    try:
        log_path = _abs_path(NEWS_LOG_PATH) if 'NEWS_LOG_PATH' in globals() else os.path.abspath(os.path.join(PROJECT_ROOT, 'logs/news_daemon.log'))
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")
    except Exception:
        pass


from langchain_core.documents import Document

from .config import (
    NEWS_ENABLED,
    NEWS_KEYWORDS,
    NEWS_POLL_INTERVAL_SEC,
    NEWS_TEXT_DIR,
    NEWS_SENTENCE_PATH,
    NEWS_RSS_HL,
    NEWS_RSS_GL,
    NEWS_RSS_CEID,
    NEWS_MAX_ITEMS_PER_KEYWORD,
    NEWS_DUP_DISTANCE_THRESHOLD,
    NEWS_MANIFEST_PATH,
    NEWS_LOG_PATH,
    NEWS_PID_PATH,
)

from .news_fetcher import fetch_google_news, 대표문장_추출, stable_id
from .vectorstore import get_vectorstore, add_news_documents_to_vectorstore


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_manifest():
    manifest_path = _abs_path(NEWS_MANIFEST_PATH)
    if not os.path.exists(manifest_path):
        return {"version": 1, "items": {}}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "items" not in data:
            data["items"] = {}
        return data
    except Exception:
        return {"version": 1, "items": {}}


def _save_manifest(data):
    manifest_path = _abs_path(NEWS_MANIFEST_PATH)
    _ensure_dir(os.path.dirname(manifest_path))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def _is_similar_already(vectorstore, sentence: str) -> bool:
    """
    유사 뉴스 제외:
    - Chroma에 similarity_search_with_score로 nearest 1개를 찾아
    - distance가 임계값보다 작으면(더 유사) => 중복/유사로 간주하여 스킵
    """
    try:
        results = vectorstore.similarity_search_with_score(sentence, k=1)
        if not results:
            return False
        _, dist = results[0]
        return dist is not None and dist < NEWS_DUP_DISTANCE_THRESHOLD
    except Exception as e:
        _log(f"WARN similarity_check_failed: {e}")
        # 검색 자체가 실패하면 중복 판단을 하지 않고 넣도록(보수적으로) 처리
        return False


def run_once():
    if not NEWS_ENABLED:
        return {"added": 0, "skipped": 0, "errors": 0}

    vs = get_vectorstore()
    manifest = _load_manifest()

    added_docs = []
    added = 0
    skipped = 0
    errors = 0

    for kw in (NEWS_KEYWORDS or []):
        try:
            entries = fetch_google_news(
                keyword=kw,
                hl=NEWS_RSS_HL,
                gl=NEWS_RSS_GL,
                ceid=NEWS_RSS_CEID,
                max_items=NEWS_MAX_ITEMS_PER_KEYWORD,
            )
            _log(f"INFO fetched keyword='{kw}' items={len(entries)}")
        except Exception as e:
            _log(f"ERROR fetch keyword='{kw}': {e}")
            
            errors += 1
            continue

        for e in entries:
            title = e.get("title", "")
            link = e.get("link", "")
            published = e.get("published", "")
            uid = stable_id(title, link )
            if uid in manifest["items"]:
                skipped += 1
                continue

            sentence = 대표문장_추출(e.get("title",""), e.get("summary",""))
            if not sentence:
                skipped += 1
                continue

            # semantic dedup (유사 기사 제외)
            if _is_similar_already(vs, sentence):
                manifest["items"][uid] = {
                    "status": "skipped_similar",
                    "keyword": kw,
                    "title": e.get("title",""),
                    "link": e.get("link",""),
                    "published": e.get("published",""),
                    "seen_at": datetime.now().isoformat(timespec="seconds"),
                }
                skipped += 1
                continue

            # 대표문장.txt 기록
            line = f"[{datetime.now().isoformat(timespec='seconds')}] ({kw}) {sentence} | {e.get('link','')}"


            doc = Document(
                page_content=sentence,
                metadata={
                    "type": "news",
                    "source": "google_news_rss",
                    "keyword": kw,
                    "title": title,
                    "url": link,
                    "published": published,
                    "uid": uid,
                },
            )
            added_docs.append(doc)  
            
            manifest["items"][uid] = {
                "status": "added",
                "keyword": kw,
                "title": e.get("title",""),
                "link": e.get("link",""),
                "published": e.get("published",""),
                "ingested_at": datetime.now().isoformat(timespec="seconds"),
            }

    # ✅ 신규 뉴스가 있을 때만 임베딩 추가
    if added_docs:
        n = add_news_documents_to_vectorstore(added_docs, vs=vs)
        added += n

    _save_manifest(manifest)

    return {"added": added, "skipped": skipped, "errors": errors}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def ensure_daemon_started():
    """
    메인에서 호출하면:
    - 이미 실행 중이면 재시작하지 않음
    - 아니면 백그라운드로 news_daemon --run 을 실행
    """
    _ensure_dir(os.path.dirname(_abs_path(NEWS_PID_PATH)))
    _ensure_dir(os.path.dirname(_abs_path(NEWS_LOG_PATH)))

    pid_path = _abs_path(NEWS_PID_PATH)
    if os.path.exists(pid_path):
        try:
            with open(pid_path, "r", encoding="utf-8") as f:
                pid = int(f.read().strip())
            if pid and _pid_alive(pid):
                return False
        except Exception:
            pass

    # 백그라운드 프로세스 실행 (작업 폴더를 PROJECT_ROOT로 고정)
    cmd = [sys.executable, "-m", "trag.news_daemon", "--run"]
    _log(f"INFO starting news daemon: {' '.join(cmd)} cwd={PROJECT_ROOT}")

    log_path = _abs_path(NEWS_LOG_PATH)
    pid_path = _abs_path(NEWS_PID_PATH)

    # PYTHONPATH를 TRAG 루트로 고정해서 어디서 실행해도 `import trag`가 되게 함
    env = {**os.environ}
    env["PYTHONPATH"] = PROJECT_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["TRAG_NEWS_DAEMON"] = "1"

    _ensure_dir(os.path.dirname(log_path))
    with open(log_path, "a", encoding="utf-8") as logf:
        p = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=logf,
            cwd=PROJECT_ROOT,
            env=env,
        )

    _ensure_dir(os.path.dirname(pid_path))
    with open(pid_path, "w", encoding="utf-8") as f:
        f.write(str(p.pid))

    print(f"[NEWS_DAEMON] started pid={p.pid} (log={log_path})", flush=True)
    _log(f"INFO started pid={p.pid} (log={log_path})")

    return True


def run_loop():
    text_dir = _abs_path(NEWS_TEXT_DIR)
    log_path = _abs_path(NEWS_LOG_PATH)
    pid_path = _abs_path(NEWS_PID_PATH)

    _ensure_dir(text_dir)
    _ensure_dir(os.path.dirname(log_path))
    _ensure_dir(os.path.dirname(pid_path))

    _log(f"INFO daemon loop started pid={os.getpid()} interval={NEWS_POLL_INTERVAL_SEC}s keywords={len(NEWS_KEYWORDS or [])}")

    # pid 기록(직접 실행 시)
    try:
        with open(pid_path, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    tick = 0
    while True:
        tick += 1
        started_at = datetime.now()
        try:
            res = run_once()

            # 로그 파일 기록(기존 유지)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat(timespec='seconds')}] run_once: {res}\n")

            # ✅ 주기적으로 동작하고 있음을 stdout에도 출력(백그라운드 실행 시 로그로 리다이렉트됨)
            next_in = max(30, int(NEWS_POLL_INTERVAL_SEC))
            msg = (
                f"[NEWS_DAEMON] tick={tick} at={started_at.isoformat(timespec='seconds')} "
                f"added={res.get('added')} skipped={res.get('skipped')} errors={res.get('errors')} "
                f"next_in={next_in}s"
            )
            print(msg, flush=True)
            _log(msg)

        except Exception as e:
            err_msg = f"[NEWS_DAEMON] tick={tick} ERROR: {e}"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat(timespec='seconds')}] ERROR: {e}\n")
            print(err_msg, flush=True)
            _log(err_msg)

        time.sleep(max(30, int(NEWS_POLL_INTERVAL_SEC)))


if __name__ == "__main__":
    if "--run" in sys.argv:
        run_loop()


def _write_news_item_file(uid: str, kw: str, sentence: str, link: str, title: str, published: str) -> str:
    """
    뉴스 1건을 개별 txt 파일로 저장하고, 저장된 파일의 절대경로를 반환합니다.
    """
    text_dir = _abs_path(NEWS_TEXT_DIR)
    _ensure_dir(text_dir)

    # 파일명은 안정적인 uid 기반 (중복 저장 방지)
    fname = f"news_{uid}.txt"
    fpath = os.path.join(text_dir, fname)

    # 이미 존재하면 덮어쓰지 않음(= 신규만 저장)
    if os.path.exists(fpath):
        return fpath

    payload = []
    payload.append(f"keyword: {kw}")
    payload.append(f"title: {title}")
    payload.append(f"published: {published}")
    payload.append(f"url: {link}")
    payload.append("")
    payload.append(sentence.strip())
    payload.append("")  # newline

    with open(fpath, "w", encoding="utf-8") as f:
        f.write("\n".join(payload))

    return fpath