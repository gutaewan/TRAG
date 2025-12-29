import re
import time
import hashlib
import requests
import feedparser
from urllib.parse import quote_plus
from typing import List, Dict, Any


def google_news_rss_url(query: str, hl: str, gl: str, ceid: str) -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"


def fetch_google_news(keyword: str, hl: str, gl: str, ceid: str, max_items: int = 20) -> List[Dict[str, Any]]:
    url = google_news_rss_url(keyword, hl, gl, ceid)
    # Google RSS는 가끔 느릴 수 있어 timeout 지정
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    
    print(f"Fetched RSS for keyword='{keyword}' with status {r.status_code}")

    feed = feedparser.parse(r.text)
    items = []
    for e in feed.entries[:max_items]:
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()
        published = getattr(e, "published", "") or getattr(e, "updated", "")
        summary = getattr(e, "summary", "") or getattr(e, "description", "")

        items.append({
            "keyword": keyword,
            "title": title,
            "link": link,
            "published": published,
            "summary": summary,
        })
    return items


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def 대표문장_추출(title: str, summary_html: str) -> str:
    """
    대표문장 전략(간단/견고):
    - summary에서 HTML 제거 후
    - 첫 문장(마침표/다/요/!/? 기준)을 뽑되
    - 너무 짧으면 title을 보강
    """
    summary = _strip_html(summary_html)
    if not summary:
        return title.strip()

    # 문장 후보
    # Python re는 가변 길이 look-behind를 지원하지 않으므로 look-behind 없이 첫 문장을 추출합니다.
    # 규칙: [.!?] 또는 '다.'(문장 종결)까지를 첫 문장으로 간주
    m = re.search(r"[.!?](?:\s+|$)|다\.(?:\s+|$)", summary)
    if m:
        first = summary[: m.end()].strip()
    else:
        first = summary.strip()

    if len(first) < 25:
        # 요약 첫 문장이 너무 짧으면 제목+요약 조합
        merged = f"{title.strip()} - {first}"
        return merged.strip()

    return first


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def stable_id(title: str, link: str) -> str:
    base = normalize_text(title) + "|" + (link or "").strip()
    return hashlib.sha256(base.encode("utf-8")).hexdigest()