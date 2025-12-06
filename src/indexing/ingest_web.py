from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urldefrag, urlparse
import hashlib
import json
import time

import requests

from src.config.settings import Settings
from src.utils.text import normalize_pdf_text  # reuse robust cleaner
from src.utils.logger import get_logger


@dataclass(frozen=True)
class WebDoc:
    doc_id: str
    url: str
    title: str
    text: str
    meta: Dict[str, Any]


def _is_allowed(url: str, allowed_domain: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        return host == allowed_domain or host.endswith("." + allowed_domain)
    except Exception:
        return False


def _canonicalize(url: str) -> str:
    # remove fragments (#section) and normalize
    u, _frag = urldefrag(url)
    return u.strip()


def _hash_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def _cache_paths(cache_dir: Path, url: str) -> Tuple[Path, Path]:
    h = _hash_url(url)
    html_path = cache_dir / f"{h}.html"
    meta_path = cache_dir / f"{h}.json"
    return html_path, meta_path


def _load_cached(cache_dir: Path, url: str) -> Optional[WebDoc]:
    html_path, meta_path = _cache_paths(cache_dir, url)
    if not html_path.exists() or not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        text, title = _extract_main_text(html, url)
        text = normalize_pdf_text(text)
        return WebDoc(
            doc_id=meta["doc_id"],
            url=meta["url"],
            title=title or meta.get("title", meta["url"]),
            text=text,
            meta=meta,
        )
    except Exception:
        return None


def _save_cache(cache_dir: Path, url: str, html: str, title: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    html_path, meta_path = _cache_paths(cache_dir, url)

    doc_id = f"web::{_hash_url(url)[:12]}"
    meta = {
        "doc_id": doc_id,
        "url": url,
        "title": title,
        "source": "web",
    }
    html_path.write_text(html, encoding="utf-8", errors="ignore")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _robots_allows(session: requests.Session, base_url: str, target_url: str, timeout: float = 10.0) -> bool:
    """
    Basic robots.txt handling:
    - Fetch robots.txt at scheme://host/robots.txt
    - If cannot fetch, allow
    - Only checks Disallow for User-agent: *
    This is intentionally simple; good enough for a student project pipeline.
    """
    try:
        parsed = urlparse(base_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        r = session.get(robots_url, timeout=timeout)
        if r.status_code != 200:
            return True

        lines = [ln.strip() for ln in r.text.splitlines()]
        ua_star = False
        disallow: List[str] = []

        for ln in lines:
            if not ln or ln.startswith("#"):
                continue
            low = ln.lower()
            if low.startswith("user-agent:"):
                ua = ln.split(":", 1)[1].strip()
                ua_star = (ua == "*")
            elif ua_star and low.startswith("disallow:"):
                path = ln.split(":", 1)[1].strip()
                if path:
                    disallow.append(path)

        path = urlparse(target_url).path or "/"
        for rule in disallow:
            if rule == "/":
                return False
            if path.startswith(rule):
                return False
        return True
    except Exception:
        return True


def _extract_main_text(html: str, url: str) -> Tuple[str, str]:
    """
    Prefer trafilatura for main-text extraction; fall back to BeautifulSoup.
    Returns (text, title).
    """
    title = ""

    # 1) Try trafilatura
    try:
        import trafilatura  # type: ignore

        downloaded = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            include_images=False,
            output_format="txt",
        )
        # trafilatura may return None if it can't extract
        if downloaded and downloaded.strip():
            title = _extract_title_bs4(html) or url
            return downloaded, title
    except Exception:
        pass

    # 2) Fallback: BeautifulSoup
    text = _fallback_bs4_text(html)
    title = _extract_title_bs4(html) or url
    return text, title


def _extract_title_bs4(html: str) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.text:
            return soup.title.text.strip()
    except Exception:
        pass
    return ""


def _fallback_bs4_text(html: str) -> str:
    """
    Conservative extraction: removes nav/script/style and returns visible text.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        # Keep paragraphs-like text
        text = soup.get_text(separator="\n")
        return text
    except Exception:
        # last resort
        return html


def _extract_links(html: str, base_url: str) -> List[str]:
    """
    Extract <a href="..."> links (same-domain filtering happens later).
    """
    out: List[str] = []
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a.get("href") or ""
            href = href.strip()
            if not href:
                continue
            abs_url = urljoin(base_url, href)
            abs_url = _canonicalize(abs_url)
            out.append(abs_url)
    except Exception:
        return out
    return out


def ingest_web(cfg: Optional[Settings] = None) -> List[WebDoc]:
    """
    Crawl from cfg.web.seeds, cache pages, extract main text,
    return WebDoc list for downstream chunking/embedding/upsert.
    """
    cfg = cfg or Settings.load()
    logger = get_logger()

    if not cfg.web.enabled:
    # keep behavior explicit: do nothing unless enabled
        logger.info("Web ingestion disabled (cfg.web.enabled=false).")
        return []

    cache_dir = Path(cfg.web_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    seeds = cfg.web.seeds
    allowed_domain = cfg.web.allowed_domain
    max_pages = int(cfg.web.max_pages)
    rate_limit = float(cfg.web.rate_limit_seconds)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "ConstructorCapstoneBot/1.0 (+https://constructor.university/)",
            "Accept": "text/html,application/xhtml+xml",
        }
    )

    queue: List[str] = []
    seen: Set[str] = set()
    docs: List[WebDoc] = []

    for s in seeds:
        u = _canonicalize(s)
        if _is_allowed(u, allowed_domain):
            queue.append(u)
            seen.add(u)

    while queue and len(docs) < max_pages:
        url = queue.pop(0)

        if not _robots_allows(session, url, url):
            logger.info(f"[web] robots blocked: {url}")
            continue

        cached = _load_cached(cache_dir, url)
        if cached:
            docs.append(cached)
            continue

        # Rate limit
        time.sleep(rate_limit)

        try:
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                logger.info(f"[web] skip {url} status={r.status_code}")
                continue

            html = r.text or ""
            text, title = _extract_main_text(html, url)
            text = normalize_pdf_text(text)

            _save_cache(cache_dir, url, html, title)

            doc_id = f"web::{_hash_url(url)[:12]}"
            docs.append(
                WebDoc(
                    doc_id=doc_id,
                    url=url,
                    title=title or url,
                    text=text,
                    meta={"source": "web", "url": url, "title": title or url},
                )
            )

            # Discover more links
            for link in _extract_links(html, url):
                if len(seen) >= max_pages * 8:
                    # cap link frontier growth
                    break
                if not _is_allowed(link, allowed_domain):
                    continue
                if link in seen:
                    continue
                seen.add(link)
                queue.append(link)

        except Exception as e:
            logger.info(f"[web] error {url}: {e}")
            continue

    logger.info(f"[web] docs fetched: {len(docs)} (max_pages={max_pages})")
    return docs
