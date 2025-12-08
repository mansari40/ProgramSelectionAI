# src/indexing/ingest_web.py
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


# ----------------------------
# URL + filtering helpers
# ----------------------------

_BLOCKED_HOSTS: Set[str] = {
    # Causes SSL cert verification errors in many environments; not needed for applicant guidance.
    "teamwork.constructor.university",
}

# Schemes we never want to crawl
_BLOCKED_SCHEMES = ("mailto:", "tel:", "javascript:")

# File extensions we never want to download as "web pages"
_BLOCKED_EXTS = (
    ".pdf",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".mp4", ".mov", ".avi", ".mkv",
    ".mp3", ".wav",
    ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
)


def _canonicalize(url: str) -> str:
    """Remove fragments and strip whitespace."""
    if not url:
        return ""
    u, _ = urldefrag(url.strip())
    return u.strip()


def _is_allowed_domain(url: str, allowed_domain: str) -> bool:
    """
    Allow exact domain or any subdomain of allowed_domain.
    """
    try:
        host = (urlparse(url).netloc or "").lower()
        return host == allowed_domain or host.endswith("." + allowed_domain)
    except Exception:
        return False


def _has_blocked_ext(url: str) -> bool:
    p = urlparse(url).path.lower()
    return any(p.endswith(ext) for ext in _BLOCKED_EXTS)


def _allow_url(
    url: str,
    allowed_domain: str,
    allow_paths: Optional[List[str]] = None,
    deny_paths: Optional[List[str]] = None,
) -> bool:
    """
    Final gate before crawling a URL.
    - Must be http(s)
    - Must be allowed domain
    - Must not be blocked host
    - Must not look like a binary file by extension
    - Optional allow/deny by path prefix
    """
    u = _canonicalize(url)
    if not u:
        return False

    low = u.lower()
    if any(low.startswith(s) for s in _BLOCKED_SCHEMES):
        return False

    parsed = urlparse(u)
    if parsed.scheme not in ("http", "https"):
        return False

    host = (parsed.netloc or "").lower()
    if host in _BLOCKED_HOSTS:
        return False

    if not _is_allowed_domain(u, allowed_domain):
        return False

    if _has_blocked_ext(u):
        return False

    path = parsed.path or "/"

    if deny_paths:
        for dp in deny_paths:
            if dp and path.startswith(dp):
                return False

    if allow_paths:
        # allow only if it starts with one of allow_paths
        ok = any(path.startswith(ap) for ap in allow_paths if ap)
        if not ok:
            return False

    return True


def _hash_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def _cache_paths(cache_dir: Path, url: str) -> Tuple[Path, Path]:
    h = _hash_url(url)
    html_path = cache_dir / f"{h}.html"
    meta_path = cache_dir / f"{h}.json"
    return html_path, meta_path


# ----------------------------
# robots.txt (simple)
# ----------------------------

def _robots_allows(
    session: requests.Session,
    base_url: str,
    target_url: str,
    timeout: float = 10.0,
) -> bool:
    """
    Very simple robots.txt:
    - fetch scheme://host/robots.txt
    - only parse User-agent: * and Disallow:
    - if robots cannot be fetched, allow
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


# ----------------------------
# Text extraction
# ----------------------------

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
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        return soup.get_text(separator="\n")
    except Exception:
        return html


def _extract_main_text(html: str, url: str) -> Tuple[str, str]:
    """
    Prefer trafilatura if available, else BeautifulSoup fallback.
    Returns (text, title).
    """
    # 1) trafilatura
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            include_images=False,
            output_format="txt",
        )
        if extracted and extracted.strip():
            title = _extract_title_bs4(html) or url
            return extracted, title
    except Exception:
        pass

    # 2) fallback
    text = _fallback_bs4_text(html)
    title = _extract_title_bs4(html) or url
    return text, title


def _extract_links(html: str, base_url: str) -> List[str]:
    """
    Extract absolute links from <a href="...">.
    Filters out obvious bad schemes early.
    """
    out: List[str] = []
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            low = href.lower()
            if any(low.startswith(s) for s in _BLOCKED_SCHEMES):
                continue

            abs_url = urljoin(base_url, href)
            abs_url = _canonicalize(abs_url)

            # Protect against malformed concatenations: base + absolute
            # e.g., "https://constructor.university/https://constructor.university/..."
            if "://constructor.university/https://" in abs_url:
                continue

            out.append(abs_url)
    except Exception:
        return out
    return out


# ----------------------------
# Cache load/save
# ----------------------------

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


def _save_cache(cache_dir: Path, url: str, html: str, title: str) -> str:
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
    return doc_id


# ----------------------------
# Ingest web (crawl)
# ----------------------------

def ingest_web(cfg: Optional[Settings] = None) -> List[WebDoc]:
    """
    Crawl from cfg.web.seeds, cache HTML pages, extract main text.
    Only returns HTML pages as WebDoc (no PDFs, no images).
    """
    cfg = cfg or Settings.load()
    logger = get_logger()

    if not cfg.web.enabled:
        logger.info("Web ingestion disabled (web.enabled=false).")
        return []

    cache_dir = Path(cfg.web_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    seeds = [s for s in (cfg.web.seeds or []) if s]
    allowed_domain = cfg.web.allowed_domain
    max_pages = int(cfg.web.max_pages)
    rate_limit = float(cfg.web.rate_limit_seconds)

    # Optional allow/deny path prefixes (safe defaults: allow all)
    allow_paths = getattr(cfg.web, "allow_paths", None)
    deny_paths = getattr(cfg.web, "deny_paths", None)

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
        if _allow_url(u, allowed_domain, allow_paths=allow_paths, deny_paths=deny_paths):
            queue.append(u)
            seen.add(u)

    while queue and len(docs) < max_pages:
        url = queue.pop(0)
        url = _canonicalize(url)

        if not _allow_url(url, allowed_domain, allow_paths=allow_paths, deny_paths=deny_paths):
            continue

        if not _robots_allows(session, url, url):
            logger.info(f"[web] robots blocked: {url}")
            continue

        cached = _load_cached(cache_dir, url)
        if cached:
            docs.append(cached)
            continue

        time.sleep(rate_limit)

        try:
            # HEAD first to avoid downloading binaries
            try:
                h = session.head(url, timeout=12, allow_redirects=True)
                ctype = (h.headers.get("Content-Type") or "").lower()
                if ctype and ("text/html" not in ctype) and ("application/xhtml" not in ctype):
                    logger.info(f"[web] skip non-html {url} content-type={ctype.split(';')[0]}")
                    continue
            except Exception:
                # If HEAD fails, we proceed with GET (some servers block HEAD)
                pass

            r = session.get(url, timeout=20)
            if r.status_code != 200:
                logger.info(f"[web] skip {url} status={r.status_code}")
                continue

            ctype = (r.headers.get("Content-Type") or "").lower()
            if ctype and ("text/html" not in ctype) and ("application/xhtml" not in ctype):
                logger.info(f"[web] skip non-html {url} content-type={ctype.split(';')[0]}")
                continue

            html = r.text or ""
            text, title = _extract_main_text(html, url)
            text = normalize_pdf_text(text)

            if not text.strip():
                # Don't waste embeddings on empty extractions
                continue

            doc_id = _save_cache(cache_dir, url, html, title)

            docs.append(
                WebDoc(
                    doc_id=doc_id,
                    url=url,
                    title=title or url,
                    text=text,
                    meta={"source": "web", "url": url, "title": title or url, "doc_id": doc_id},
                )
            )

            # Discover more links (bounded)
            for link in _extract_links(html, url):
                if len(seen) >= max_pages * 10:
                    break
                link = _canonicalize(link)
                if link in seen:
                    continue
                if not _allow_url(link, allowed_domain, allow_paths=allow_paths, deny_paths=deny_paths):
                    continue
                seen.add(link)
                queue.append(link)

        except requests.exceptions.SSLError as e:
            logger.info(f"[web] skip ssl {url}: {e}")
            continue
        except requests.exceptions.RequestException as e:
            logger.info(f"[web] error {url}: {e}")
            continue
        except Exception as e:
            logger.info(f"[web] error {url}: {e}")
            continue

    logger.info(f"[web] docs fetched: {len(docs)} (max_pages={max_pages})")
    return docs
