from __future__ import annotations

from typing import Optional
import requests

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore


def _safe_get(url: str, timeout: int = 8) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "AIEAS/1.0"})
        if r.status_code == 200:
            return r.text
    except Exception:
        return None
    return None


def fetch_github_profile(identifier: str) -> str:
    """Fetch public GitHub profile text for a username or profile URL.

    Returns a short concatenation of bio, blog, and popular repo names when
    available. This tries the GitHub public API first (rate-limited) and
    falls back to simple HTML scraping.
    """
    if not identifier:
        return ""

    # Normalize to username if a profile URL is provided
    username = identifier.strip().rstrip("/")
    if username.startswith("http"):
        username = username.split("/")[-1]

    # Try public API
    api_url = f"https://api.github.com/users/{username}"
    try:
        resp = requests.get(api_url, headers={"User-Agent": "AIEAS/1.0"}, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            parts = [data.get("name") or "", data.get("bio") or "", data.get("blog") or ""]
            return "\n".join(p for p in parts if p)
    except Exception:
        pass

    # Fallback to scraping the public profile page
    html = _safe_get(f"https://github.com/{username}")
    if not html or not BeautifulSoup:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    bio_el = soup.select_one(".p-note") or soup.select_one(".user-profile-bio")
    name_el = soup.select_one(".vcard-fullname")
    parts = []
    if name_el:
        parts.append(name_el.get_text(strip=True))
    if bio_el:
        parts.append(bio_el.get_text(strip=True))
    # add repository names
    repo_els = soup.select("article h3 a")[:5]
    if repo_els:
        parts.append("Repos: " + ", ".join(a.get_text(strip=True) for a in repo_els))
    return "\n".join(parts)


def fetch_linkedin_profile(url: str) -> str:
    """Try to extract public LinkedIn profile text. LinkedIn blocks scraping
    aggressively; this function is best-effort and may return empty string.
    Prefer copy/paste of the public profile or using official APIs.
    """
    if not url:
        return ""

    html = _safe_get(url)
    if not html or not BeautifulSoup:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    # Attempt to extract headline, about, and sections
    parts = []
    headline = soup.select_one(".pv-text-details__left-panel .text-heading-xlarge, .top-card-layout__first-subline")
    about = soup.select_one("section.pv-about-section, .core-section-container__content")
    if headline:
        parts.append(headline.get_text(strip=True))
    if about:
        parts.append(about.get_text(strip=True))

    # fallback: meta description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        parts.append(meta_desc.get("content"))

    return "\n".join(p for p in parts if p)


def extract_profile_links(text: str) -> dict:
    """Extract GitHub username/URL and LinkedIn profile URL from arbitrary text.

    Returns a dict with optional keys: 'github' and 'linkedin'. Values are the
    detected raw strings (username or URL). This is a best-effort extractor
    using simple regexes sufficient for resume text.
    """
    if not text:
        return {}

    import re

    out = {}

    # GitHub: github.com/username or plain username (heuristic: words with no spaces
    # and only alphanum/-/_). We'll prefer explicit URLs first.
    m = re.search(r"https?://(?:www\.)?github\.com/([A-Za-z0-9_-]+)", text)
    if m:
        out["github"] = m.group(1)
    else:
        # look for common 'github:' patterns
        m2 = re.search(r"github[:\s]*([A-Za-z0-9_-]+)", text, flags=re.I)
        if m2:
            out["github"] = m2.group(1)

    # LinkedIn: look for linkedin.com/in/... or linkedin.com/pub/...
    # ensure hyphen is placed safely in the character class (or escaped) to
    # avoid creating unintended ranges like _-/
    m = re.search(r"https?://(?:[A-Za-z0-9_\-]+\.)?linkedin\.com/(?:in|pub)/[A-Za-z0-9_/%-]+", text)
    if m:
        out["linkedin"] = m.group(0)
    else:
        # look for linkedin mention followed by a URL or handle
        m2 = re.search(r"linkedin[:\s]*(https?://[^\s]+)", text, flags=re.I)
        if m2:
            out["linkedin"] = m2.group(1)

    return out
