#!/usr/bin/env python3
"""
Citation extractor for Vietnamese legal texts.

Goals:
- Extract (Điều/ Khoản/ Điểm) and act code references (e.g., 91/2017/NĐ-CP, 104/2016/TT-BCA).
- Be robust to ordering, casing, and diacritics.
- Prefer ID-based citations; support name-based as optional/ambiguous.

Output model (CitationHit):
- article (int|None), clause (int|None), point (str|None)
- act_code_raw (str|None), act_code_norm (str|None)
- method: "ID" | "NAME"
- ambiguity: bool
- span: (start, end) of the matched window

Notes:
- Name-based resolution is kept minimal; if a registry is provided, you can plug further mapping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import unicodedata

try:
    # Prefer project normalization util if available
    from ...utils.law_registry import normalize_act_code
except Exception:  # pragma: no cover
    def normalize_act_code(code: Optional[str]) -> Optional[str]:  # type: ignore
        return (code or "").strip().upper() or None


# --- Regex components -----------------------------------------------------

# Vietnamese diacritics range: À-Ỵà-ỵ (but we primarily match uppercase in codes)
UPPER_LETTERS = r"A-ZĐÂÊÔƠƯÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"

# Location patterns: điểm a khoản 2 điều 14 (order can vary)
RE_DIEM = re.compile(r"(?i)\bđiểm\s+([a-z])\b", re.UNICODE)
RE_KHOAN = re.compile(r"(?i)\bkhoản\s+(\d{1,2})\b", re.UNICODE)
RE_DIEU = re.compile(r"(?i)\bđiều\s+(\d{1,3})\b", re.UNICODE)
# ASCII fallbacks (no diacritics)
RE_DIEM_ASCII = re.compile(r"(?i)\bdiem\s+([a-z])\b")
RE_KHOAN_ASCII = re.compile(r"(?i)\bkhoan\s+(\d{1,2})\b")
RE_DIEU_ASCII = re.compile(r"(?i)\bdieu\s+(\d{1,3})\b")

# Multi-article cluster pattern (Điều 21, 22, 23 và Điều 24 ...)
RE_DIEU_CLUSTER = re.compile(
    r"(?i)điều\s+\d{1,3}(?:[\s,]*(?:và)?\s*\d{1,3})*",
    re.UNICODE,
)
RE_DIEU_CLUSTER_ASCII = re.compile(
    r"(?i)dieu\s+\d{1,3}(?:[\s,]*(?:va)?\s*\d{1,3})*"
)

# ID-based act code like 91/2017/NĐ-CP or 104/2016/TT-BCA (allow hyphens)
# Capture number/year/type; type can include hyphens and uppercase with diacritics
RE_ACT_CODE = re.compile(
    rf"\b(\d{{1,3}})/(19\d{{2}}|20\d{{2}})/([{UPPER_LETTERS}][{UPPER_LETTERS}-]+)\b",
    re.UNICODE,
)

# Optional prefixes like "Nghị định", "Thông tư" before the code (not required)
RE_PREFIXED_CODE = re.compile(
    rf"(?i)\b(nghị\s*định|thông\s*tư|quyết\s*định|luật|bộ\s*luật)[^\n\r]{{0,40}}?"  # short window
    rf"(\d{{1,3}}/(?:19\d{{2}}|20\d{{2}})/[{UPPER_LETTERS}][{UPPER_LETTERS}-]+)\b",
    re.UNICODE,
)

# Name-based (very limited, e.g., "Bộ luật Dân sự 2015"): capture year
RE_NAME_BASED = re.compile(
    r"(?i)\b(bộ\s*luật|luật|nghị\s*định|thông\s*tư)[^\n\r]{0,60}?((?:19|20)\d{2})\b",
    re.UNICODE,
)
RE_NAME_BASED_ASCII = re.compile(
    r"(?i)\b(bo\s*luat|luat|nghi\s*dinh|thong\s*tu)[^\n\r]{0,60}?((?:19|20)\d{2})\b",
)

# Keyword-only (no year) name-based, short window after keyword
RE_NAME_KEYWORD_ONLY = re.compile(
    r"(?i)\b(bộ\s*luật|luật|nghị\s*định|thông\s*tư)[^\n\r]{0,40}",
    re.UNICODE,
)
RE_NAME_KEYWORD_ONLY_ASCII = re.compile(
    r"(?i)\b(bo\s*luat|luat|nghi\s*dinh|thong\s*tu)[^\n\r]{0,40}"
)


@dataclass
class ActLocation:
    point: Optional[str] = None
    clause: Optional[int] = None
    article: Optional[int] = None

    @classmethod
    def from_text(cls, text: str) -> "ActLocation":
        # Trích Điều và (nếu có) Khoản; bỏ qua Điểm mặc định
        article = None
        clause = None
        for m in RE_DIEU.finditer(text):
            try:
                article = int(m.group(1))
            except Exception:
                pass
        for m in RE_DIEU_ASCII.finditer(text):
            try:
                article = int(m.group(1))
            except Exception:
                pass
        for m in RE_KHOAN.finditer(text):
            try:
                clause = int(m.group(1))
            except Exception:
                pass
        for m in RE_KHOAN_ASCII.finditer(text):
            try:
                clause = int(m.group(1))
            except Exception:
                pass
        return cls(point=None, clause=clause, article=article)


@dataclass
class CitationHit:
    article: Optional[int]
    clause: Optional[int]
    point: Optional[str]
    act_code_raw: Optional[str]
    act_code_norm: Optional[str]
    method: str  # "ID" or "NAME"
    ambiguity: bool
    span: Tuple[int, int]
    text_window: str


def _window(text: str, start: int, end: int, radius: int = 160) -> str:
    a = max(0, start - radius)
    b = min(len(text), end + radius)
    return text[a:b]


def _normalize_code_ascii(code: Optional[str]) -> Optional[str]:
    if code is None:
        return None
    # Normalize dashes and remove diacritics, including mapping Đ/đ
    s = code.replace("–", "-").replace("—", "-")
    s = s.replace("Đ", "D").replace("đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s


def _make_hit(text: str, start: int, end: int, code: Optional[str], method: str, *, article: Optional[int] = None) -> CitationHit:
    window = _window(text, start, end)
    # Find location from window
    # Allow override of article when cluster parsing provides explicit number
    loc = ActLocation.from_text(window)
    if article is not None:
        loc.article = article
    raw = code
    norm = _normalize_code_ascii(code)
    ambiguity = code is None
    return CitationHit(
        article=loc.article,
        clause=loc.clause,
        point=loc.point,
        act_code_raw=raw,
        act_code_norm=norm,
        method=method,
        ambiguity=ambiguity or (norm is None),
        span=(start, end),
        text_window=window,
    )


def _extract_articles_before(text: str, start: int) -> List[int]:
    """Find the nearest Điều-cluster BEFORE the given index and return its numbers."""
    left = text[max(0, start - 160): start]
    matches = list(RE_DIEU_CLUSTER.finditer(left))
    matches += list(RE_DIEU_CLUSTER_ASCII.finditer(left))
    if not matches:
        return []
    last = matches[-1]
    nums = [int(x) for x in re.findall(r"\d{1,3}", last.group(0))]
    return list(dict.fromkeys(nums))


def _extract_articles_after(text: str, end: int) -> List[int]:
    """Find the first Điều-cluster AFTER the given index and return its numbers."""
    right = text[end: min(len(text), end + 160)]
    m = RE_DIEU_CLUSTER.search(right) or RE_DIEU_CLUSTER_ASCII.search(right)
    if not m:
        return []
    nums = [int(x) for x in re.findall(r"\d{1,3}", m.group(0))]
    return list(dict.fromkeys(nums))


def _find_articles_for_code(text: str, start: int, end: int) -> List[int]:
    """Choose the most plausible Điều numbers associated with a specific code occurrence.

    Preference: nearest cluster before the code; if none, take the first cluster after.
    """
    before = _extract_articles_before(text, start)
    if before:
        return before
    after = _extract_articles_after(text, end)
    return after


def _normalize_str(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _resolve_name_with_registry(window: str, registry: Any, year_hint: Optional[int] = None) -> Optional[str]:
    """Try to map a generic name-based mention (no explicit code) to an act_code via registry.

    Heuristic:
    - Normalize window and candidate names (remove accents, lowercase).
    - Remove generic words (bo, luat, nghi, dinh, thong, tu, cua, ve, nam, nay).
    - If a candidate's act_name/official_title contains the remaining phrase tokens, consider a match.
    - If multiple, prefer higher year; fallback to longest name match.
    """
    if not registry:
        return None

    win = _normalize_str(window)
    # Strip generic keywords
    # Detect type keyword to constrain mapping
    type_keyword = None
    if re.search(r"\bbo\s+luat\b", win):
        type_keyword = "bo luat"
    elif re.search(r"\bluat\b", win):
        type_keyword = "luat"
    elif re.search(r"\bnghi\s+dinh\b", win):
        type_keyword = "nghi dinh"
    elif re.search(r"\bthong\s+tu\b", win):
        type_keyword = "thong tu"

    win_core = re.sub(r"\b(bo|luat|nghi|dinh|thong|tu|cua|ve|nam|nay)\b", " ", win)
    win_core = re.sub(r"\s+", " ", win_core).strip()
    if not win_core:
        return None

    # Access registry data
    entries = []
    data = getattr(registry, "_data", None)
    if isinstance(data, dict):
        entries = list(data.values())
    # If structure unknown, give up
    if not entries:
        return None

    best_code = None
    best_score = -1
    best_year = -1

    for info in entries:
        # info fields: act_code, act_name, official_title, year
        name = getattr(info, "act_name", None) or ""
        title = getattr(info, "official_title", None) or ""
        cand = _normalize_str(name) or _normalize_str(title)
        if not cand:
            continue
        # Constrain by type keyword: avoid mismatching Thông tư vs Bộ luật
        if type_keyword is not None:
            if type_keyword == "bo luat" and not cand.startswith("bo luat"):
                continue
            if type_keyword == "luat" and not cand.startswith("luat"):
                # allow 'bo luat' as a superset of 'luat'
                if not cand.startswith("bo luat"):
                    continue
            if type_keyword == "nghi dinh" and not cand.startswith("nghi dinh"):
                continue
            if type_keyword == "thong tu" and not cand.startswith("thong tu"):
                continue
        # simple containment score
        if win_core and win_core in cand:
            score = len(win_core)
        elif cand and cand in win_core:
            score = len(cand)
        else:
            # token overlap
            wtok = set(win_core.split())
            ctok = set(cand.split())
            overlap = len(wtok & ctok)
            if overlap == 0:
                continue
            score = overlap

        y = getattr(info, "year", None)
        yval = int(y) if isinstance(y, int) else -1
        # prefer year_hint if provided
        if year_hint is not None and yval != year_hint:
            # small penalty when not matching hint
            score_adj = score - 1
        else:
            score_adj = score

        if score_adj > best_score or (score_adj == best_score and yval > best_year):
            best_score = score_adj
            best_year = yval
            best_code = getattr(info, "act_code", None)

    return best_code


def extract_citations(
    text: str,
    registry: Optional[Any] = None,
    allow_name_based: bool = True,
    *,
    include_point: bool = False,
    article_only: bool = False,
) -> List[CitationHit]:
    """Extract legal citations from a string.

    - Prefer ID-based act codes (NN/YYY/TYPE)
    - Optionally attempt name-based matches (ambiguous=True unless resolved definitively)
    - Returns deduplicated hits by (act_code_norm, article, clause, point) if possible
    """
    if not text or not isinstance(text, str):
        return []

    hits: List[CitationHit] = []

    # 1) Strict ID-based codes
    for m in RE_ACT_CODE.finditer(text):
        code = m.group(0)
        articles = _find_articles_for_code(text, m.start(), m.end())
        if articles:
            for art in articles:
                hits.append(_make_hit(text, m.start(), m.end(), code, method="ID", article=art))
        else:
            hits.append(_make_hit(text, m.start(), m.end(), code, method="ID"))

    # 2) Prefixed codes (covers variants like "Nghị định ... 91/2017/NĐ-CP")
    for m in RE_PREFIXED_CODE.finditer(text):
        code = m.group(2)
        articles = _find_articles_for_code(text, m.start(), m.end())
        if articles:
            for art in articles:
                hits.append(_make_hit(text, m.start(2), m.end(2), code, method="ID", article=art))
        else:
            hits.append(_make_hit(text, m.start(2), m.end(2), code, method="ID"))

    # 3) Optional: name-based (kept ambiguous unless registry resolves uniquely)
    if allow_name_based:
        for m in RE_NAME_BASED.finditer(text):
            year = None
            try:
                year = int(m.group(2))
            except Exception:
                year = None
            # Try to resolve with registry if present, otherwise mark ambiguous
            code: Optional[str] = None
            window = _window(text, m.start(), m.end())
            if registry is not None:
                try:
                    code = _resolve_name_with_registry(window, registry, year_hint=year)
                except Exception:
                    code = None
            # Extract Điều clusters
            articles: List[int] = []
            for cm in RE_DIEU_CLUSTER.finditer(window):
                articles.extend([int(x) for x in re.findall(r"\d{1,3}", cm.group(0))])
            for cm in RE_DIEU_CLUSTER_ASCII.finditer(window):
                articles.extend([int(x) for x in re.findall(r"\d{1,3}", cm.group(0))])
            articles = list(dict.fromkeys(articles))
            loc = ActLocation.from_text(window)
            if articles:
                for art in articles:
                    hits.append(_make_hit(text, m.start(), m.end(), code, method="NAME", article=art))
            else:
                hits.append(_make_hit(text, m.start(), m.end(), code, method="NAME", article=loc.article))

        # ASCII fallback with year
        for m in RE_NAME_BASED_ASCII.finditer(text):
            window = _window(text, m.start(), m.end())
            code = None
            if registry is not None:
                try:
                    code = _resolve_name_with_registry(window, registry, year_hint=None)
                except Exception:
                    code = None
            articles: List[int] = []
            for cm in RE_DIEU_CLUSTER.finditer(window):
                articles.extend([int(x) for x in re.findall(r"\d{1,3}", cm.group(0))])
            for cm in RE_DIEU_CLUSTER_ASCII.finditer(window):
                articles.extend([int(x) for x in re.findall(r"\d{1,3}", cm.group(0))])
            articles = list(dict.fromkeys(articles))
            loc = ActLocation.from_text(window)
            if articles:
                for art in articles:
                    hits.append(_make_hit(text, m.start(), m.end(), code, method="NAME", article=art))
            else:
                hits.append(_make_hit(text, m.start(), m.end(), code, method="NAME", article=loc.article))

        # Keyword-only (no year): produce ambiguous NAME hit so that callers can choose to resolve using registry/alias
        for m in RE_NAME_KEYWORD_ONLY.finditer(text):
            window = _window(text, m.start(), m.end())
            code = None
            if registry is not None:
                try:
                    code = _resolve_name_with_registry(window, registry, year_hint=None)
                except Exception:
                    code = None
            articles: List[int] = []
            for cm in RE_DIEU_CLUSTER.finditer(window):
                articles.extend([int(x) for x in re.findall(r"\d{1,3}", cm.group(0))])
            for cm in RE_DIEU_CLUSTER_ASCII.finditer(window):
                articles.extend([int(x) for x in re.findall(r"\d{1,3}", cm.group(0))])
            articles = list(dict.fromkeys(articles))
            loc = ActLocation.from_text(window)
            if articles:
                for art in articles:
                    hits.append(_make_hit(text, m.start(), m.end(), code, method="NAME", article=art))
            else:
                hits.append(_make_hit(text, m.start(), m.end(), code, method="NAME", article=loc.article))
        for m in RE_NAME_KEYWORD_ONLY_ASCII.finditer(text):
            window = _window(text, m.start(), m.end())
            code = None
            if registry is not None:
                try:
                    code = _resolve_name_with_registry(window, registry, year_hint=None)
                except Exception:
                    code = None
            articles: List[int] = []
            for cm in RE_DIEU_CLUSTER.finditer(window):
                articles.extend([int(x) for x in re.findall(r"\d{1,3}", cm.group(0))])
            for cm in RE_DIEU_CLUSTER_ASCII.finditer(window):
                articles.extend([int(x) for x in re.findall(r"\d{1,3}", cm.group(0))])
            articles = list(dict.fromkeys(articles))
            loc = ActLocation.from_text(window)
            if articles:
                for art in articles:
                    hits.append(_make_hit(text, m.start(), m.end(), code, method="NAME", article=art))
            else:
                hits.append(_make_hit(text, m.start(), m.end(), code, method="NAME", article=loc.article))

    # Remove ambiguous NAME hits shadowed by nearby ID hits for same article
    id_positions: List[Tuple[int, Optional[int]]] = []  # (mid, article)
    for h in hits:
        if h.method == "ID" and h.article is not None:
            mid = (h.span[0] + h.span[1]) // 2
            id_positions.append((mid, h.article))

    filtered: List[CitationHit] = []
    for h in hits:
        if h.method == "NAME" and h.ambiguity and h.article is not None:
            mid = (h.span[0] + h.span[1]) // 2
            if any((abs(mid - im) <= 120 and h.article == art) for im, art in id_positions):
                continue  # drop ambiguous NAME overshadowed by ID
        filtered.append(h)

    # Post-process: article-only mode (normalize clause/point to None)
    if article_only:
        for h in hits:
            h.clause = None
            h.point = None

    # Deduplicate by (code or raw, article)
    dedup: Dict[Tuple[Optional[str], Optional[int]], CitationHit] = {}
    for h in filtered:
        key = (h.act_code_norm or h.act_code_raw, h.article)
        if key in dedup:
            if dedup[key].ambiguity and not h.ambiguity:
                dedup[key] = h
        else:
            dedup[key] = h

    return list(dedup.values())


# ---- Simple CLI for quick manual check -----------------------------------
if __name__ == "__main__":  # pragma: no cover
    import sys as _sys
    sample = (
        "Theo điểm a khoản 2 Điều 14 Nghị định 91/2017/NĐ-CP và Điều 8 Thông tư 104/2016/TT-BCA ..."
    )
    text = sample if len(_sys.argv) == 1 else " ".join(_sys.argv[1:])
    out = extract_citations(text)
    for i, h in enumerate(out, 1):
        print(f"[{i}] {h.method} | {h.act_code_norm or h.act_code_raw} | Đ{h.article} K{h.clause} Điểm {h.point} | amb={h.ambiguity}")
