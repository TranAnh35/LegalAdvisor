#!/usr/bin/env python3
"""Build a minimal law registry from the processed LegalAdvisor corpus.

The registry is used only for display/citation resolution. It does not affect
FAISS retrieval or embedding quality.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

TYPE_MAP = {
    "nđ-cp": "NĐ-CP",
    "nd-cp": "NĐ-CP",
    "tt": "TT",
    "tt-bca": "TT-BCA",
    "tt-bnn": "TT-BNN",
    "tt-btc": "TT-BTC",
    "tt-btp": "TT-BTP",
    "tt-bnv": "TT-BNV",
    "tt-byt": "TT-BYT",
    "tt-bkhcn": "TT-BKHCN",
    "ttlt": "TTLT",
    "qh13": "QH13",
    "qh14": "QH14",
    "qh15": "QH15",
    "lh": "LH",
    "qh": "QH",
}

TYPE_LABELS = {
    "NĐ-CP": "Nghị định",
    "TT": "Thông tư",
    "TT-BCA": "Thông tư",
    "TT-BNN": "Thông tư",
    "TT-BTC": "Thông tư",
    "TT-BTP": "Thông tư",
    "TT-BNV": "Thông tư",
    "TT-BYT": "Thông tư",
    "TT-BKHCN": "Thông tư",
    "TTLT": "Thông tư liên tịch",
    "QH13": "Luật",
    "QH14": "Luật",
    "QH15": "Luật",
    "LH": "Luật",
    "QH": "Luật",
}


def normalize_act_code(raw_code: str) -> str:
    raw_code = (raw_code or "").strip()
    if not raw_code:
        return ""
    parts = raw_code.split("/")
    if len(parts) < 3:
        return raw_code.upper()
    number = parts[0]
    year = parts[1]
    type_raw = "/".join(parts[2:])
    type_norm = TYPE_MAP.get(type_raw.lower(), type_raw.upper())
    return f"{number}/{year}/{type_norm}"


def fallback_title(act_code: str, act_type: Optional[str] = None) -> str:
    parts = act_code.split("/")
    if len(parts) < 3:
        return act_code
    type_norm = act_type or parts[2]
    label = TYPE_LABELS.get(type_norm, type_norm)
    return f"{label} {parts[0]}/{parts[1]}/{type_norm}"


def iter_chunks(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def load_crawler_titles(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    titles: Dict[str, Dict[str, Any]] = {}
    if not path or not path.exists():
        return titles
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            raw_code = row.get("doc_code") or row.get("so_hieu") or row.get("act_code") or ""
            code = normalize_act_code(str(raw_code))
            if not code:
                continue
            title = row.get("title") or row.get("trich_yeu") or row.get("official_title") or ""
            titles[code] = {
                "title": str(title).strip(),
                "issuer": row.get("issuer") or row.get("co_quan_ban_hanh"),
                "doc_type": row.get("doc_type") or row.get("loai_van_ban"),
            }
    return titles


def build_registry(chunks_path: Path, crawler_docs_path: Optional[Path]) -> list[Dict[str, Any]]:
    counts: Dict[str, int] = defaultdict(int)
    meta: Dict[str, Dict[str, Any]] = {}
    crawler_titles = load_crawler_titles(crawler_docs_path)

    for row in iter_chunks(chunks_path):
        corpus_id = str(row.get("corpus_id") or "").strip()
        if not corpus_id:
            continue
        raw_code = corpus_id.split("+", 1)[0]
        code = normalize_act_code(raw_code)
        if not code:
            continue
        counts[code] += 1
        entry = meta.setdefault(code, {})
        if row.get("year") and not entry.get("year"):
            try:
                entry["year"] = int(row.get("year"))
            except Exception:
                entry["year"] = row.get("year")
        if row.get("type") and not entry.get("raw_type"):
            entry["raw_type"] = row.get("type")

    registry: list[Dict[str, Any]] = []
    for code in sorted(counts):
        parts = code.split("/")
        act_type = parts[2] if len(parts) >= 3 else None
        enriched = crawler_titles.get(code, {})
        official_title = enriched.get("title") or fallback_title(code, act_type)
        act_name = official_title
        registry.append(
            {
                "act_code": code,
                "act_name": act_name,
                "official_title": official_title,
                "year": meta.get(code, {}).get("year"),
                "act_type": act_type,
                "issuer": enriched.get("issuer"),
                "article_count": counts[code],
            }
        )
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/registry/law_registry.json from chunks_schema.jsonl")
    parser.add_argument("--chunks", default="data/processed/zalo-legal/chunks_schema.jsonl")
    parser.add_argument("--crawler-docs", default="", help="Optional crawler documents JSONL for title enrichment. Empty by default to avoid wrong title collisions.")
    parser.add_argument("--output", default="data/registry/law_registry.json")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    crawler_docs_path = Path(args.crawler_docs) if args.crawler_docs else None
    output_path = Path(args.output)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    registry = build_registry(chunks_path, crawler_docs_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(registry)} registry entries to {output_path}")


if __name__ == "__main__":
    main()
