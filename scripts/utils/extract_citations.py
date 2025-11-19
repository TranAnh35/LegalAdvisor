#!/usr/bin/env python3
"""
Quick CLI to extract citations from input text or a file.

Usage:
  python -m scripts.extract_citations --text "Điều 14 Nghị định 91/2017/NĐ-CP ..."
  python -m scripts.extract_citations --file path/to/text.txt

Outputs JSON lines with fields: article, clause, point, act_code_raw, act_code_norm, method, ambiguity
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root on path, then import via src.retrieval...
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.citation.extract import extract_citations  # type: ignore
from src.utils.law_registry import get_registry  # type: ignore

# Force UTF-8 stdout on Windows consoles to avoid UnicodeEncodeError
try:  # Python 3.7+
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description="Extract legal citations from text")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="Raw text to analyze")
    g.add_argument("--file", type=str, help="Path to a UTF-8 text file")

    parser.add_argument(
        "--article-only",
        action="store_true",
        default=True,
        help="Chỉ sử dụng Điều (article); ép clause/point=None để đồng bộ với hệ thống",
    )

    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        p = Path(args.file)
        if not p.exists():
            print(json.dumps({"error": f"File not found: {p}"}, ensure_ascii=False))
            sys.exit(1)
        text = p.read_text(encoding="utf-8", errors="ignore")

    # Cố gắng dùng registry để hỗ trợ resolve name-based (không năm)
    registry = None
    try:
        registry = get_registry()
    except Exception:
        registry = None

    hits = extract_citations(text, registry=registry, article_only=args.article_only)
    for h in hits:
        obj = {
            "article": h.article,
            "clause": h.clause,
            "point": h.point,
            "act_code_raw": h.act_code_raw,
            "act_code_norm": h.act_code_norm,
            "method": h.method,
            "ambiguity": h.ambiguity,
        }
        print(json.dumps(obj, ensure_ascii=False))


if __name__ == "__main__":
    main()
