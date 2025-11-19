"""Normalize Unicode for Zalo Legal corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, Tuple

from ftfy import fix_text
from tqdm import tqdm


JsonDict = Dict[str, object]


def read_jsonl(path: Path) -> Iterator[Tuple[int, JsonDict]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover
                raise ValueError(f"Loi JSON o dong {line_no}: {exc}") from exc


def normalize_entry(record: JsonDict) -> Tuple[JsonDict, bool]:
    """Normalize title/text fields; return (record, changed?)."""

    changed = False
    output = dict(record)

    for key in ("title", "text"):
        value = output.get(key)
        if isinstance(value, str) and value:
            fixed = fix_text(value)
            if fixed != value:
                output[key] = fixed
                changed = True

    return output, changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Unicode cho corpus Zalo Legal")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/zalo_ai_legal_text_retrieval/corpus.jsonl"),
        help="Path corpus goc",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/zalo-legal/corpus_cleaned.jsonl"),
        help="Path corpus sau khi normalize",
    )

    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Khong tim thay file dau vao: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    changed_rows = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for _, record in tqdm(read_jsonl(input_path), desc="Normalize corpus", unit="line"):
            normalized, changed = normalize_entry(record)
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            total += 1
            if changed:
                changed_rows += 1

    print(f"OK - Ghi {total} dong vao {output_path}")
    print(f"Thong tin: {changed_rows} dong duoc sua Unicode")


if __name__ == "__main__":
    main()