#!/usr/bin/env python3
"""Create the processed Zalo Legal chunks used by the retriever."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing.zalo_legal import preprocess_corpus


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main() -> None:
    configure_console_encoding()
    output_file, num_chunks = preprocess_corpus()
    print("\nDone")
    print(f"Output: {output_file}")
    print(f"Chunks: {num_chunks}")


if __name__ == "__main__":
    main()