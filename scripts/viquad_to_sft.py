#!/usr/bin/env python3
"""
Convert ViQuAD (SQuAD-style) to an instruction-tuning JSONL for generative models.

Output JSONL fields:
{
  "id": str,
  "prompt": str,   # includes context + question
  "response": str  # first ground-truth answer
}

Usage:
  python scripts/viquad_to_sft.py --input data/raw/ViQuAD/train.json --output data/processed/viquad_sft_train.jsonl --max-context-chars 2000
"""

import json
import argparse
import os
from typing import List, Dict


def load_viquad(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("ViQuAD file should be a JSON array of examples")
    return data


def build_prompt(context: str, question: str, max_context_chars: int = None) -> str:
    ctx = context.strip()
    if max_context_chars and len(ctx) > max_context_chars:
        ctx = ctx[:max_context_chars] + "..."
    prompt = (
        "Bạn là trợ lý tiếng Việt. Trả lời CHỈ dựa trên đoạn văn sau.\n\n"
        f"Đoạn văn:\n{ctx}\n\n"
        f"Câu hỏi: {question}\n"
    )
    return prompt


def convert(input_path: str, output_path: str, max_context_chars: int = None, limit: int = None) -> int:
    data = load_viquad(input_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for i, ex in enumerate(data):
            if limit and i >= limit:
                break
            ex_id = ex.get("id") or ex.get("uit_id") or str(i)
            question = (ex.get("question") or "").strip()
            context = (ex.get("context") or "").strip()
            answers = ex.get("answers", {})
            texts = answers.get("text", []) or []
            if not question or not context or not texts:
                continue
            response = texts[0]
            record = {
                "id": ex_id,
                "prompt": build_prompt(context, question, max_context_chars=max_context_chars),
                "response": response,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-context-chars", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    n = convert(args.input, args.output, max_context_chars=args.max_context_chars, limit=args.limit)
    print(f"Wrote {n} SFT records -> {args.output}")
