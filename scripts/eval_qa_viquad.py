#!/usr/bin/env python3
"""
Evaluate QA predictions on ViQuAD (SQuAD-style) with EM and token-level F1.

Usage (offline prediction file):
  python scripts/eval_qa_viquad.py --dataset data/raw/ViQuAD/validation.json --pred predictions.json --out results/qa/viquad_eval.json

Where predictions.json is a dict mapping example id -> predicted answer string.

Notes:
- We do not call external APIs here. This is an offline evaluator.
- For quick sampling, use --limit N to evaluate the first N samples.
"""

import json
import argparse
import os
import re
import string
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple


def normalize_answer(s: str) -> str:
    """Lower, normalize unicode, remove punctuation/articles/extra whitespace.
    Close to SQuAD official script with minor Vietnamese-friendly tweaks.
    """
    if s is None:
        return ""
    s = s.strip()
    # Unicode normalize
    s = unicodedata.normalize("NFKC", s)
    # Lowercase
    s = s.lower()

    # Remove punctuation
    def remove_punc(text: str) -> str:
        table = str.maketrans("", "", string.punctuation)
        return text.translate(table)

    s = remove_punc(s)

    # Remove articles (tiếng Anh). Với tiếng Việt, giữ nguyên, chỉ bỏ khoảng trắng thừa
    s = re.sub(r"\s+", " ", s)

    return s.strip()


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    scores = [metric_fn(prediction, gt) for gt in ground_truths]
    return max(scores) if scores else 0.0


def load_viquad(path: str, limit: int = None) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("ViQuAD file should be a JSON array of examples")
    return data[:limit] if limit else data


def evaluate(dataset_path: str, pred_path: str, out_path: str = None, limit: int = None) -> Dict[str, float]:
    ds = load_viquad(dataset_path, limit=limit)
    with open(pred_path, "r", encoding="utf-8") as f:
        predictions: Dict[str, str] = json.load(f)

    total = 0
    exact_matches = 0
    f1_sum = 0.0

    missing = 0

    for ex in ds:
        ex_id = ex.get("id") or ex.get("uit_id")
        ans = ex.get("answers", {})
        gt_texts = ans.get("text", []) or []
        if not gt_texts:
            # impossible QAs might exist; skip or treat as empty
            continue
        total += 1
        pred = predictions.get(ex_id)
        if pred is None:
            missing += 1
            pred = ""
        exact_matches += 1 if metric_max_over_ground_truths(exact_match_score, pred, gt_texts) else 0
        f1_sum += metric_max_over_ground_truths(f1_score, pred, gt_texts)

    em = 100.0 * exact_matches / total if total else 0.0
    f1 = 100.0 * f1_sum / total if total else 0.0

    result = {
        "dataset": os.path.basename(dataset_path),
        "total": total,
        "missing_predictions": missing,
        "EM": round(em, 4),
        "F1": round(f1, 4),
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to ViQuAD json (train/validation/test)")
    parser.add_argument("--pred", required=True, help="Path to predictions.json (id -> answer)")
    parser.add_argument("--out", default=None, help="Where to write metrics json")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N samples")
    args = parser.parse_args()

    evaluate(args.dataset, args.pred, args.out, args.limit)
