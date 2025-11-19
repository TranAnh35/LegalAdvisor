#!/usr/bin/env python3
"""
Script: export_act_codes.py

Mục tiêu:
- Quét dataset (chunks_schema.jsonl hoặc corpus_cleaned.jsonl) để trích xuất danh sách mã văn bản (act_code) ở cấp Điều.
- Mỗi dòng trong dataset hiện là một Điều với định dạng corpus_id: NUMBER/YEAR/TYPE(+SUFFIX_Điều).
  Ví dụ: '159/2020/nđ-cp+13' => act_code chuẩn hoá: '159/2020/NĐ-CP'.
- Xuất ra file JSON và thống kê: tổng số Điều, tổng số act_code unique, top 10 văn bản theo số Điều.

Đầu ra:
- data/registry/act_codes_stats.json
- data/registry/act_codes_unique.json

Chạy:
  conda activate LegalAdvisor
  python scripts/export_act_codes.py --input data/processed/zalo-legal/chunks_schema.jsonl

"""
import json
import argparse
import os
from collections import Counter
from typing import Set, Tuple

INPUT_DEFAULT = "data/processed/zalo-legal/chunks_schema.jsonl"
OUTPUT_STATS = "data/registry/act_codes_stats.json"
OUTPUT_UNIQUE = "data/registry/act_codes_unique.json"

# Các kiểu văn bản chuẩn phổ biến để nhận dạng và viết hoa.
TYPE_NORMALIZATION_MAP = {
    'nđ-cp': 'NĐ-CP', 'nd-cp': 'NĐ-CP',
    'tt-bca': 'TT-BCA', 'tt-bnn': 'TT-BNN', 'tt-btc': 'TT-BTC', 'tt-btp': 'TT-BTP',
    'tt-bnv': 'TT-BNV', 'tt-byt': 'TT-BYT', 'tt-bkhcn': 'TT-BKHCN',
    'qh13': 'QH13', 'qh14': 'QH14', 'qh15': 'QH15',
}


def normalize_act_code(number_year: str, type_raw: str) -> str:
    """Chuẩn hoá mã văn bản ở cấp văn bản (không gồm điều)."""
    tnorm = TYPE_NORMALIZATION_MAP.get(type_raw.lower(), type_raw.upper())
    return f"{number_year}/{tnorm}" if '/' in number_year else f"{number_year}/{tnorm}"


def parse_corpus_id(corpus_id: str) -> Tuple[str, str]:
    """Tách corpus_id thành (number/year, type).
    Ví dụ: '159/2020/nđ-cp+13' -> ('159/2020', 'nđ-cp')
    """
    # Tách phần trước dấu '+' nếu có
    base = corpus_id.split('+')[0]
    parts = base.split('/')
    if len(parts) < 3:
        return '', ''
    number_year = '/'.join(parts[0:2])  # 159/2020
    type_part = parts[2]
    return number_year, type_part


def collect_act_codes(input_path: str) -> Tuple[Set[str], Counter, int]:
    """Đọc file JSONL và thu thập act_code unique + đếm số điều theo act_code."""
    act_codes: Set[str] = set()
    counts = Counter()
    total_lines = 0

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            corpus_id = data.get('corpus_id', '')
            if not corpus_id:
                continue
            number_year, type_part = parse_corpus_id(corpus_id)
            if not number_year or not type_part:
                continue
            act_code = normalize_act_code(number_year, type_part)
            act_codes.add(act_code)
            counts[act_code] += 1

    return act_codes, counts, total_lines


def main():
    parser = argparse.ArgumentParser(description="Xuất danh sách act_code đang dùng")
    parser.add_argument('--input', default=INPUT_DEFAULT, help='Đường dẫn file chunks_schema.jsonl')
    parser.add_argument('--stats-out', default=OUTPUT_STATS, help='File thống kê xuất ra')
    parser.add_argument('--unique-out', default=OUTPUT_UNIQUE, help='File danh sách unique xuất ra')
    args = parser.parse_args()

    act_codes, counts, total_lines = collect_act_codes(args.input)

    # Thống kê top 10 văn bản theo số Điều
    top10 = counts.most_common(10)

    stats = {
        'total_lines': total_lines,
        'unique_act_codes': len(act_codes),
        'top10_by_articles': [{'act_code': ac, 'articles': c} for ac, c in top10]
    }

    os.makedirs(os.path.dirname(args.stats_out), exist_ok=True)
    with open(args.stats_out, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    with open(args.unique_out, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(act_codes)), f, ensure_ascii=False, indent=2)

    print(f"✅ Đã xuất: {args.unique_out} ({len(act_codes)} mã)")
    print(f"✅ Thống kê: {args.stats_out}")


if __name__ == '__main__':
    main()
