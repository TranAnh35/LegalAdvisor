#!/usr/bin/env python3
"""
CLI hợp nhất cho các tác vụ dữ liệu của LegalAdvisor.

Các lệnh hỗ trợ:
- download-viquad: Tải (hoặc tạo mock) bộ dữ liệu ViQuAD
- split-chunks: Tạo split IDs cho retrieval chunks từ SQLite/Parquet (thí nghiệm)
- export-txt: Xuất txt từ SQLite/Parquet để debug/tham khảo
"""

import argparse
import json
from pathlib import Path
from typing import List

from ..utils.paths import get_project_root, get_processed_data_dir


def cmd_download_viquad() -> bool:
    """Tải ViQuAD về data/raw/ViQuAD (hoặc tạo mock nếu lỗi)."""
    base = get_project_root() / 'data' / 'raw' / 'ViQuAD'
    base.mkdir(parents=True, exist_ok=True)

    existing = [base / 'train.json', base / 'validation.json', base / 'test.json']
    if all(p.exists() for p in existing):
        print("✅ Dataset ViQuAD đã có sẵn.")
        return True

    print("🚀 Đang tải ViQuAD từ Hugging Face...")
    try:
        from datasets import load_dataset  # type: ignore
        dataset = load_dataset("bigscience-data/roots_ca_viquiquad")
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                out = base / f"{split}.json"
                with open(out, 'w', encoding='utf-8') as f:
                    json.dump([dict(it) for it in dataset[split]], f, ensure_ascii=False, indent=2)
                print(f"💾 Lưu {split}: {out}")
        print("✅ Tải ViQuAD thành công!")
        return True
    except Exception as e:
        print(f"⚠️ Không thể tải từ Hugging Face ({e}). Tạo mock dataset...")

    mock = {
        "train": [
            {
                "context": "Điều 1... Mọi người đều bình đẳng về quyền lợi và nghĩa vụ công dân...",
                "question": "Mọi người đều bình đẳng về điều gì?",
                "answers": {"text": ["quyền lợi và nghĩa vụ công dân"], "answer_start": [15]},
            },
        ] * 50,
        "validation": [
            {
                "context": "Điều 3... Quyền con người, quyền công dân chỉ có thể bị hạn chế...",
                "question": "Quyền con người có thể bị hạn chế khi nào?",
                "answers": {"text": ["theo quy định của luật trong trường hợp cần thiết"], "answer_start": [58]},
            },
        ] * 10,
        "test": [
            {
                "context": "Điều 4... quyền bất khả xâm phạm về thân thể...",
                "question": "Mọi người có những quyền gì về thân thể?",
                "answers": {"text": ["quyền sống, quyền bất khả xâm phạm về thân thể"], "answer_start": [15]},
            },
        ] * 10,
    }

    for split in ['train', 'validation', 'test']:
        out = base / f"{split}.json"
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(mock[split], f, ensure_ascii=False, indent=2)
        print(f"💾 Lưu mock {split}: {out}")
    print("✅ Đã tạo mock ViQuAD!")
    return True


def cmd_split_chunks(limit: int = 200000) -> bool:
    """Tạo split IDs cho chunks từ SQLite/Parquet (chỉ phục vụ thí nghiệm)."""
    from sklearn.model_selection import train_test_split  # type: ignore

    processed = get_processed_data_dir()
    sqlite_path = processed / 'smart_chunks_stable.db'
    parquet_path = processed / 'smart_chunks_stable.parquet'

    rows: List[int] = []
    try:
        if sqlite_path.exists():
            import sqlite3
            conn = sqlite3.connect(str(sqlite_path))
            cur = conn.cursor()
            cur.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id LIMIT ?", (int(limit),))
            rows = [r[0] for r in cur.fetchall()]
            conn.close()
        elif parquet_path.exists():
            import pandas as pd  # type: ignore
            df = pd.read_parquet(parquet_path)
            rows = df.sort_values('chunk_id').head(limit)['chunk_id'].astype(int).tolist()
        else:
            print("❌ Không tìm thấy SQLite/Parquet để split chunks")
            return False
    except Exception as e:
        print(f"❌ Lỗi đọc processed data: {e}")
        return False

    print(f"📊 Tổng số chunk IDs (giới hạn): {len(rows)}")

    train_ids, temp_ids = train_test_split(rows, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    out_dir = processed / 'splits'
    out_dir.mkdir(exist_ok=True)
    for name, ids in {
        'train_ids': train_ids,
        'validation_ids': val_ids,
        'test_ids': test_ids,
    }.items():
        out = out_dir / f"{name}.json"
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(ids, f, ensure_ascii=False, indent=2)
        print(f"💾 Lưu {name}: {len(ids)} → {out}")

    print("✅ Hoàn thành split chunks IDs!")
    return True


def cmd_export_txt(limit: int = 2000) -> bool:
    """Xuất txt từ processed data, nhóm theo doc_file (debug)."""
    processed = get_processed_data_dir()
    sqlite_path = processed / 'smart_chunks_stable.db'
    parquet_path = processed / 'smart_chunks_stable.parquet'
    out_dir = get_project_root() / 'data' / 'txt_documents'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    try:
        if sqlite_path.exists():
            import sqlite3
            conn = sqlite3.connect(str(sqlite_path))
            cur = conn.cursor()
            cur.execute("SELECT doc_file, content FROM chunks ORDER BY chunk_id LIMIT ?", (int(limit),))
            rows = cur.fetchall()
            conn.close()
        elif parquet_path.exists():
            import pandas as pd  # type: ignore
            df = pd.read_parquet(parquet_path)
            df = df.sort_values('chunk_id').head(limit)
            rows = list(zip(df['doc_file'].tolist(), df['content'].tolist()))
        else:
            print("❌ Không tìm thấy SQLite/Parquet")
            return False
    except Exception as e:
        print(f"❌ Lỗi đọc processed data: {e}")
        return False

    from collections import defaultdict
    grouped = defaultdict(list)
    for doc_file, content in rows:
        name = Path(doc_file).name if doc_file else 'unknown'
        text = (content or '').replace('_', ' ').strip()
        if text:
            grouped[name].append(text)

    total = 0
    for name, lines in grouped.items():
        try:
            with open(out_dir / f"{name}.txt", 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(lines))
            total += 1
        except Exception:
            pass

    print(f"✅ Đã xuất {total} file vào {out_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="LegalAdvisor Data CLI")
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('download-viquad', help='Tải ViQuAD hoặc tạo mock nếu lỗi')

    sp_split = sub.add_parser('split-chunks', help='Tạo split IDs cho chunks (thí nghiệm)')
    sp_split.add_argument('--limit', type=int, default=200000, help='Giới hạn số chunk đọc')

    sp_export = sub.add_parser('export-txt', help='Xuất txt từ processed data để debug')
    sp_export.add_argument('--limit', type=int, default=2000, help='Giới hạn số chunk xuất')

    args = parser.parse_args()

    if args.command == 'download-viquad':
        ok = cmd_download_viquad()
    elif args.command == 'split-chunks':
        ok = cmd_split_chunks(limit=args.limit)
    elif args.command == 'export-txt':
        ok = cmd_export_txt(limit=args.limit)
    else:
        ok = False

    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()


