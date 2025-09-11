#!/usr/bin/env python3
"""
Enrich metadata: thêm effective_date/effective_year cho VNLegalText.

Nguồn:
- data/raw/VNLegalText/**/*.xml (cố gắng trích ngày hiệu lực từ các thẻ/thuộc tính hoặc nội dung)

Đầu vào/đầu ra:
- Đọc models/retrieval/metadata.json
- Ghi lại metadata.json (backup metadata.backup.json) với trường mới:
  - effective_date: YYYY-MM-DD (nếu suy luận được)
  - effective_year: int (nếu suy luận được)

Cách chạy (PowerShell):
  conda activate LegalAdvisor
  python scripts\enrich_effective_year.py --dry-run  # chỉ xem thống kê
  python scripts\enrich_effective_year.py            # cập nhật metadata.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import xml.etree.ElementTree as ET


def _get_project_root() -> Path:
    env = os.getenv("LEGALADVISOR_ROOT")
    if env:
        p = Path(env).resolve()
        if p.exists():
            return p
    # scripts/ nằm ngay dưới root
    return Path(__file__).resolve().parent.parent


def _resolve_path(p: str, root: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


DATE_PATTERNS = [
    # 01/02/2021, 01-02-2021
    re.compile(r"(?P<d>\d{1,2})[\/-](?P<m>\d{1,2})[\/-](?P<y>\d{4})"),
    # ngày 01 tháng 02 năm 2021 (có thể có chữ thường/hoa)
    re.compile(r"ngày\s+(?P<d>\d{1,2})\s+tháng\s+(?P<m>\d{1,2})\s+năm\s+(?P<y>\d{4})", re.IGNORECASE),
    # năm 2015
    re.compile(r"năm\s+(?P<y>\d{4})", re.IGNORECASE),
]


def normalize_text(s: str) -> str:
    return (s or "").replace("_", " ").strip()


def parse_date_str(ds: str) -> Optional[Tuple[str, int]]:
    s = ds.strip()
    for pat in DATE_PATTERNS[:2]:
        m = pat.search(s)
        if m:
            try:
                d = int(m.group("d"))
                mth = int(m.group("m"))
                y = int(m.group("y"))
                dt = datetime(y, mth, d).strftime("%Y-%m-%d")
                return dt, y
            except Exception:
                continue
    # fallback: chỉ năm
    m = DATE_PATTERNS[2].search(s)
    if m:
        try:
            y = int(m.group("y"))
            return f"{y}-01-01", y
        except Exception:
            return None
    return None


def extract_effective_from_xml(xml_path: Path) -> Optional[Tuple[str, int]]:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return None

    # 1) Tìm trong thuộc tính hoặc thẻ có tên gợi ý
    KEY_HINTS = (
        "effective", "hieuluc", "ngay_hieu_luc", "ngayhieuluc",
        "validfrom", "valid_from", "ngaycohieuluc",
        "banhanh", "promulgation", "issuedate", "ngaybanhanh",
    )

    # Duyệt toàn bộ node (có thể chậm với file lớn nhưng an toàn)
    try:
        for elem in root.iter():
            tag = str(elem.tag).lower()
            if any(h in tag for h in KEY_HINTS):
                txt = (elem.text or "").strip()
                if txt:
                    got = parse_date_str(txt)
                    if got:
                        return got
            # Thuộc tính
            for k, v in elem.attrib.items():
                lk = str(k).lower()
                if any(h in lk for h in KEY_HINTS):
                    got = parse_date_str(str(v))
                    if got:
                        return got
    except Exception:
        pass

    # 2) Tìm trong text tổng quát
    try:
        text_blob = ET.tostring(root, encoding="unicode", method="text")
        text_blob = normalize_text(text_blob)
        for pat in DATE_PATTERNS:
            m = pat.search(text_blob)
            if m:
                # Ghép lại string nguyên thủy phù hợp
                span = m.group(0)
                got = parse_date_str(span)
                if got:
                    return got
    except Exception:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich effective_date/effective_year vào metadata.json")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ thống kê, không ghi đè metadata.json")
    parser.add_argument("--root", type=str, default="", help="Gốc dự án (mặc định auto từ đường dẫn script)")
    parser.add_argument("--raw-xml", type=str, default="data/raw/VNLegalText", help="Thư mục XML VNLegalText")
    parser.add_argument("--metadata", type=str, default="models/retrieval/metadata.json", help="Đường dẫn metadata.json")
    args = parser.parse_args()

    # Xác định root và đường dẫn
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    raw_xml_dir = _resolve_path(args.raw_xml, root)
    metadata_path = _resolve_path(args.metadata, root)

    if not metadata_path.exists():
        print(f"❌ Không tìm thấy metadata.json tại {metadata_path}")
        return

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Lỗi đọc metadata.json: {e}")
        return

    # Chuẩn bị map xml_stem -> (date, year)
    xml_map: Dict[str, Tuple[str, int]] = {}
    if raw_xml_dir.exists():
        xml_files = list(raw_xml_dir.rglob("*.xml"))
        for i, xp in enumerate(xml_files, 1):
            got = extract_effective_from_xml(xp)
            if got:
                xml_map[xp.stem.lower()] = got
            if i % 250 == 0:
                print(f"...đã quét {i}/{len(xml_files)} XML")
    else:
        print(f"⚠️  Không tìm thấy thư mục XML: {raw_xml_dir}")

    # Thử enrich từng bản ghi theo doc_file → stem
    updated = 0
    matched = 0
    for m in metadata:
        doc_file = str(m.get("doc_file") or "").strip()
        doc_title = normalize_text(str(m.get("doc_title") or ""))
        if not doc_file:
            continue
        stem = Path(doc_file).stem.lower()
        eff: Optional[Tuple[str, int]] = None

        # 1) match theo stem với xml
        eff = xml_map.get(stem)

        # 2) nếu chưa có, thử suy luận từ title (năm 2015)
        if not eff and doc_title:
            only_year = re.search(r"năm\s+(?P<y>\d{4})", doc_title, re.IGNORECASE)
            if only_year:
                try:
                    y = int(only_year.group("y"))
                    eff = (f"{y}-01-01", y)
                except Exception:
                    pass

        if eff:
            matched += 1
            date_str, year = eff
            cur_date = m.get("effective_date")
            cur_year = m.get("effective_year")
            if cur_date != date_str or cur_year != year:
                m["effective_date"] = date_str
                m["effective_year"] = int(year)
                updated += 1

    print(f"📊 Tổng bản ghi metadata: {len(metadata)}")
    print(f"🔎 Match được từ XML/title: {matched}")
    print(f"✏️  Cập nhật trường effective_*: {updated}")

    if not args.dry_run and updated > 0:
        # Backup
        backup_path = metadata_path.with_name("metadata.backup.json")
        backup_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        # Ghi đè
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"💾 Đã ghi metadata.json với trường effective_* (backup: {backup_path.name})")
    else:
        print("ℹ️  Dry-run hoặc không có thay đổi — không ghi file.")


if __name__ == "__main__":
    main()


