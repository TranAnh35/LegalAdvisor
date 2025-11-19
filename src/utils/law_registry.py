"""Law Registry Resolver

Chức năng:
- Tải registry văn bản pháp luật từ JSON.
- Chuẩn hoá `act_code`.
- Resolve thông tin hiển thị thân thiện ở cấp Điều.

Hai dạng nguồn dữ liệu hiện có:
1. Seed chuẩn (schema mục tiêu): mỗi entry dạng:
   {
       "act_code": "91/2015/QH13",
       "act_name": "Bộ luật Dân sự",
       "year": 2015,
       "act_type": "BO_LUAT",
       "issuer": "QH13",
       "official_title": "Bộ luật Dân sự năm 2015"
   }
2. Dữ liệu thô crawler (hiện tại file `data/registry/law_registry.json` sinh bởi script crawl):
   {
       "url": "...",
       "so_hieu": "01/2015/TT-BTP",
       "loai_van_ban": "Thông tư",
       "co_quan_ban_hanh": "Bộ Tư pháp",
       "trich_yeu": "Hướng dẫn ..."
   }

Module này cần *tự động chuyển đổi* dữ liệu crawler sang schema sử dụng nội bộ nếu thiếu trường.

Sử dụng:
    from src.utils.law_registry import get_registry
    registry = get_registry()
    info = registry.resolve_act("159/2020/NĐ-CP")
    display = registry.resolve_display("159/2020/NĐ-CP", article=13)

Ghi log khi missing mapping để phục vụ bổ sung seed.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Any

DEFAULT_REGISTRY_PATH = os.path.join("data", "registry", "law_registry.json")
ENV_REGISTRY_PATH = os.getenv("LAW_REGISTRY_PATH")  # Cho phép override qua ENV

# Chuẩn hoá type phổ biến
TYPE_MAP = {
    'nđ-cp': 'NĐ-CP', 'nd-cp': 'NĐ-CP',
    'tt-bca': 'TT-BCA', 'tt-bnn': 'TT-BNN', 'tt-btc': 'TT-BTC', 'tt-btp': 'TT-BTP', 'tt-bnv': 'TT-BNV',
    'qh13': 'QH13', 'qh14': 'QH14', 'qh15': 'QH15'
}


def normalize_act_code(act_code: str) -> str:
    """Chuẩn hoá act_code dạng NUMBER/YEAR/TYPE.

    Nếu chuỗi không khớp cấu trúc hoặc rỗng -> trả về viết hoa đơn giản.
    """
    act_code = (act_code or "").strip()
    if not act_code:
        return ""  # Cho phép rỗng ở giai đoạn chuyển đổi; sẽ bỏ qua entry
    parts = act_code.split('/')
    if len(parts) < 3:
        return act_code.upper()
    number = parts[0]
    year = parts[1]
    type_raw = parts[2]
    type_norm = TYPE_MAP.get(type_raw.lower(), type_raw.upper())
    return f"{number}/{year}/{type_norm}"


@dataclass
class ActInfo:
    act_code: str
    act_name: str
    official_title: str
    year: Optional[int] = None
    act_type: Optional[str] = None
    issuer: Optional[str] = None

    def display(self, article: Optional[int] = None) -> str:
        base = self.act_name or self.official_title or self.act_code
        if article is not None:
            return f"{base} — Điều {article}" if self.act_name else f"{self.act_code} — Điều {article}"
        return base


class LawRegistry:
    def __init__(self, data: Dict[str, ActInfo]):
        self._data = data
        self._miss_logged = set()

    @staticmethod
    def _convert_entry(entry: Dict[str, Any]) -> Optional[ActInfo]:
        """Chuyển đổi một entry (seed chuẩn hoặc crawler thô) về ActInfo.

        - Ưu tiên dùng 'act_code'; nếu thiếu dùng 'so_hieu'.
        - official_title ưu tiên 'official_title'; fallback 'trich_yeu'.
        - year cố gắng parse từ act_code (thành phần thứ 2 nếu là số) nếu không có.
        - act_type từ 'act_type' hoặc 'loai_van_ban'.
        - issuer từ 'issuer' hoặc 'co_quan_ban_hanh'.
        """
        raw_code = entry.get('act_code') or entry.get('so_hieu') or ''
        if not raw_code:
            return None
        act_code_norm = normalize_act_code(raw_code)
        if not act_code_norm:
            return None

        official_title = entry.get('official_title') or entry.get('trich_yeu') or ''
        act_name = entry.get('act_name') or ''
        # Heuristic: nếu loại văn bản là 'Luật' và act_name trống nhưng official_title viết HOA toàn bộ -> đặt act_name từ official_title chuẩn hoá
        loai = entry.get('loai_van_ban') or entry.get('act_type') or ''
        if not act_name and loai.lower() == 'luật':
            if official_title and official_title.upper() == official_title:
                act_name = official_title.title()

        # Parse year
        year = entry.get('year')
        if year is None:
            parts = act_code_norm.split('/')
            if len(parts) >= 2 and parts[1].isdigit():
                try:
                    year = int(parts[1])
                except ValueError:
                    year = None

        act_type = entry.get('act_type') or loai or None
        issuer = entry.get('issuer') or entry.get('co_quan_ban_hanh') or None

        return ActInfo(
            act_code=act_code_norm,
            act_name=act_name,
            official_title=official_title,
            year=year,
            act_type=act_type,
            issuer=issuer,
        )

    @classmethod
    def load(cls, path: str = None) -> 'LawRegistry':
        path = path or ENV_REGISTRY_PATH or DEFAULT_REGISTRY_PATH
        if not os.path.exists(path):
            print(f"⚠️ Không tìm thấy registry: {path}. Khởi tạo rỗng.")
            return cls({})
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
        except Exception as e:
            print(f"⚠️ Lỗi đọc registry: {e}. Khởi tạo rỗng.")
            return cls({})
        if not isinstance(raw, list):
            print("⚠️ Registry JSON không phải list. Khởi tạo rỗng.")
            return cls({})
        data: Dict[str, ActInfo] = {}
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            act_info = cls._convert_entry(entry)
            if not act_info:
                continue
            # Ghi đè nếu trùng mã để ưu tiên bản seed đầy đủ (heuristic: bản có act_name dài hơn / official_title dài hơn)
            existing = data.get(act_info.act_code)
            if existing:
                def _score(ai: ActInfo) -> int:
                    return (len(ai.act_name or '') * 2) + len(ai.official_title or '')
                if _score(act_info) <= _score(existing):
                    continue
            data[act_info.act_code] = act_info
        return cls(data)

    def resolve_act(self, act_code: str) -> Optional[ActInfo]:
        code_norm = normalize_act_code(act_code)
        info = self._data.get(code_norm)
        if not info and code_norm not in self._miss_logged:
            print(f"[LawRegistry] Miss mapping: {code_norm}")
            self._miss_logged.add(code_norm)
        return info

    def resolve_display(self, act_code: str, article: Optional[int] = None) -> str:
        info = self.resolve_act(act_code)
        if info:
            return info.display(article=article)
        # Fallback nếu chưa mapping
        return f"{normalize_act_code(act_code)} — Điều {article}" if article is not None else normalize_act_code(act_code)


# Tiện ích nhanh cho các module khác
_registry_singleton: Optional[LawRegistry] = None

def get_registry(path: str = None) -> LawRegistry:
    global _registry_singleton
    if _registry_singleton is None:
        _registry_singleton = LawRegistry.load(path)
    return _registry_singleton


if __name__ == '__main__':
    reg = get_registry()
    sample = ["159/2020/nđ-cp", "47/2011/tt-bca", "100/2023/qh14", "01/2015/tt-btp"]
    for ac in sample:
        print(reg.resolve_display(ac, article=1))
