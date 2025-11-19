import pytest
from src.utils.law_registry import LawRegistry, ActInfo, normalize_act_code


def test_normalize_act_code_pytest():
    assert normalize_act_code("159/2020/nd-cp") == "159/2020/NĐ-CP"
    assert normalize_act_code("47/2011/tt-bca") == "47/2011/TT-BCA"
    assert normalize_act_code("01/2015/TT-BTP") == "01/2015/TT-BTP"


def test_resolve_display_fallback_pytest():
    reg = LawRegistry({})
    disp = reg.resolve_display("104/2016/qh13", article=8)
    assert "104/2016/QH13" in disp
    assert "Điều 8" in disp


def test_convert_entry_from_crawler_schema_pytest():
    entry = {
        "url": "https://vanban.chinhphu.vn/?pageid=27160&docid=179555",
        "so_hieu": "01/2015/TT-BTP",
        "loai_van_ban": "Thông tư",
        "co_quan_ban_hanh": "Bộ Tư pháp",
        "trich_yeu": "Hướng dẫn về nghiệp vụ thực hiện chức năng, nhiệm vụ, quyền hạn của các tổ chức pháp chế",
    }
    act_info = LawRegistry._convert_entry(entry)
    assert act_info is not None
    assert act_info.act_code == "01/2015/TT-BTP"
    assert (act_info.official_title or "").strip() != ""
