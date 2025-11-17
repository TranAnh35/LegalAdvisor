#!/usr/bin/env python3
"""
Streamlit UI cho LegalAdvisor
"""

import sys
import os
import signal
import subprocess
import html

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import time
from typing import Dict, Any, Optional
from utils.law_registry import get_registry, normalize_act_code

# Cấu hình trang
st.set_page_config(
    page_title="LegalAdvisor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_BASE_URL = os.getenv("LEGALADVISOR_API_BASE_URL", "http://localhost:8000")

# Session state defaults
if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""
if "source_contents" not in st.session_state:
    st.session_state["source_contents"] = {}
if "source_errors" not in st.session_state:
    st.session_state["source_errors"] = {}
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "prefetched_ids" not in st.session_state:
    st.session_state["prefetched_ids"] = set()
# Thêm biến trạng thái hiện nguồn nào đang mở
if "active_source" not in st.session_state:
    st.session_state["active_source"] = None
if "did_warmup" not in st.session_state:
    st.session_state["did_warmup"] = False
if "citation_contents" not in st.session_state:
    st.session_state["citation_contents"] = {}
if "active_citation" not in st.session_state:
    st.session_state["active_citation"] = None
if "ref_article_contents" not in st.session_state:
    st.session_state["ref_article_contents"] = {}
if "active_ref_article" not in st.session_state:
    st.session_state["active_ref_article"] = None

st.markdown(
    """
    <style>
    .la-source-box {
        background-color: #f7f9ff;
        border: 1px solid #d7dcf4;
        border-radius: 10px;
        padding: 12px 16px;
        color: #1e2335;
        white-space: pre-wrap;
        line-height: 1.5;
        font-size: 0.95rem;
        font-family: 'Segoe UI', sans-serif;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
    }
    .la-source-box strong {
        color: #111421;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(ttl=15, show_spinner=False)
def cached_get(url: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None

def check_api_health(max_retries=1, timeout=5):
    """Kiểm tra trạng thái API (đã cache 15s) để tránh delay khi toggle UI."""
    data = cached_get(f"{API_BASE_URL}/health", timeout=timeout)
    if data is None and max_retries > 1:
        # Thử lại nhẹ nhàng (ít lần) nếu cache miss và request fail
        for _ in range(max_retries - 1):
            time.sleep(0.5)
            data = cached_get(f"{API_BASE_URL}/health", timeout=timeout)
            if data:
                break
    return data

def ask_question(question: str, top_k: int = 3) -> Optional[Dict[str, Any]]:
    """Gửi câu hỏi đến API"""
    try:
        payload = {"question": question, "top_k": top_k}
        response = requests.post(f"{API_BASE_URL}/ask", json=payload)
        try:
            data = response.json()
        except ValueError:
            data = {"message": response.text or "Unknown error"}

        data.setdefault("status_code", response.status_code)
        data["ok"] = response.status_code == 200
        return data
    except requests.RequestException as e:
        st.error(f"Lỗi kết nối API: {str(e)}")
        return None

def get_stats():
    """Lấy thống kê hệ thống"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        return response.json() if response.status_code == 200 else None
    except:
        return None


def get_health_details():
    data = cached_get(f"{API_BASE_URL}/health/details", timeout=5)
    return data


def trigger_reinitialize_rag():
    try:
        response = requests.post(f"{API_BASE_URL}/debug/reinit", timeout=10)
        try:
            data = response.json()
        except ValueError:
            data = {"message": response.text or "Unknown error"}
        data.setdefault("status_code", response.status_code)
        return data
    except Exception as e:
        st.error(f"Không thể gọi reinit: {e}")
        return None

def warmup_backend(llm: bool = False) -> Optional[Dict[str, Any]]:
    """Gọi warmup backend một lần để giảm độ trễ lần đầu.

    Không hiển thị spinner; im lặng nếu lỗi.
    """
    try:
        resp = requests.post(f"{API_BASE_URL}/warmup", params={"llm": str(bool(llm)).lower()}, timeout=8)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None

def main():
    """Main UI"""

    # Header
    st.title("⚖️ LegalAdvisor")
    st.markdown("**Hệ thống hỏi đáp pháp luật tiếng Việt**")
    st.markdown("Nội dung chỉ sử dụng cho mục đích tham khảo.")
    st.markdown("---")

    # Kiểm tra API
    health = check_api_health()

    if not health:
        st.error("⚠️ API không khả dụng. Vui lòng khởi động server trước.")
        st.code("python launcher.py")
        st.info("💡 Hoặc chạy API riêng lẻ:")
        st.code("python src/app/api.py")
        return

        # Kiểm tra RAG system
    health_details = get_health_details()

    # Warmup backend một lần trong mỗi session để giảm cold-start
    if health and not st.session_state.get("did_warmup", False):
        _ = warmup_backend(llm=False)
        st.session_state["did_warmup"] = True

    if not health.get("rag_loaded", False):
        st.warning("⚠️ RAG system chưa được tải. Một số tính năng có thể không hoạt động.")
        st.info("💡 Kiểm tra GOOGLE_API_KEY và khởi động API bằng launcher:")
        st.code("$env:GOOGLE_API_KEY='YOUR_KEY'; python launcher.py")
        if st.button("🔄 Thử khởi động lại RAG", use_container_width=True):
            reinit_result = trigger_reinitialize_rag()
            if reinit_result and reinit_result.get("status_code") == 200 and reinit_result.get("rag_loaded"):
                st.success("✅ Đã yêu cầu khởi động lại RAG thành công. Vui lòng đợi vài giây rồi thử lại.")
            elif reinit_result:
                message = reinit_result.get("message") or reinit_result.get("detail") or "Không thể khởi động lại RAG."
                st.error(f"❌ {message}")
                if reinit_result.get("rag_error"):
                    st.error(f"Lỗi: {reinit_result['rag_error']}")
            else:
                st.error("❌ Không thể khởi động lại RAG.")

    # Sidebar
    with st.sidebar:
        st.header("📊 Thông tin hệ thống")

        if health:
            st.success(f"✅ API: {health['status']}")
            st.info(f"RAG System: {'✅ Loaded' if health['rag_loaded'] else '❌ Not loaded'}")
            if health_details:
                st.caption(
                    f"🕒 Lần thử RAG cuối: {health_details.get('last_attempt_at') or 'Chưa có'}\n"
                    f"✅ Lần thành công cuối: {health_details.get('last_success_at') or 'Chưa có'}\n"
                    f"🔁 Số lần thử: {health_details.get('retry_attempts', 0)}"
                )

        # Ẩn thống kê chi tiết (không phù hợp người dùng cuối)

        st.markdown("---")
        st.markdown("### 🔗 Links")
        st.markdown("- [API Docs](/docs)")
        st.markdown("- [GitHub](https://github.com)")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("❓ Đặt câu hỏi pháp luật")

        # Input
        question = st.text_area(
            "Nhập câu hỏi của bạn:",
            height=100,
            placeholder="Ví dụ: Quyền của công dân là gì? Thủ tục ly hôn như thế nào?",
            value=st.session_state.get("question_input", ""),
            key="question_text_area"
        )
        st.session_state["question_input"] = question

        # Settings
        col_a, col_b = st.columns(2)
        with col_a:
            top_k = st.slider(
                "Số nguồn tài liệu",
                min_value=1,
                max_value=10,
                value=5,
                help="Tăng số nguồn để thu thập thêm ngữ cảnh khi cần",
            )
        with col_b:
            submit_button = st.button("🔍 Tìm câu trả lời", type="primary", use_container_width=True)

        # Submit
        if submit_button and question.strip():
            with st.spinner("🔄 Đang xử lý câu hỏi..."):
                result = ask_question(question.strip(), top_k)
                st.session_state["last_result"] = result
                # Chuẩn bị sẵn danh sách nguồn hiển thị để lần toggle không phải tính lại
                prepared_sources = []
                registry = get_registry()
                # Prefetch nội dung tài liệu để lần bấm 'Xem tài liệu' không bị trễ
                try:
                    if result and result.get("ok") and result.get("sources"):
                        for src in result["sources"]:
                            cid = src.get("chunk_id")
                            if cid is None:
                                continue
                            if cid in st.session_state["prefetched_ids"]:
                                continue
                            # Ưu tiên dùng content_full từ response để tránh gọi API riêng
                            content = (src.get("content_full") or src.get("content") or "")
                            if not content:
                                # Fallback an toàn: gọi endpoint /sources/{id} nếu thiếu content
                                try:
                                    resp = requests.get(f"{API_BASE_URL}/sources/{cid}", timeout=3)
                                    if resp.status_code == 200:
                                        data = resp.json()
                                        content = data.get("content", "") or ""
                                except Exception:
                                    content = ""
                            content = content.replace('_', ' ')
                            if content:
                                st.session_state["source_contents"][cid] = content
                                st.session_state["prefetched_ids"].add(cid)
                        # Tạo danh sách nguồn đã render sẵn
                        for i, source in enumerate(result["sources"], 1):
                            corpus_id = source.get('corpus_id') or f"Nguồn {i}"
                            type_ = source.get('type') or ""
                            number = source.get('number') or ""
                            year = source.get('year') or ""
                            suffix = source.get('suffix')
                            chunk_id = source.get('chunk_id')
                            score = source.get('score')
                            raw_code = (str(corpus_id).split('+')[0] if corpus_id else '').strip()
                            act_code_norm = normalize_act_code(raw_code) if raw_code else ""
                            info = registry.resolve_act(act_code_norm) if act_code_norm else None
                            is_digit_article = str(suffix or '').isdigit()
                            if info:
                                article_part = f"Điều {suffix}" if is_digit_article else "Điều ?"
                                loai = info.act_type or "Văn bản"
                                trich_yeu = (info.official_title or info.act_name or info.act_code or "").strip()
                                issuer = (info.issuer or "").strip()
                                main_line = f"{article_part} — {loai} — {trich_yeu}"
                                if issuer:
                                    main_line += f" — Được ban hành bởi {issuer}"
                                caption = f"Mã: `{act_code_norm}`"
                            else:
                                main_line = f"`{corpus_id}`"
                                caption = None
                            prepared_sources.append({
                                "chunk_id": chunk_id,
                                "main": main_line,
                                "caption": caption,
                                "score": score,
                            })
                        st.session_state["prepared_sources"] = prepared_sources
                except Exception:
                    pass

        # Display result (cached in session state)
        result = st.session_state.get("last_result")
        if result is None:
            pass
        elif result.get("ok"):
            # Hiển thị kết quả
            st.success("✅ Đã tìm thấy câu trả lời!")

            # Answer
            st.subheader("💡 Câu trả lời")
            st.write(result.get("answer", "Không có câu trả lời"))

            # Bỏ hiển thị độ tin cậy theo yêu cầu

            # Sources: ưu tiên danh sách tài liệu đã gộp (sources_grouped); fallback về sources chunk
            show_grouped = isinstance(result.get("sources_grouped"), list) and len(result.get("sources_grouped") or []) > 0
            sources_grouped = result.get("sources_grouped") or []
            if show_grouped or result.get("sources"):
                st.subheader("📚 Nguồn tài liệu")
                registry = get_registry()
                if show_grouped:
                    for idx, g in enumerate(sources_grouped, 1):
                        act_code = g.get("act_code") or ""
                        articles = g.get("articles") or []
                        info = None
                        try:
                            info = registry.resolve_act(act_code) if act_code else None
                        except Exception:
                            info = None
                        if info:
                            loai = info.act_type or "Văn bản"
                            trich_yeu = (info.official_title or info.act_name or info.act_code or "").strip()
                            issuer = (info.issuer or "").strip()
                            if articles:
                                arts_str = ", ".join(str(a) for a in articles)
                                article_part = f"Điều {arts_str}"
                            else:
                                article_part = "Văn bản"
                            main_line = f"{article_part} — {loai} — {trich_yeu}"
                            if issuer:
                                main_line += f" — Được ban hành bởi {issuer}"
                            caption = f"Mã: `{act_code}`"
                        else:
                            if articles:
                                arts_str = ", ".join(str(a) for a in articles)
                                main_line = f"Điều {arts_str} — `{act_code}`"
                            else:
                                main_line = f"`{act_code}`"
                            caption = None

                        st.markdown(f"**[{idx}]** {main_line}")
                        if caption:
                            st.caption(caption)

                        # Hiển thị nút xem nội dung theo từng Điều (đồng bộ với phần trích dẫn)
                        if articles:
                            for art in articles:
                                ref_key = f"ref:{act_code}:{int(art)}"
                                lbl = "Ẩn nội dung Điều {0}".format(int(art)) if st.session_state["active_ref_article"] == ref_key else "Xem nội dung Điều {0}".format(int(art))
                                if st.button(lbl, key=f"btn_ref_{ref_key}"):
                                    if st.session_state["active_ref_article"] == ref_key:
                                        st.session_state["active_ref_article"] = None
                                    else:
                                        if ref_key not in st.session_state["ref_article_contents"]:
                                            try:
                                                resp = requests.get(
                                                    f"{API_BASE_URL}/citations/content",
                                                    params={"act_code": act_code, "article": int(art)},
                                                    timeout=8,
                                                )
                                                if resp.status_code == 200:
                                                    data = resp.json()
                                                    merged = (data.get("merged_content") or "").replace('_', ' ').strip()
                                                    if merged:
                                                        st.session_state["ref_article_contents"][ref_key] = merged
                                                    else:
                                                        # Fallback giống phần trích dẫn: ghép từ danh sách items
                                                        items = data.get("items", []) or []
                                                        merged_local = "\n\n".join(
                                                            (itm.get("content") or "").replace('_', ' ').strip()
                                                            for itm in items if (itm.get("content") or "").strip()
                                                        ).strip()
                                                        st.session_state["ref_article_contents"][ref_key] = merged_local or "Không có nội dung"
                                                else:
                                                    st.session_state["ref_article_contents"][ref_key] = f"Không thể tải nội dung (HTTP {resp.status_code})"
                                            except Exception as e:
                                                st.session_state["ref_article_contents"][ref_key] = f"Lỗi khi tải nội dung: {e}"
                                        st.session_state["active_ref_article"] = ref_key
                                    st.rerun()
                                if st.session_state["active_ref_article"] == ref_key:
                                    content = st.session_state["ref_article_contents"].get(ref_key, "Không có nội dung")
                                    escaped = html.escape(content)
                                    st.markdown(
                                        f"<div class='la-source-box'>{escaped}</div>",
                                        unsafe_allow_html=True,
                                    )
                else:
                    # Fallback: hiển thị theo chunk như trước
                    prepared = st.session_state.get("prepared_sources")
                    if not prepared:
                        prepared = []
                        for idx, _ in enumerate(result["sources"], 1):
                            prepared.append({"main": f"Nguồn {idx}", "caption": None, "score": None, "chunk_id": None})
                    for i, p in enumerate(prepared, 1):
                        st.markdown(f"**[{i}]** {p['main']}")
                        if p.get("caption"):
                            st.caption(p["caption"]) 
                        chunk_id = p.get("chunk_id") or result["sources"][i-1].get("chunk_id")
                        label = "Ẩn nội dung" if st.session_state["active_source"] == chunk_id else "Xem nội dung tham khảo"
                        if st.button(label, key=f"btn_{chunk_id}"):
                            if st.session_state["active_source"] == chunk_id:
                                st.session_state["active_source"] = None
                            else:
                                st.session_state["active_source"] = chunk_id
                            st.rerun()
                        if st.session_state["active_source"] == chunk_id:
                            content = st.session_state["source_contents"].get(chunk_id, "Không có nội dung")
                            escaped = html.escape(content)
                            st.markdown(
                                f"<div class='la-source-box'>{escaped}</div>",
                                unsafe_allow_html=True,
                            )

            # Citations: hiển thị tách biệt, không làm giảm số lượng nguồn chính
            citations = result.get("citations") or []
            if isinstance(citations, list) and len(citations) > 0:
                st.subheader("📎 Tài liệu trích dẫn")
                registry = get_registry()
                for j, c in enumerate(citations, 1):
                    code = c.get("act_code") or ""
                    arts = c.get("articles") or []
                    supplemented_by = c.get("supplemented_by") or []
                    info = None
                    try:
                        norm_code = normalize_act_code(code)
                        if norm_code:
                            info = registry.resolve_act(norm_code)
                    except Exception:
                        info = None
                    if info:
                        loai = info.act_type or "Văn bản"
                        trich_yeu = (info.official_title or info.act_name or info.act_code or "").strip()
                        issuer = (info.issuer or "").strip()
                        # Đưa các Điều được trích dẫn (của văn bản này) lên đầu theo yêu cầu
                        try:
                            arts_sorted_for_title = sorted(set(int(a) for a in (arts or [])))
                            if arts_sorted_for_title:
                                arts_str = ", ".join(str(a) for a in arts_sorted_for_title)
                                header = f"Điều {arts_str} — {loai} — {trich_yeu}"
                            else:
                                header = f"{loai} — {trich_yeu}"
                        except Exception:
                            header = f"{loai} — {trich_yeu}"
                        if issuer:
                            header += f" — Ban hành bởi {issuer}"
                        # Nếu backend trả về danh sách tài liệu tham khảo có trích dẫn tới văn bản này -> hiển thị "Bổ sung cho ..."
                        try:
                            titles: list[str] = []
                            # supplemented_by là danh sách dict {act_code, articles} trong đó
                            # articles = các Điều của VĂN BẢN THAM CHIẾU (nơi đề cập đến trích dẫn)
                            for ref in supplemented_by:
                                ref_code = ref.get("act_code") if isinstance(ref, dict) else None
                                ref_arts = ref.get("articles") if isinstance(ref, dict) else []
                                if not ref_code:
                                    continue
                                nref = normalize_act_code(ref_code)
                                inf = registry.resolve_act(nref) if nref else None
                                # Build article string
                                art_str = ""
                                try:
                                    if isinstance(ref_arts, list) and len(ref_arts) > 0:
                                        art_str = "Điều " + ",".join(str(int(a)) for a in ref_arts)
                                except Exception:
                                    art_str = ""

                                if inf:
                                    loai_r = inf.act_type or "Văn bản"
                                    trich_yeu_r = (inf.official_title or inf.act_name or inf.act_code or "").strip()
                                    if art_str:
                                        titles.append(f"{art_str} - {loai_r} - {trich_yeu_r}")
                                    else:
                                        titles.append(f"{loai_r} - {trich_yeu_r}")
                                else:
                                    if art_str:
                                        titles.append(f"{art_str} - {nref}")
                                    else:
                                        titles.append(f"{nref}")
                            if titles:
                                # Gộp ngắn gọn; nếu nhiều thì nối bằng dấu phẩy
                                header += f" — Bổ sung cho {', '.join(titles)}"
                        except Exception:
                            pass
                        st.markdown(f"**[{j}]** {header}")
                        st.caption(f"Mã: `{norm_code}`")
                    else:
                        st.markdown(f"**[{j}]** `{code}`")

                    # Danh sách Điều được trích dẫn (nếu có)
                    if isinstance(arts, list) and len(arts) > 0:
                        try:
                            arts_sorted = sorted(set(int(a) for a in arts))
                        except Exception:
                            arts_sorted = arts
                        # Hiển thị từng Điều kèm nút xem nội dung (giống phần tham khảo)
                        for art in arts_sorted:
                            cit_key = f"{norm_code}:{int(art)}" if 'norm_code' in locals() and norm_code else f"{code}:{int(art)}"
                            label = "Ẩn nội dung" if st.session_state["active_citation"] == cit_key else f"Xem nội dung trích dẫn — Điều {art}"
                            if st.button(label, key=f"btn_cit_{cit_key}"):
                                if st.session_state["active_citation"] == cit_key:
                                    st.session_state["active_citation"] = None
                                else:
                                    # Prefetch nếu chưa có
                                    if cit_key not in st.session_state["citation_contents"]:
                                        try:
                                            resp = requests.get(
                                                f"{API_BASE_URL}/citations/content",
                                                params={"act_code": norm_code or code, "article": int(art)},
                                                timeout=8,
                                            )
                                            if resp.status_code == 200:
                                                data = resp.json()
                                                merged_api = (data.get("merged_content") or "").replace('_', ' ').strip()
                                                if merged_api:
                                                    st.session_state["citation_contents"][cit_key] = merged_api
                                                else:
                                                    items = data.get("items", []) or []
                                                    merged_local = "\n\n".join(
                                                        (itm.get("content") or "").replace('_', ' ')
                                                        for itm in items
                                                    ).strip()
                                                    st.session_state["citation_contents"][cit_key] = merged_local or "Không có nội dung"
                                            else:
                                                st.session_state["citation_contents"][cit_key] = f"Không thể tải nội dung (HTTP {resp.status_code})"
                                        except Exception as e:
                                            st.session_state["citation_contents"][cit_key] = f"Lỗi khi tải nội dung: {e}"
                                    st.session_state["active_citation"] = cit_key
                                st.rerun()
                            if st.session_state["active_citation"] == cit_key:
                                content = st.session_state["citation_contents"].get(cit_key, "Không có nội dung")
                                escaped = html.escape(content)
                                st.markdown(
                                    f"<div class='la-source-box'>{escaped}</div>",
                                    unsafe_allow_html=True,
                                )

        else:
            detail = result.get("detail") or result.get("message") or result.get("error")
            if isinstance(detail, dict):
                primary_msg = detail.get("message") or detail.get("error") or "Không thể xử lý câu hỏi."
                hint = detail.get("hint")
                retry_after = detail.get("retry_after") or detail.get("retry_after_seconds")
            else:
                primary_msg = detail or "Không thể xử lý câu hỏi."
                hint = None
                retry_after = None

            status_code = result.get("status_code")
            if status_code == 429:
                st.error(f"❌ {primary_msg}")
                if retry_after:
                    st.info(f"Vui lòng thử lại sau khoảng {retry_after} giây.")
            else:
                st.error(f"❌ {primary_msg}")
            if hint:
                st.info(f"💡 {hint}")

            with st.expander("Chi tiết lỗi"):
                st.json(result)

    with col2:
        st.subheader("📝 Câu hỏi mẫu")

        sample_questions = [
            "Quyền của công dân là gì?",
            "Thủ tục ly hôn như thế nào?",
            "Quy định về lao động cho người Việt Nam?",
            "Phạt vi phạm giao thông như thế nào?",
            "Quyền sở hữu trí tuệ được bảo vệ ra sao?"
        ]

        for q in sample_questions:
            if st.button(q, use_container_width=True, key=f"sample_{q}"):
                st.session_state["question_input"] = q
                # Tự động submit câu hỏi mẫu
                with st.spinner("🔄 Đang xử lý câu hỏi..."):
                    result = ask_question(q.strip(), top_k=top_k)
                    st.session_state["last_result"] = result
                    # Prefetch nội dung tài liệu (ưu tiên content_full từ response)
                    try:
                        if result and result.get("ok") and result.get("sources"):
                            for src in result["sources"]:
                                cid = src.get("chunk_id")
                                if cid is None:
                                    continue
                                if cid in st.session_state["prefetched_ids"]:
                                    continue
                                content = (src.get("content_full") or src.get("content") or "")
                                if not content:
                                    try:
                                        resp = requests.get(f"{API_BASE_URL}/sources/{cid}", timeout=3)
                                        if resp.status_code == 200:
                                            data = resp.json()
                                            content = data.get("content", "") or ""
                                    except Exception:
                                        content = ""
                                content = content.replace('_', ' ')
                                if content:
                                    st.session_state["source_contents"][cid] = content
                                    st.session_state["prefetched_ids"].add(cid)
                            # Chuẩn bị danh sách nguồn render sẵn cho mẫu
                            prepared_sources = []
                            registry = get_registry()
                            for i, source in enumerate(result["sources"], 1):
                                corpus_id = source.get('corpus_id') or f"Nguồn {i}"
                                suffix = source.get('suffix')
                                chunk_id = source.get('chunk_id')
                                score = source.get('score')
                                raw_code = (str(corpus_id).split('+')[0] if corpus_id else '').strip()
                                act_code_norm = normalize_act_code(raw_code) if raw_code else ""
                                info = registry.resolve_act(act_code_norm) if act_code_norm else None
                                is_digit_article = str(suffix or '').isdigit()
                                if info:
                                    article_part = f"Điều {suffix}" if is_digit_article else "Điều ?"
                                    loai = info.act_type or "Văn bản"
                                    trich_yeu = (info.official_title or info.act_name or info.act_code or "").strip()
                                    issuer = (info.issuer or "").strip()
                                    main_line = f"{article_part} — {loai} — {trich_yeu}"
                                    if issuer:
                                        main_line += f" — Được ban hành bởi {issuer}"
                                    caption = f"Mã: `{act_code_norm}`"
                                else:
                                    main_line = f"`{corpus_id}`"
                                    caption = None
                                prepared_sources.append({
                                    "chunk_id": chunk_id,
                                    "main": main_line,
                                    "caption": caption,
                                    "score": score,
                                })
                            st.session_state["prepared_sources"] = prepared_sources
                    except Exception:
                        pass
                # Force UI rerun để hiển thị kết quả
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("*LegalAdvisor v1.0 - Hệ thống hỏi đáp pháp luật sử dụng AI*")

def run_ui_server(host="localhost", port=8501):
    """Chạy Streamlit UI server trực tiếp"""
    print("🚀 Khởi động LegalAdvisor UI server...")
    print(f"📡 Host: {host}")
    print(f"🔌 Port: {port}")
    print("🛑 Nhấn Ctrl+C để dừng server")
    print("=" * 50)

    try:
        # Chạy streamlit với subprocess nhưng có signal handling tốt hơn
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            __file__,
            "--server.address", host,
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.runOnSave", "false"
        ]

        # Chạy subprocess với proper signal handling
        process = subprocess.Popen(cmd)

        def signal_handler(signum, frame):
            print(f"\n🛑 Nhận tín hiệu {signum}, đang dừng UI server...")
            process.terminate()
            try:
                process.wait(timeout=5)
                print("✅ UI server đã dừng!")
            except subprocess.TimeoutExpired:
                process.kill()
                print("✅ UI server đã force kill!")
            sys.exit(0)

        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Wait for process
        process.wait()

    except Exception as e:
        print(f"❌ Lỗi khi khởi động UI server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="LegalAdvisor UI Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8501, help="Port to bind to")

    args = parser.parse_args()

    # Chạy UI
    if len(sys.argv) > 1:
        # Nếu có arguments, chạy server mode
        run_ui_server(host=args.host, port=args.port)
    else:
        # Nếu không có arguments, chạy UI trực tiếp
        main()
