#!/usr/bin/env python3
"""
Streamlit UI cho LegalAdvisor
"""

import sys
import os
import signal
import subprocess

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import time
from typing import Dict, Any, Optional

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
if "auto_submit" not in st.session_state:
    st.session_state["auto_submit"] = False

def check_api_health(max_retries=3, timeout=5):
    """Kiểm tra trạng thái API với retry"""
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                st.warning(f"⚠️ API trả về status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            if attempt == max_retries - 1:  # Chỉ hiển thị lỗi ở lần thử cuối
                st.error(f"❌ Không thể kết nối API (timeout: {timeout}s)")
        except Exception as e:
            if attempt == max_retries - 1:  # Chỉ hiển thị lỗi ở lần thử cuối
                st.error(f"❌ Lỗi kiểm tra API: {e}")

        if attempt < max_retries - 1:
            time.sleep(2)

    return None

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
    try:
        response = requests.get(f"{API_BASE_URL}/health/details", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


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

def main():
    """Main UI"""

    # Header
    st.title("⚖️ LegalAdvisor")
    st.markdown("**Hệ thống hỏi đáp pháp luật tiếng Việt**")
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
            top_k = st.slider("Số nguồn tài liệu", 1, 5, 3)
        with col_b:
            submit_button = st.button("🔍 Tìm câu trả lời", type="primary", use_container_width=True)

        # Submit
        if submit_button and question.strip():
            with st.spinner("🔄 Đang xử lý câu hỏi..."):
                result = ask_question(question.strip(), top_k)
                st.session_state["last_result"] = result
                # Prefetch nội dung tài liệu để lần bấm 'Xem tài liệu' không bị trễ
                try:
                    if result and result.get("ok") and result.get("sources"):
                        for src in result["sources"]:
                            cid = src.get("chunk_id")
                            if cid is None:
                                continue
                            if cid in st.session_state["prefetched_ids"]:
                                continue
                            resp = requests.get(f"{API_BASE_URL}/sources/{cid}", timeout=3)
                            if resp.status_code == 200:
                                data = resp.json()
                                content = (data.get("content", "") or "").replace('_', ' ')
                                st.session_state["source_contents"][cid] = content
                                st.session_state["prefetched_ids"].add(cid)
                except Exception:
                    pass

        # Tự động submit nếu chọn câu hỏi mẫu
        if st.session_state.get("auto_submit") and st.session_state.get("question_input", "").strip():
            result = ask_question(st.session_state["question_input"].strip(), top_k)
            st.session_state["last_result"] = result
            # Prefetch nội dung tài liệu
            try:
                if result and result.get("ok") and result.get("sources"):
                    for src in result["sources"]:
                        cid = src.get("chunk_id")
                        if cid is None:
                            continue
                        if cid in st.session_state["prefetched_ids"]:
                            continue
                        resp = requests.get(f"{API_BASE_URL}/sources/{cid}", timeout=3)
                        if resp.status_code == 200:
                            data = resp.json()
                            content = (data.get("content", "") or "").replace('_', ' ')
                            st.session_state["source_contents"][cid] = content
                            st.session_state["prefetched_ids"].add(cid)
            except Exception:
                pass
            st.session_state["auto_submit"] = False

        # Luôn hiển thị last_result nếu có, để tránh reset khi bấm nút khác
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

            # Sources: hiển thị tiêu đề gọn + nút xem nội dung theo nhu cầu
            if result.get("sources"):
                st.subheader("📚 Nguồn tài liệu")
                for i, source in enumerate(result["sources"], 1):
                    corpus_id = source.get('corpus_id') or f"Nguồn {i}"
                    type_ = source.get('type') or ""
                    number = source.get('number') or ""
                    year = source.get('year') or ""
                    suffix = source.get('suffix')
                    dieu = f" - Điều {suffix}" if str(suffix or '').isdigit() else ""
                    score = source.get('score')
                    chunk_id = source.get('chunk_id')

                    st.markdown(f"**[{i}]** `{corpus_id}` ({type_} - {number} - {year}{dieu})")
                    if isinstance(score, (int, float)):
                        st.caption(f"Điểm: {score:.4f}")
                    # Nút: khi bấm thì chỉ mở/đóng đúng nguồn này, không gọi API (đã prefetch)
                    label = "Ẩn nội dung" if st.session_state["active_source"] == chunk_id else "Xem nội dung tham khảo"
                    if st.button(label, key=f"btn_{chunk_id}"):
                        if st.session_state["active_source"] == chunk_id:
                            st.session_state["active_source"] = None
                        else:
                            st.session_state["active_source"] = chunk_id
                    # Chỉ hiện nội dung nếu được mở
                    if st.session_state["active_source"] == chunk_id:
                        content = st.session_state["source_contents"].get(chunk_id, "Không có nội dung")
                        st.text_area(
                            "Nội dung tài liệu:",
                            content,
                            height=200,
                            disabled=True
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
                st.session_state["auto_submit"] = True

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
