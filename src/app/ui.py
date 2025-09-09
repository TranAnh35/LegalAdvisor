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
import json
import time
from typing import Dict, Any

# Cấu hình trang
st.set_page_config(
    page_title="LegalAdvisor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_BASE_URL = "http://localhost:8000"

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

def ask_question(question: str, top_k: int = 3) -> Dict[str, Any]:
    """Gửi câu hỏi đến API"""
    try:
        payload = {"question": question, "top_k": top_k}
        response = requests.post(f"{API_BASE_URL}/ask", json=payload)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Lỗi kết nối API: {str(e)}")
        return None

def get_stats():
    """Lấy thống kê hệ thống"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        return response.json() if response.status_code == 200 else None
    except:
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
    if not health.get("rag_loaded", False):
        st.warning("⚠️ RAG system chưa được tải. Một số tính năng có thể không hoạt động.")
        st.info("💡 Kiểm tra GOOGLE_API_KEY và khởi động API bằng launcher:")
        st.code("$env:GOOGLE_API_KEY='YOUR_KEY'; python launcher.py")

    # Sidebar
    with st.sidebar:
        st.header("📊 Thông tin hệ thống")

        if health:
            st.success(f"✅ API: {health['status']}")
            st.info(f"RAG System: {'✅ Loaded' if health['rag_loaded'] else '❌ Not loaded'}")

        # Thống kê
        stats = get_stats()
        if stats and "error" not in stats:
            st.subheader("📈 Thống kê")
            st.metric("Tổng chunks", f"{stats.get('total_chunks', 0):,}")
            st.metric("Tổng từ", f"{stats.get('total_words', 0):,}")
            st.metric("Trung bình từ/chunk", f"{stats.get('avg_words_per_chunk', 0):.1f}")

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
            placeholder="Ví dụ: Quyền của công dân là gì? Thủ tục ly hôn như thế nào?"
        )

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

                if result:
                    # Hiển thị kết quả
                    st.success("✅ Đã tìm thấy câu trả lời!")

                    # Answer
                    st.subheader("💡 Câu trả lời")
                    st.write(result.get("answer", "Không có câu trả lời"))

                    # Confidence
                    confidence = result.get("confidence", 0)
                    st.metric("Độ tin cậy", f"{confidence:.3f}")

                    # Sources
                    if result.get("sources"):
                        st.subheader("📚 Nguồn tài liệu")

                        for i, source in enumerate(result["sources"], 1):
                            with st.expander(f"📄 Nguồn {i}: {source.get('title', source.get('doc_file', f'Nguồn {i}'))}"):
                                st.write(f"**Điểm số:** {source['score']:.4f}")
                                st.write(f"**File:** {source.get('title', source.get('doc_file', 'N/A'))}")

                                # Lấy nội dung chunk nếu cần
                                if st.button(f"Xem nội dung", key=f"source_{i}"):
                                    try:
                                        chunk_response = requests.get(f"{API_BASE_URL}/sources/{source['chunk_id']}")
                                        if chunk_response.status_code == 200:
                                            chunk_data = chunk_response.json()
                                            st.text_area(
                                                "Nội dung tài liệu:",
                                                chunk_data.get("content", "Không có nội dung"),
                                                height=200,
                                                disabled=True
                                            )
                                    except:
                                        st.error("Không thể tải nội dung")

                else:
                    st.error("❌ Không thể xử lý câu hỏi. Vui lòng thử lại.")

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
            if st.button(q, use_container_width=True):
                st.session_state.question = q

        # Copy từ session state
        if "question" in st.session_state:
            st.text_area("Câu hỏi được chọn:", st.session_state.question, disabled=True)

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
