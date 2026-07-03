#!/usr/bin/env python3
"""Launcher đơn giản cho LegalAdvisor Mini."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


api_process: subprocess.Popen | None = None
ui_process: subprocess.Popen | None = None


def configure_console_encoding() -> None:
    """Dùng UTF-8 khi Windows console đang ở code page hẹp."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def print_rebuild_steps() -> None:
    print("   Pipeline chuẩn bị dữ liệu/index:")
    print("      python scripts/dataset/download.py        # nếu chưa có raw corpus")
    print("      python scripts/zalo_legal_preprocess.py")
    print("      python src/retrieval/build_index.py")
    print("      python scripts/utils/build_law_registry.py")


def check_requirements() -> bool:
    """Kiểm tra dữ liệu, index và cấu hình tối thiểu trước khi chạy app."""
    print("Kiểm tra yêu cầu hệ thống...")

    for dir_path in ("data/processed", "models"):
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"Đã tạo thư mục: {dir_path}")

    chunks_path = Path("data/processed/zalo-legal/chunks_schema.jsonl")
    if not chunks_path.exists():
        print("Chưa tìm thấy dữ liệu đã tiền xử lý:")
        print(f"   - {chunks_path}")
        print_rebuild_steps()
        return False

    retrieval_dir = Path("models/retrieval")
    index_dir = retrieval_dir / "index"
    index_path = index_dir / "chunks_index.faiss"
    info_path = index_dir / "model_info.json"
    metadata_path = index_dir / "metadata.json"

    old_index_path = retrieval_dir / "faiss_index.bin"
    old_info_path = retrieval_dir / "model_info.json"
    old_metadata_path = retrieval_dir / "metadata.json"

    has_new_index = index_path.exists() and info_path.exists()
    has_old_index = old_index_path.exists() and old_info_path.exists()

    if not has_new_index and not has_old_index:
        print("Thiếu FAISS index hoặc model_info.json.")
        print_rebuild_steps()
        return False

    selected_info = info_path if has_new_index else old_info_path
    selected_metadata = metadata_path if has_new_index else old_metadata_path
    location = "models/retrieval/index" if has_new_index else "models/retrieval legacy"

    try:
        model_info = json.loads(selected_info.read_text(encoding="utf-8"))
        model_name = model_info.get("model_name") or model_info.get("base_model") or "unknown"
        dim = model_info.get("embedding_dim", "unknown")
        chunks = model_info.get("num_chunks", "unknown")
        segments = model_info.get("num_segments", "unknown")
        print(f"Retrieval index: {location}")
        print(f"Model: {model_name} | dim={dim} | chunks={chunks} | segments={segments}")
    except Exception:
        print("Không đọc được model_info.json để hiển thị thông tin retrieval.")

    if not selected_metadata.exists():
        print("Chưa tìm thấy metadata.json; việc này chủ yếu ảnh hưởng endpoint thống kê.")

    print("Kiểm tra hoàn tất.")
    return True


def detect_gpu() -> bool:
    env_override = os.environ.get("LEGALADVISOR_USE_GPU")
    if env_override is not None:
        use_gpu = env_override.lower() in ("1", "true", "yes", "on")
        print("LEGALADVISOR_USE_GPU: bật GPU" if use_gpu else "LEGALADVISOR_USE_GPU: dùng CPU")
        return use_gpu

    try:
        import torch

        if torch.cuda.is_available():
            print("Phát hiện GPU, sẽ dùng GPU cho API.")
            return True
    except Exception:
        pass

    print("Không dùng GPU, chạy CPU.")
    return False


def start_api_server(use_gpu: bool = False) -> bool:
    """Khởi động FastAPI backend."""
    global api_process

    try:
        load_dotenv()
    except Exception:
        pass

    env = os.environ.copy()
    groq_key = (env.get("GROQ_API_KEY") or "").strip()
    if not groq_key or groq_key.lower().startswith("your_"):
        print("GROQ_API_KEY chưa được thiết lập. Hãy tạo .env từ .env.sample.")
        return False

    env["RAG_ENGINE"] = "groq"
    env["LEGALADVISOR_USE_GPU"] = "1" if use_gpu else "0"

    cmd = [
        sys.executable,
        "-m",
        "src.app.api",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    if use_gpu:
        cmd.append("--use-gpu")

    print("Khởi động API server...")
    api_process = subprocess.Popen(cmd, env=env)
    print(f"API server PID: {api_process.pid}")
    return True


def start_ui_server() -> bool:
    """Khởi động Streamlit UI."""
    global ui_process

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/app/ui.py",
        "--server.address",
        "localhost",
        "--server.port",
        "8501",
        "--browser.gatherUsageStats",
        "false",
        "--server.headless",
        "true",
    ]

    print("Khởi động UI server...")
    ui_process = subprocess.Popen(cmd, env=os.environ.copy())
    print(f"UI server PID: {ui_process.pid}")
    return True


def stop_servers() -> None:
    """Dừng API và UI nếu đang chạy."""
    global api_process, ui_process

    print("\nĐang dừng servers...")
    for name, process in (("API", api_process), ("UI", ui_process)):
        if process is None:
            continue
        try:
            process.terminate()
            process.wait(timeout=5)
            print(f"{name} server đã dừng.")
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"{name} server đã bị buộc dừng.")
        except Exception as exc:
            print(f"Lỗi khi dừng {name}: {exc}")


def signal_handler(signum, frame) -> None:  # type: ignore[no-untyped-def]
    print(f"\nNhận tín hiệu {signum}, đang tắt hệ thống...")
    stop_servers()
    sys.exit(0)


def wait_for_api(max_wait_seconds: int = 60) -> bool:
    """Đợi API healthcheck sẵn sàng."""
    import requests

    start_time = time.time()
    attempt = 0
    while time.time() - start_time < max_wait_seconds:
        attempt += 1
        if api_process and api_process.poll() is not None:
            print("API server đã dừng trong lúc khởi động.")
            print_rebuild_steps()
            return False

        try:
            response = requests.get("http://localhost:8000/health", timeout=3)
            if response.status_code == 200:
                print("API server đã sẵn sàng.")
                return True
            print(f"/health trả về {response.status_code} (attempt {attempt})")
        except Exception:
            pass
        time.sleep(1)

    print("Không thể kết nối API trong 60 giây.")
    print("Hãy kiểm tra GROQ_API_KEY, models/retrieval, data/processed và log API.")
    print_rebuild_steps()
    return False


def main() -> None:
    configure_console_encoding()
    try:
        load_dotenv()
    except Exception:
        pass

    print("\n" + "=" * 50)
    print("LegalAdvisor - Hệ thống hỗ trợ pháp lý")
    print("=" * 50 + "\n")

    use_gpu = detect_gpu()
    print("Sử dụng Groq cho text generation.")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not check_requirements():
        sys.exit(1)

    try:
        if not start_api_server(use_gpu=use_gpu):
            sys.exit(1)
        if not wait_for_api():
            stop_servers()
            sys.exit(1)
        if not start_ui_server():
            stop_servers()
            sys.exit(1)

        print("\nHệ thống đã sẵn sàng.")
        print("Web UI: http://localhost:8501")
        print("API: http://localhost:8000")
        print("API Docs: http://localhost:8000/docs")
        print("Nhấn Ctrl+C để dừng hệ thống.")

        while True:
            if api_process and api_process.poll() is not None:
                print("API server đã dừng bất ngờ.")
                break
            if ui_process and ui_process.poll() is not None:
                print("UI server đã dừng bất ngờ.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_servers()


if __name__ == "__main__":
    main()