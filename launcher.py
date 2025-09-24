#!/usr/bin/env python3
"""
Launcher đơn giản cho LegalAdvisor
"""

import sys
import os
import signal
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv
import json

def check_requirements():
    """Kiểm tra các yêu cầu cơ bản"""
    print("🔍 Kiểm tra yêu cầu hệ thống...")

    # Kiểm tra Python version
    if sys.version_info < (3, 8):
        print(f"❌ Cần Python >= 3.8, hiện tại: {sys.version}")
        return False

    # Kiểm tra GPU và hiển thị thông tin
    print("🔥 Kiểm tra GPU support...")
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} ({gpu_count} GPU(s))")
            print("   🚀 LegalAdvisor will use GPU acceleration for better performance!")
        else:
            print("⚠️  GPU not available - using CPU mode")
            print("   💡 Run 'python check_gpu.py' for GPU setup instructions")
    except ImportError:
        print("⚠️  PyTorch not found - GPU check skipped")

    # Kiểm tra thư mục cần thiết
    required_dirs = ["data/processed", "models"]
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"⚠️ Thiếu thư mục: {dir_path}")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✅ Đã tạo thư mục: {dir_path}")

    # Kiểm tra dataset
    dataset_files = [
        # Chuẩn: dữ liệu đã xử lý ở SQLite/Parquet
        "data/processed/smart_chunks_stable.db",
        "data/processed/smart_chunks_stable.parquet",
        "data/raw/ViQuAD/train.json"
    ]

    missing_datasets = []
    for file_path in dataset_files:
        if not Path(file_path).exists():
            missing_datasets.append(file_path)

    if missing_datasets:
        print("ℹ️  Một số datasets chưa có (không bắt buộc để chạy launcher):")
        for missing in missing_datasets:
            print(f"   - {missing}")
        print("   → Có thể tạo riêng khi cần.")

    # Kiểm tra mô hình retrieval đã sẵn sàng chưa
    retrieval_dir = Path("models/retrieval")
    index_path = retrieval_dir / "faiss_index.bin"
    meta_path = retrieval_dir / "metadata.json"
    info_path = retrieval_dir / "model_info.json"
    if not retrieval_dir.exists() or not index_path.exists() or not meta_path.exists() or not info_path.exists():
        print("⚠️  Thiếu mô hình retrieval (FAISS/metadata/model_info).")
        print("   💡 Vui lòng chạy riêng bước build index trước khi launch:")
        print("      conda activate LegalAdvisor")
        print("      python src/retrieval/build_index.py")
    else:
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                mi = json.load(f)
            model_name = mi.get('model_name')
            dim = mi.get('embedding_dim')
            metric = mi.get('metric_type', 'ip')
            pooling = mi.get('pooling', 'unknown')
            print(f"🔧 Retrieval model: {model_name} | dim={dim} | metric={metric} | pooling={pooling}")
        except Exception:
            print("ℹ️  Không đọc được model_info.json để hiển thị thông tin mô hình.")

    print("✅ Kiểm tra hoàn thành!")
    return True

# Global variables
api_process = None
ui_process = None

def start_api_server(use_gpu=False):
    """Khởi động API server với subprocess
    
    Args:
        use_gpu (bool): Có sử dụng GPU hay không
    """
    global api_process

    try:
        print("🚀 Khởi động API server...")
        cmd = [
            sys.executable,
            "-m", "src.app.api",
            "--host", "0.0.0.0",
            "--port", "8000"
        ]
        
        # Thêm tùy chọn --use-gpu nếu được yêu cầu
        if use_gpu:
            cmd.append("--use-gpu")
            print("   🚀 Chế độ GPU đã được kích hoạt")
        else:
            print("   ⚡ Chế độ CPU")

        # Nạp .env để lấy GOOGLE_API_KEY nếu có
        try:
            load_dotenv()
        except Exception:
            pass

        # Bắt buộc sử dụng Gemini: yêu cầu GOOGLE_API_KEY và đặt RAG_ENGINE=gemini
        env = os.environ.copy()
        if not env.get("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY chưa được thiết lập. Vui lòng tạo .env và đặt GOOGLE_API_KEY.")
        env["RAG_ENGINE"] = "gemini"
        # Truyền hint sử dụng GPU cho các tiến trình con
        env["LEGALADVISOR_USE_GPU"] = "1" if use_gpu else "0"
        api_process = subprocess.Popen(cmd, env=env)
        print("✅ API server đã khởi động (PID: {})".format(api_process.pid))

    except Exception as e:
        print(f"❌ Lỗi khởi động API: {e}")
        return False

    return True

def start_ui_server():
    """Khởi động UI server bằng streamlit run để tránh cảnh báo bare mode"""
    global ui_process

    try:
        print("🚀 Khởi động UI server (streamlit run)...")
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "src/app/ui.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "true"
        ]

        env = os.environ.copy()
        ui_process = subprocess.Popen(cmd, env=env)
        print("✅ UI server đã khởi động (PID: {})".format(ui_process.pid))

    except Exception as e:
        print(f"❌ Lỗi khởi động UI: {e}")
        return False

    return True

def stop_servers():
    """Dừng tất cả servers"""
    global api_process, ui_process

    print("\n🔄 Đang dừng servers...")

    # Dừng API process
    if api_process:
        try:
            api_process.terminate()
            api_process.wait(timeout=5)
            print("✅ API server stopped")
        except subprocess.TimeoutExpired:
            api_process.kill()
            print("✅ API server force killed")
        except Exception as e:
            print(f"⚠️ Lỗi dừng API: {e}")

    # Dừng UI process
    if ui_process:
        try:
            ui_process.terminate()
            ui_process.wait(timeout=5)
            print("✅ UI server stopped")
        except subprocess.TimeoutExpired:
            ui_process.kill()
            print("✅ UI server force killed")
        except Exception as e:
            print(f"⚠️ Lỗi dừng UI: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\n🛑 Nhận tín hiệu {signum}, đang tắt hệ thống...")
    stop_servers()
    print("👋 Cảm ơn đã sử dụng LegalAdvisor!")
    sys.exit(0)

def main():
    """Hàm chính"""
    # Nạp .env sớm để các ENV như GOOGLE_API_KEY/LEGALADVISOR_* có hiệu lực
    try:
        load_dotenv()
    except Exception:
        pass
    print("\n" + "="*50)
    print("   🏛️  LegalAdvisor - Hệ thống hỗ trợ pháp lý")
    print("   🚀 Phiên bản: 2.0 (Gemini Integration)")
    print("="*50 + "\n")
    
    # Kiểm tra xem có GPU không
    use_gpu = False
    try:
        import torch
        if torch.cuda.is_available():
            use_gpu = True
            print("✅ Đã phát hiện GPU, sẽ sử dụng GPU để tăng tốc xử lý")
        else:
            print("ℹ️  Không phát hiện GPU, sẽ sử dụng CPU")
    except ImportError:
        print("⚠️  Không thể kiểm tra GPU do chưa cài đặt PyTorch")
    print("🤖 Sử dụng Google Gemini cho text generation (bắt buộc)")

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Kiểm tra yêu cầu
    if not check_requirements():
        sys.exit(1)

    print("\n🚀 Khởi động hệ thống...")

    try:
        # Khởi động API server
        if not start_api_server(use_gpu=use_gpu):
            print("❌ Không thể khởi động API server")
            sys.exit(1)

        # Đợi API khởi động với vòng retry (tối đa 60 giây)
        print("⏳ Đợi API server khởi động hoàn toàn (tối đa 60s)...")
        import requests
        max_wait_seconds = 60
        start_time_wait = time.time()
        attempt = 0
        while True:
            attempt += 1
            # Nếu process API đã thoát, thông báo lỗi sớm
            if api_process and api_process.poll() is not None:
                print("❌ API server đã dừng trong quá trình khởi động. Vui lòng xem logs hiển thị từ API.")
                print("💡 Gợi ý: kiểm tra GOOGLE_API_KEY, thư mục models/retrieval và kết nối internet.")
                print("   → Nếu cần xây lại index: python src/retrieval/build_index.py")
                sys.exit(1)

            try:
                response = requests.get("http://localhost:8000/health", timeout=3)
                if response.status_code == 200:
                    print("✅ API server đã sẵn sàng!")
                    break
                else:
                    print(f"⚠️ /health trả về: {response.status_code} (attempt {attempt})")
            except Exception:
                # Chưa sẵn sàng, tiếp tục đợi
                pass

            elapsed = time.time() - start_time_wait
            if elapsed >= max_wait_seconds:
                print("❌ Không thể kết nối API trong 60 giây.")
                print("💡 Gợi ý: kiểm tra GOOGLE_API_KEY, thư mục models/retrieval và logs của API.")
                print("   → Nếu thiếu index: python src/retrieval/build_index.py")
                break
            time.sleep(1)

        # Khởi động UI server
        if not start_ui_server():
            print("❌ Không thể khởi động UI server")
            stop_servers()
            sys.exit(1)

        print("\n🎉 Hệ thống đã sẵn sàng!")
        print("=" * 50)
        print("📱 Truy cập:")
        print("   - Web UI: http://localhost:8501")
        print("   - API: http://localhost:8000")
        print("   - API Docs: http://localhost:8000/docs")
        print("\n🛑 Nhấn Ctrl+C để dừng hệ thống")
        print("=" * 50)

        # Giữ main thread chạy và monitor processes
        while True:
            # Kiểm tra xem processes còn chạy không
            if api_process and api_process.poll() is not None:
                print("⚠️ API server đã dừng bất ngờ")
                break
            if ui_process and ui_process.poll() is not None:
                print("⚠️ UI server đã dừng bất ngờ")
                break

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Đang dừng hệ thống...")
        stop_servers()
        print("👋 Cảm ơn đã sử dụng LegalAdvisor!")

    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        stop_servers()
        print("👋 Cảm ơn đã sử dụng LegalAdvisor!")

if __name__ == "__main__":
    main()
