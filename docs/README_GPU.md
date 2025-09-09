# 🚀 Hướng Dẫn Sử Dụng GPU Với LegalAdvisor

## Tổng Quan

LegalAdvisor hỗ trợ GPU acceleration để tăng đáng kể hiệu suất xử lý. Tài liệu này hướng dẫn cách cài đặt và sử dụng GPU.

## Kiểm Tra GPU Hiện Tại

Chạy script kiểm tra GPU:

```bash
python check_gpu.py
```

Script sẽ:
- ✅ Kiểm tra phiên bản Python
- 🔥 Kiểm tra PyTorch GPU support
- 🤗 Kiểm tra transformers GPU support
- 🔍 Kiểm tra FAISS GPU support
- ⚡ Benchmark hiệu suất GPU vs CPU

## Yêu Cầu Hệ Thống

### Phần Cứng
- **GPU NVIDIA** với CUDA support (GTX 10xx, RTX 20xx/30xx/40xx series)
- **RAM**: Tối thiểu 8GB (16GB khuyến nghị)
- **Ổ cứng**: Tối thiểu 20GB trống

### Phần Mềm
- **Python**: 3.8+
- **CUDA Toolkit**: 11.8 hoặc 12.1
- **NVIDIA Drivers**: Phiên bản mới nhất

## Cài Đặt GPU Support

### Bước 1: Cài Đặt CUDA Toolkit

#### Windows
1. Tải CUDA Toolkit từ: https://developer.nvidia.com/cuda-downloads
2. Chọn phiên bản phù hợp:
   - **CUDA 11.8**: Cho GPU GTX/RTX 20xx series
   - **CUDA 12.1**: Cho GPU RTX 30xx/40xx series
3. Chạy installer với quyền Administrator
4. Restart máy tính

#### Linux
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

### Bước 2: Cài Đặt PyTorch Với CUDA

Trong environment LegalAdvisor:

```bash
# Kích hoạt conda environment
conda activate LegalAdvisor

# Gỡ bỏ phiên bản cũ
pip uninstall torch torchvision torchaudio

# Cài đặt PyTorch với CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Hoặc CUDA 12.1 (cho GPU RTX 30xx/40xx)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Bước 3: Cài Đặt FAISS GPU

```bash
# Gỡ bỏ phiên bản CPU
pip uninstall faiss-cpu

# Cài đặt phiên bản GPU
pip install faiss-gpu
```

### Bước 4: Verify Cài Đặt

```bash
# Chạy kiểm tra GPU
python check_gpu.py

# Chạy launcher để kiểm tra
python launcher.py
```

## Hiệu Suất Dự Kiến

### So Sánh GPU vs CPU

| Task | CPU (i7-10700K) | GPU (RTX 3060) | Tăng tốc |
|------|----------------|----------------|----------|
| Embedding (384-dim) | ~2-3 giây | ~0.1-0.2 giây | **10-20x** |
| LLM Generation | ~5-10 giây | ~0.5-1 giây | **5-10x** |
| FAISS Search | ~1-2 giây | ~0.05-0.1 giây | **15-30x** |
| **Tổng thời gian** | ~8-15 giây | ~0.7-1.3 giây | **6-15x** |

### Yêu Cầu Bộ Nhớ

| Model | CPU Memory | GPU Memory |
|-------|------------|------------|
| PhoBERT | ~500MB | ~1GB |
| GPT-2 VN | ~1GB | ~2GB |
| FAISS Index | ~2GB | ~3GB |
| **Tổng cộng** | ~3.5GB | ~6GB |

## Troubleshooting

### Lỗi "CUDA out of memory"

**Nguyên nhân**: GPU không đủ bộ nhớ cho model
**Giải pháp**:
1. Giảm batch size trong inference
2. Sử dụng model quantization
3. Tăng RAM GPU hoặc sử dụng GPU có nhiều RAM hơn

### Lỗi "No CUDA-capable device"

**Nguyên nhân**: Driver NVIDIA cũ hoặc CUDA không tương thích
**Giải pháp**:
1. Cập nhật NVIDIA drivers
2. Cài đặt lại CUDA Toolkit
3. Restart máy tính

### Lỗi "DLL load failed"

**Nguyên nhân**: PyTorch và CUDA version không tương thích
**Giải pháp**:
1. Gỡ bỏ PyTorch: `pip uninstall torch torchvision torchaudio`
2. Cài đặt lại với version phù hợp
3. Đảm bảo CUDA Toolkit được cài đặt đúng

## Cấu Hình Nâng Cao

### Quantization

Để giảm bộ nhớ GPU, sử dụng 4-bit quantization:

```python
# Trong legal_rag.py
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### Multi-GPU Support

LegalAdvisor hỗ trợ multi-GPU:

```python
# Tự động phân bổ model lên nhiều GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Tự động phân bổ
    quantization_config=quantization_config
)
```

## Monitoring GPU Usage

### Sử Dụng nvidia-smi

```bash
# Xem usage GPU real-time
nvidia-smi -l 1

# Xem processes sử dụng GPU
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

### Trong Code

```python
import torch

# Xem GPU memory usage
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB used")
print(f"GPU Memory: {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
```

## FAQ

### Q: Tôi có thể chạy LegalAdvisor không có GPU?

**A**: Có, LegalAdvisor sẽ tự động chuyển về CPU mode. Tuy nhiên hiệu suất sẽ chậm hơn nhiều.

### Q: GPU nào được khuyến nghị?

**A**: RTX 3060 trở lên với tối thiểu 8GB VRAM. RTX 4070 hoặc A-series cho hiệu suất tốt nhất.

### Q: Có thể sử dụng AMD GPU?

**A**: Hiện tại chưa hỗ trợ. LegalAdvisor chỉ hỗ trợ NVIDIA GPU với CUDA.

### Q: Làm sao để biết GPU có hoạt động?

**A**: Chạy `python check_gpu.py` và xem benchmark. Nếu speedup > 2x thì GPU hoạt động tốt.

## Support

Nếu gặp vấn đề với GPU setup:

1. Chạy `python check_gpu.py` để diagnose
2. Kiểm tra logs trong `logs/` folder
3. Tạo issue trên GitHub với thông tin:
   - GPU model
   - CUDA version
   - PyTorch version
   - Output của `check_gpu.py`

---

🎯 **Mục tiêu**: LegalAdvisor với GPU sẽ xử lý câu hỏi pháp luật chỉ trong **1-2 giây** thay vì **10-15 giây** với CPU!
