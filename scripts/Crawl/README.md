## 🚀 Lệnh sử dụng thực tế:
### A. Cào toàn bộ (Lần đầu tiên hoặc muốn làm mới kho luật):

```bash
conda run -n LegalAdvisor scrapy crawl vbpl
```
### B. Cào cập nhật (Chỉ lấy luật mới phát hành):

```bash
conda run -n LegalAdvisor scrapy crawl vbpl -a incremental=True
```
### C. Cào một Điều/Khoản/Điểm cụ thể để kiểm tra tính chính xác: (Ví dụ cào lại Bộ luật Dân sự để xem sự phân tách chi tiết)

```bash
conda run -n LegalAdvisor scrapy crawl vbpl -a item_id=95942
```