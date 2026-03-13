# -*- coding: utf-8 -*-
"""
Scrapy Settings cho Legal Crawler.

Cấu hình tối ưu cho việc cào dữ liệu pháp luật từ Cổng TTĐT Chính Phủ:
- Rate limiting: Bảo vệ server, tránh bị block
- Retry: Xử lý lỗi mạng tự động
- Output: JSONL format phục vụ HyperbolicRAG Pipeline
"""

BOT_NAME = "legal_crawler"
SPIDER_MODULES = ["legal_crawler.spiders"]
NEWSPIDER_MODULE = "legal_crawler.spiders"

import os
from pathlib import Path
# Detect project root (LegalAdvisor)
# settings.py is in LegalAdvisor/scripts/crawl/legal_crawler/legal_crawler/
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent.parent)

# ==============================================================================
# ROBOTS.TXT & CRAWL ETHICS
# ==============================================================================
ROBOTSTXT_OBEY = True

# ==============================================================================
# CONCURRENT REQUESTS & RATE LIMITING
# ==============================================================================
CONCURRENT_REQUESTS = 2               # Chỉ 2 request song song
CONCURRENT_REQUESTS_PER_DOMAIN = 2    # Giới hạn theo domain
DOWNLOAD_DELAY = 2.0                  # Nghỉ 2s giữa mỗi request
RANDOMIZE_DOWNLOAD_DELAY = True       # Random hóa delay (0.5x -> 1.5x)
DOWNLOAD_TIMEOUT = 30                 # Timeout 30s

# ==============================================================================
# RETRY
# ==============================================================================
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# ==============================================================================
# AUTO THROTTLE (tự điều chỉnh tốc độ theo response time)
# ==============================================================================
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 2
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.5

# ==============================================================================
# HEADERS
# ==============================================================================
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ==============================================================================
# PIPELINES
# ==============================================================================
ITEM_PIPELINES = {
    "legal_crawler.pipelines.DeduplicationPipeline": 100,
    "legal_crawler.pipelines.CleanTextPipeline": 200,
    "legal_crawler.pipelines.SQLitePipeline": 300,
    "legal_crawler.pipelines.ProgressPipeline": 400,
    "legal_crawler.pipelines.JsonlExportPipeline": 900,
}

# ==============================================================================
# OUTPUT
# ==============================================================================
# Đường dẫn mặc định, có thể override bằng -a output_dir=...
LEGAL_OUTPUT_DIR = "data/raw/crawled"

# ==============================================================================
# LOGGING
# ==============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

# ==============================================================================
# CACHE (HttpCache)
# CHÚ Ý:
#   - Để HTTPCACHE_ENABLED = False khi **crawl production** để tránh dữ liệu cũ
#     và tiết kiệm hàng chục GB dung lượng đĩa (mỗi trang HTML ~100-500KB).
#   - Chỉ bật True khi **phát triển/debug** để test nhanh mà không cần tải lại.
# ==============================================================================
HTTPCACHE_ENABLED = False  
# HTTPCACHE_EXPIRATION_SECS = 86400  # 1 ngày (dùng khi dev)
# HTTPCACHE_DIR = ".scrapy_cache"    # Sẽ bị .gitignore
# HTTPCACHE_IGNORE_HTTP_CODES = [403, 404, 500, 502, 503]

# ==============================================================================
# ENCODING
# ==============================================================================
FEED_EXPORT_ENCODING = "utf-8"

# Tắt telnet console (bảo mật)
TELNETCONSOLE_ENABLED = False

# Request fingerprinting
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
