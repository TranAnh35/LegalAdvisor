# -*- coding: utf-8 -*-
"""
Scrapy Items: Định nghĩa cấu trúc dữ liệu đầu ra cho Legal Crawler.

Schema được thiết kế tối ưu cho HyperbolicRAG:
- Nội dung thuần khiết (không nhúng reference cứng)
- Metadata phân cấp phục vụ Hierarchy Tree
"""

import scrapy


class LegalDocumentItem(scrapy.Item):
    """Thông tin tổng quan Văn bản pháp luật (Parent Node)."""
    node_type = scrapy.Field()          # "document"
    doc_code = scrapy.Field()           # "91/2015/QH13"
    title = scrapy.Field()              # Tên đầy đủ (Trích yếu)
    doc_type = scrapy.Field()           # "Luật", "Nghị định", "Thông tư"...
    issuer = scrapy.Field()             # Cơ quan ban hành
    issue_date = scrapy.Field()         # Ngày ban hành
    effective_date = scrapy.Field()     # Ngày có hiệu lực
    status = scrapy.Field()            # "Còn hiệu lực" / "Hết hiệu lực"
    source_url = scrapy.Field()        # URL gốc trên Cổng TTĐT
    crawled_at = scrapy.Field()        # Timestamp lúc crawl


class LegalArticleItem(scrapy.Item):
    """Nội dung chi tiết - có thể là một Điều, Khoản hoặc Điểm (Granular Node)."""
    node_type = scrapy.Field()          # "article", "clause", "point"
    doc_code = scrapy.Field()           # Mã văn bản cha: "91/2015/QH13"
    article_number = scrapy.Field()     # Số Điều: 279
    clause_number = scrapy.Field()      # Số Khoản: 1, 2, 3 (nếu có)
    point_id = scrapy.Field()           # Định danh Điểm: a, b, c (nếu có)
    title = scrapy.Field()              # Tiêu đề Điều hoặc tóm tắt Khoản/Điểm
    content = scrapy.Field()            # Nội dung thuần túy của đơn vị này
    hierarchy_path = scrapy.Field()     # ["Phần I", "Chương II", "Điều 1", "Khoản 1"]
