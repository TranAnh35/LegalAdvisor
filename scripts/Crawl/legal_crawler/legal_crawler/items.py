# -*- coding: utf-8 -*-
import scrapy

class LegalDocumentItem(scrapy.Item):
    node_type = scrapy.Field()      # 'document'
    doc_code = scrapy.Field()       # Số ký hiệu (ví dụ: 79/2025/qh15)
    title = scrapy.Field()          # Tên đầy đủ
    doc_type = scrapy.Field()       # Loại văn bản
    issuer = scrapy.Field()         # Cơ quan ban hành
    issue_date = scrapy.Field()     # Ngày ban hành
    effective_date = scrapy.Field() # Ngày có hiệu lực
    status = scrapy.Field()         # Tình trạng hiệu lực
    source_url = scrapy.Field()
    crawled_at = scrapy.Field()
    full_markdown = scrapy.Field()
    parse_status = scrapy.Field()   # 'SUCCESS', 'SUSPECT', 'FAILED' (MỚI)
    parse_log = scrapy.Field()      # Lưu lý do bị đánh dấu SUSPECT (MỚI)

class LegalArticleItem(scrapy.Item):
    # ID của node: [doc_code]_art_1_clause_1_guest_art_5_clause_4
    node_id = scrapy.Field()        # ID duy nhất theo sơ đồ Namespace (MỚI)
    node_type = scrapy.Field()      # 'article', 'clause', 'point'
    doc_code = scrapy.Field()       # Văn bản gốc sở hữu node này
    
    parent_id = scrapy.Field()      # Node cha (MỚI)
    article_number = scrapy.Field()
    clause_number = scrapy.Field()
    point_id = scrapy.Field()
    
    title = scrapy.Field()          # Tiêu đề Điều
    content = scrapy.Field()        # Nội dung text
    hierarchy_path = scrapy.Field() # List: ['Điều 1', 'Khoản 2']
    is_amendment = scrapy.Field()   # Đánh dấu đây là nội dung sửa đổi (GUEST) (MỚI)
