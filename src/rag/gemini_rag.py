#!/usr/bin/env python3
"""
Gemini RAG implementation for LegalAdvisor
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np  # type: ignore

from dotenv import load_dotenv
from ..retrieval.service import RetrievalService
from ..utils.law_registry import normalize_act_code  # type: ignore

GEMINI_MODEL = "gemini-2.5-flash-lite"

def _vietnamese_doc_title(type_code: str, number: str) -> str:
    """Chuyển type+number thành tên văn bản thân thiện.
    Ví dụ: ttlt-bca-btp-vksndtc-tandtc + 13/2012 ->
    "Thông tư liên tịch 13/2012/TTLT-BCA-BTP-VKSNDTC-TANDTC"
    """
    if not type_code:
        return number or "Văn bản pháp luật"
    code = (type_code or '').lower()
    code_upper = (type_code or '').upper()
    mapping = {
        'nđ-cp': 'Nghị định',
        'nd-cp': 'Nghị định',
        'tt': 'Thông tư',
        'tt-bca': 'Thông tư',
        'tt-byt': 'Thông tư',
        'ttlt': 'Thông tư liên tịch',
        'ttlt-bca-btp-vksndtc-tandtc': 'Thông tư liên tịch',
        'qđ-ttg': 'Quyết định',
        'qd-ttg': 'Quyết định',
        'lh': 'Luật',
        'qh': 'Luật',
    }
    vn_type = mapping.get(code, code_upper)
    return f"{vn_type} {number}/{code_upper}"

def format_retrieved_docs(docs: List[Dict[str, Any]]) -> str:
    """Ghép tài liệu tham chiếu kèm tiêu đề luật và nội dung đầy đủ."""
    formatted_docs: List[str] = []
    for i, doc in enumerate(docs, 1):
        corpus_id = doc.get('corpus_id') or ''
        type_code = doc.get('type') or ''
        number = doc.get('number') or ''
        year = doc.get('year') or ''
        suffix = doc.get('suffix') or ''
        dieu = f"Điều {suffix}" if str(suffix).isdigit() else ''

        law_title = _vietnamese_doc_title(type_code, number)

        content = (doc.get('content_full') or doc.get('content') or '').strip()
        # Hiển thị thân thiện: thay '_' bằng ' ' trong nội dung tham chiếu
        snippet = content.replace('_', ' ')

        formatted_docs.append(
            f"[Nguồn {i}] {law_title}{(' - ' + dieu) if dieu else ''} — `{corpus_id}`\n{snippet}\n(điểm: {doc.get('score', 0):.2f})"
        )
    return "\n\n".join(formatted_docs)

class GeminiRAG:
    """RAG implementation using Google's Gemini for legal question answering"""
    
    def __init__(self, use_gpu: bool = False):
        """Initialize the GeminiRAG system"""
        self.use_gpu = use_gpu
        self.retriever = None
        self.model = None
        self.metadata = {}
        
        # Initialize components
        self._initialize_retriever()
        self._initialize_gemini()
        
        # Log only via logger, not print (to avoid console noise)
        import logging
        _logger = logging.getLogger("legaladvisor.rag")
        _logger.info("GeminiRAG initialized successfully")
    
    def _initialize_retriever(self):
        """Initialize unified RetrievalService"""
        try:
            self.retriever = RetrievalService(use_gpu=self.use_gpu)
            # Mirror thông tin phục vụ /stats
            self.model_info = getattr(self.retriever, 'model_info', {})
            self.metadata = getattr(self.retriever, 'metadata', {})
        except Exception as e:
            raise RuntimeError(f"Failed to initialize retriever: {str(e)}")
    
    def _initialize_gemini(self):
        """Initialize the Gemini model"""
        try:
            # Load env and require API key at runtime (not at import time)
            load_dotenv()
            google_api_key = os.getenv('GOOGLE_API_KEY')
            if not google_api_key:
                raise RuntimeError("GOOGLE_API_KEY not found in environment variables")

            # Import and configure google.generativeai lazily so importing this
            # module (or running tests that mock RAG) does not fail when the key
            # is not set.
            import google.generativeai as genai  # imported here intentionally

            genai.configure(api_key=google_api_key)

            # Initialize the Gemini model
            generation_config = {
                "temperature": 0.1,  # thấp hơn để giảm suy diễn
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 2048 * 4,
            }

            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                },
            ]

            self.model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini model: {str(e)}")
    
    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        try:
            if not self.retriever:
                return []
            return self.retriever.retrieve(query, top_k=top_k)
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []

    @staticmethod
    def _score_to_similarity(score: float) -> float:
        """Chuyển inner-product trên vector chuẩn hoá [-1,1] về [0,1] để ước lượng confidence."""
        try:
            x = float(score)
            # IP của vectors đã normalize nằm trong [-1, 1]. Quy về [0,1]
            return max(0.0, min((x + 1.0) / 2.0, 1.0))
        except Exception:
            return 0.0

    def _group_by_document(self, docs: List[Dict[str, Any]], top_k_docs: int) -> List[Dict[str, Any]]:
        """Nhóm các chunk theo tài liệu (act_code_norm) và chọn top-k tài liệu.

        Chiến lược gộp có thể cấu hình qua ENV:
        - LEGALADVISOR_GROUP_STRATEGY: mean (mặc định) | max | topm_mean
        - LEGALADVISOR_GROUP_TOPM: số m khi dùng topm_mean (mặc định 3)
        """
        strategy = os.getenv("LEGALADVISOR_GROUP_STRATEGY", "mean").strip().lower()
        try:
            top_m = int(os.getenv("LEGALADVISOR_GROUP_TOPM", "3"))
        except Exception:
            top_m = 3
        groups: Dict[str, Dict[str, Any]] = {}
        for d in docs:
            corpus_id = str(d.get('corpus_id') or '').strip()
            if not corpus_id:
                # Bỏ qua chunk thiếu nhận diện tài liệu
                continue
            raw_code = corpus_id.split('+')[0].strip()
            code_norm = normalize_act_code(raw_code) if raw_code else ''
            if not code_norm:
                continue
            g = groups.setdefault(code_norm, {
                'act_code': code_norm,
                'corpus_id': corpus_id,
                'articles': set(),
                'docs': [],
                'scores': [],
                'best_score': None,
                'group_score': 0.0,
            })
            # Thu thập Điều nếu có
            suf = d.get('suffix')
            if suf is not None and str(suf).isdigit():
                try:
                    g['articles'].add(int(suf))
                except Exception:
                    pass
            g['docs'].append(d)
            sc = float(d.get('score', 0.0))  # IP similarity do FAISS trả về
            g['scores'].append(sc)
            if g['best_score'] is None or sc > g['best_score']:
                g['best_score'] = sc

        # Tính điểm nhóm theo chiến lược
        for code, g in groups.items():
            scores: List[float] = g.get('scores', [])
            if not scores:
                g['group_score'] = 0.0
                continue
            if strategy == 'max':
                g['group_score'] = float(max(scores))
            elif strategy == 'topm_mean':
                m = max(1, int(top_m))
                topm = sorted(scores, reverse=True)[:m]
                g['group_score'] = float(sum(topm) / max(1, len(topm)))
            else:  # default: mean
                g['group_score'] = float(sum(scores) / max(1, len(scores)))

        # Chuyển set -> list và sắp xếp theo Điều tăng dần
        grouped: List[Dict[str, Any]] = []
        for code, g in groups.items():
            arts = sorted(list(g['articles'])) if g['articles'] else []
            grouped.append({
                'act_code': g['act_code'],
                'corpus_id': g['corpus_id'],
                'articles': arts,
                'article_count': len(arts),
                'group_score': float(g['group_score']),
                'best_score': float(g['best_score']) if g['best_score'] is not None else None,
            })

        # Sắp xếp theo group_score giảm dần, tie-break bằng best_score giảm dần
        grouped.sort(key=lambda x: (-x['group_score'], -(x['best_score'] if x['best_score'] is not None else -1e9)))
        return grouped[:max(1, int(top_k_docs))]

    def _get_chunk_content_by_id(self, chunk_id: int) -> Optional[str]:
        """Đọc content theo chunk_id từ SQLite hoặc Parquet (lazy)."""
        try:
            if not self.retriever:
                return None
            return self.retriever.get_chunk_content(int(chunk_id))
        except Exception:
            return None

    def _fetch_article_content_worker(self, code: str, article_num: int, max_segments_per_article: int) -> Dict[str, Any]:
        """Worker function để fetch content của một article (dùng cho parallel execution).
        
        Returns:
            Dict với keys: code, article_num, text, success, error
        """
        try:
            # Lấy segments của article
            seg_defs = self.retriever.get_article_segments_text(
                chunk_id=None,  # Sẽ xác định từ code+article_num bên trong
                max_segments=max_segments_per_article
            ) if hasattr(self.retriever, 'get_article_segments_text') else []
            
            # Fallback: lấy text trực tiếp
            text = self.retriever.get_article_text(code, article_num)
            if not text:
                # Try to get all content for this article
                article_contents = self.retriever.get_article_contents(code, article_num)
                if article_contents:
                    text = '\n\n'.join(c.get('content', '') for c in article_contents if c.get('content'))
            
            text = (text or '').replace('_', ' ').strip()
            
            return {
                'code': code,
                'article_num': article_num,
                'text': text,
                'success': bool(text),
                'error': None
            }
        except Exception as e:
            import logging
            logger = logging.getLogger("legaladvisor.rag")
            logger.warning(f"Failed to fetch article {code}+{article_num}: {e}")
            return {
                'code': code,
                'article_num': article_num,
                'text': '',
                'success': False,
                'error': str(e)
            }
    
    def _build_llm_context(self, retrieved_docs: List[Dict[str, Any]], sources_grouped: List[Dict[str, Any]]) -> str:
        """Xây dựng ngữ cảnh cho LLM bao gồm:
        - Nội dung của toàn bộ top-k tài liệu (ở cấp văn bản). Nếu có danh sách Điều, ưu tiên ghép theo Điều.
        - Nội dung các tài liệu được trích dẫn từ các tài liệu top-k (nếu phát hiện), trong giới hạn dung lượng.
        Giới hạn qua ENV:
        - LEGALADVISOR_CONTEXT_TOTAL_CHARS (mặc định 20000)
        - LEGALADVISOR_CONTEXT_DOC_CHARS (mặc định 6000 mỗi văn bản)
        - LEGALADVISOR_CONTEXT_CITATION_CHARS (mặc định 4000 tổng cho phần trích dẫn)
        
        OPTIMIZATION: Parallel fetching của articles sử dụng ThreadPoolExecutor
        """
        if not self.retriever:
            return ""
        try:
            total_budget = int(os.getenv("LEGALADVISOR_CONTEXT_TOTAL_CHARS", "20000"))
        except Exception:
            total_budget = 20000
        try:
            per_doc_budget = int(os.getenv("LEGALADVISOR_CONTEXT_DOC_CHARS", "6000"))
        except Exception:
            per_doc_budget = 6000
        try:
            citation_budget = int(os.getenv("LEGALADVISOR_CONTEXT_CITATION_CHARS", "4000"))
        except Exception:
            citation_budget = 4000
        try:
            max_segments_per_article = int(os.getenv("LEGALADVISOR_CONTEXT_MAX_SEGMENTS_PER_ARTICLE", "3"))
        except Exception:
            max_segments_per_article = 3
        if max_segments_per_article <= 0:
            max_segments_per_article = 3

        # Map (act_code_norm, Điều) -> chunk_id và danh sách segment part trúng theo score
        article_chunk_ids: Dict[tuple, int] = {}
        article_hit_parts: Dict[tuple, List[int]] = {}
        for d in (retrieved_docs or []):
            corpus_id = str(d.get('corpus_id') or '').strip()
            if not corpus_id:
                continue
            raw_code = corpus_id.split('+')[0].strip()
            code_norm = normalize_act_code(raw_code) if raw_code else ''
            if not code_norm:
                continue
            suffix = d.get('suffix')
            art: Optional[int] = None
            if suffix is not None and str(suffix).isdigit():
                try:
                    art = int(suffix)
                except Exception:
                    art = None
            if art is None:
                try:
                    parts = corpus_id.split('+', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        art = int(parts[1])
                except Exception:
                    art = None
            if art is None:
                continue
            key = (code_norm, int(art))
            chunk_id = d.get('chunk_id')
            try:
                if chunk_id is not None and key not in article_chunk_ids:
                    article_chunk_ids[key] = int(chunk_id)
            except Exception:
                pass
            segs = d.get('segments') or []
            parts_order: List[int] = []
            for seg in segs:
                p = seg.get('part')
                try:
                    p_int = int(p)
                except Exception:
                    continue
                if p_int not in parts_order:
                    parts_order.append(p_int)
            if parts_order and key not in article_hit_parts:
                article_hit_parts[key] = parts_order

        parts: List[str] = []
        remaining = total_budget

        # ===== OPTIMIZATION: Parallel fetching của articles =====
        # 1) Chuẩn bị danh sách articles cần fetch
        articles_to_fetch: List[tuple] = []  # (code, article_num)
        for g in (sources_grouped or []):
            code = g.get('act_code') or ''
            articles = g.get('articles') or []
            if isinstance(articles, list) and len(articles) > 0:
                for a in articles:
                    try:
                        a_int = int(a)
                        articles_to_fetch.append((code, a_int))
                    except Exception:
                        continue
        
        # 2) Fetch tất cả articles song song sử dụng ThreadPoolExecutor
        article_contents_map: Dict[tuple, str] = {}
        if articles_to_fetch:
            try:
                max_workers = min(4, max(1, len(articles_to_fetch)))  # 4 threads tối đa
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit tất cả tasks
                    futures = {
                        executor.submit(
                            self._fetch_article_content_worker,
                            code,
                            article_num,
                            max_segments_per_article
                        ): (code, article_num)
                        for code, article_num in articles_to_fetch
                    }
                    
                    # Collect results khi chúng sẵn sàng
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            key = (result['code'], result['article_num'])
                            article_contents_map[key] = result.get('text', '')
                        except Exception as e:
                            import logging
                            logger = logging.getLogger("legaladvisor.rag")
                            logger.warning(f"Parallel fetch failed: {e}")
            except Exception as e:
                import logging
                logger = logging.getLogger("legaladvisor.rag")
                logger.warning(f"ThreadPoolExecutor error: {e}")
        
        # 1) Nội dung toàn bộ top-k tài liệu (ưu tiên các Điều/segments đã trúng)
        parts.append("[Tài liệu tham khảo (toàn văn)]")
        for g in (sources_grouped or []):
            if remaining <= 0:
                break
            code = g.get('act_code') or ''
            articles = g.get('articles') or []
            header = f"Văn bản `{code}`\n"
            body_sections: List[str] = []
            used = 0
            if isinstance(articles, list) and len(articles) > 0:
                # Ưu tiên những Điều đã có segment trúng
                hits: List[int] = []
                others: List[int] = []
                for a in articles:
                    try:
                        a_int = int(a)
                    except Exception:
                        continue
                    key = (code, a_int)
                    if key in article_hit_parts:
                        hits.append(a_int)
                    else:
                        others.append(a_int)
                ordered_articles = hits + others

                for a_int in ordered_articles:
                    quota = max(0, per_doc_budget - used)
                    if quota <= 0:
                        break
                    key = (code, int(a_int))
                    chunk_id = article_chunk_ids.get(key)
                    section_body = ""

                    # Nếu có chunk_id -> ưu tiên ghép theo các segment trúng
                    if chunk_id is not None:
                        seg_defs = self.retriever.get_article_segments_text(chunk_id, max_segments=max_segments_per_article)
                        seg_by_part: Dict[int, str] = {}
                        for seg in seg_defs or []:
                            p = seg.get('part')
                            try:
                                p_int = int(p)
                            except Exception:
                                continue
                            seg_text = str(seg.get('text') or '').replace('_', ' ').strip()
                            if seg_text:
                                seg_by_part[p_int] = seg_text
                        if seg_by_part:
                            hit_order = article_hit_parts.get(key, [])
                            ordered_parts: List[int] = []
                            for p in hit_order:
                                if p in seg_by_part and p not in ordered_parts:
                                    ordered_parts.append(p)
                            for p in sorted(seg_by_part.keys()):
                                if p not in ordered_parts:
                                    ordered_parts.append(p)
                            seg_texts: List[str] = []
                            used_seg_chars = 0
                            for p in ordered_parts:
                                if used_seg_chars >= quota:
                                    break
                                t = seg_by_part.get(p) or ""
                                if not t:
                                    continue
                                take = min(quota - used_seg_chars, len(t))
                                seg_texts.append(t[:take])
                                used_seg_chars += take
                            if seg_texts:
                                section_body = "\n\n".join(seg_texts)

                    # Fallback: lấy từ parallel-fetched cache hoặc fallback thêm lần
                    if not section_body:
                        cached_key = (code, a_int)
                        cached_text = article_contents_map.get(cached_key)
                        if cached_text:
                            section_body = cached_text[:quota]
                        else:
                            # Fallback: gọi trực tiếp (dư phòng)
                            text = self.retriever.get_article_text(code, a_int)
                            if not text:
                                continue
                            text = text.replace('_', ' ').strip()
                            section_body = text[:quota]

                    section = f"Điều {a_int}\n{section_body}"
                    used += len(section)
                    body_sections.append(section)

            # Nếu không có danh sách Điều hoặc phần Điều rỗng -> lấy toàn văn bản
            if not body_sections:
                full_text = self.retriever.get_document_text_all(code) or ""
                ft = full_text.replace('_', ' ').strip()
                body_sections.append(ft[:per_doc_budget])
                used = min(per_doc_budget, len(ft))

            body = "\n\n".join(s for s in body_sections if s)
            doc_block = header + body
            take = min(remaining, len(doc_block))
            parts.append(doc_block[:take])
            remaining -= take

        # 2) Nội dung tài liệu trích dẫn
        if remaining > 0:
            try:
                # Trích dẫn từ nội dung các chunk đã truy hồi
                from ..retrieval.citation.extract import extract_citations  # type: ignore
                from ..utils.law_registry import get_registry  # type: ignore
                reg = None
                try:
                    reg = get_registry()
                except Exception:
                    reg = None

                concat = []
                for d in (retrieved_docs or []):
                    content = (d.get('content_full') or d.get('content') or '')
                    if content:
                        concat.append(str(content))
                text_all = "\n\n".join(concat)
                hits = extract_citations(text_all, registry=reg, article_only=True)
                # Gom theo mã văn bản và danh sách Điều
                code_to_articles: Dict[str, set] = {}
                for h in hits:
                    if not h.act_code_norm:
                        continue
                    st_articles = code_to_articles.setdefault(h.act_code_norm, set())
                    if isinstance(h.article, int):
                        st_articles.add(int(h.article))

                if code_to_articles:
                    parts.append("\n[Tài liệu trích dẫn]")
                    cite_remaining = min(remaining, citation_budget)
                    for ccode, arts in code_to_articles.items():
                        if cite_remaining <= 0:
                            break
                        header = f"Văn bản được trích dẫn `{ccode}`\n"
                        sections: List[str] = []
                        used = 0
                        a_list = sorted(list(arts)) if arts else []
                        if a_list:
                            for a in a_list:
                                txt = self.retriever.get_article_text(ccode, int(a))
                                if not txt:
                                    continue
                                txt = txt.replace('_', ' ').strip()
                                quota = max(0, citation_budget - used)
                                if quota <= 0:
                                    break
                                section = f"Điều {int(a)}\n{txt}"
                                section = section[:quota]
                                used += len(section)
                                sections.append(section)
                        if not sections:
                            full = self.retriever.get_document_text_all(ccode) or ""
                            full = full.replace('_', ' ').strip()
                            sections.append(full[:citation_budget])
                            used = min(citation_budget, len(full))
                        block = header + "\n\n".join(sections)
                        take = min(cite_remaining, len(block))
                        parts.append(block[:take])
                        cite_remaining -= take
                        remaining -= take
            except Exception:
                pass

        return "\n\n".join([p for p in parts if p and p.strip()])[:total_budget]
    
    def _build_article_entries(self, retrieved_docs: List[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
        entries: Dict[tuple, Dict[str, Any]] = {}
        for doc in retrieved_docs or []:
            corpus_id = str(doc.get("corpus_id") or "").strip()
            if not corpus_id:
                continue
            raw_code = corpus_id.split("+")[0].strip()
            code_norm = normalize_act_code(raw_code) if raw_code else ""
            if not code_norm:
                continue
            suffix = doc.get("suffix")
            article: Optional[int] = None
            if suffix is not None and str(suffix).isdigit():
                try:
                    article = int(suffix)
                except Exception:
                    article = None
            if article is None:
                try:
                    parts = corpus_id.split("+", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        article = int(parts[1])
                except Exception:
                    article = None
            if article is None:
                continue
            key = (code_norm, int(article))
            entry = entries.setdefault(key, {
                "act_code": code_norm,
                "article": int(article),
                "segments": [],
                "score": -1e9,
                "best_score": -1e9,
            })
            entry["segments"].append(doc)
            score = float(doc.get("score", 0.0))
            if score > entry["score"]:
                entry["score"] = score
            if score > entry["best_score"]:
                entry["best_score"] = score
        return entries

    def _calculate_article_scores(self, article_entries: Dict[tuple, Dict[str, Any]]) -> None:
        """Tính toán Article Score dựa trên segments.
        
        Phương pháp: top-M mean (lấy top-M scores cao nhất, tính trung bình)
        Được cấu hình qua:
        - LEGALADVISOR_ARTICLE_SCORE_METHOD: mean|max|top_mean (mặc định: top_mean)
        - LEGALADVISOR_ARTICLE_SCORE_TOP_M: M (mặc định: 5)
        """
        method = os.getenv("LEGALADVISOR_ARTICLE_SCORE_METHOD", "top_mean").strip().lower()
        try:
            top_m = int(os.getenv("LEGALADVISOR_ARTICLE_SCORE_TOP_M", "5"))
        except Exception:
            top_m = 5
        top_m = max(1, top_m)

        for entry in article_entries.values():
            segments = entry.get("segments", [])
            if not segments:
                entry["score"] = 0.0
                continue

            scores = [float(seg.get("score", 0.0)) for seg in segments]
            
            if method == "max":
                entry["score"] = float(max(scores))
            elif method == "mean":
                entry["score"] = float(sum(scores) / len(scores))
            else:  # top_mean (default)
                top_scores = sorted(scores, reverse=True)[:top_m]
                entry["score"] = float(sum(top_scores) / len(top_scores))

    def _apply_adaptive_threshold(self, article_entries: Dict[tuple, Dict[str, Any]], 
                                   detail_level: str = "moderate") -> tuple[List[Dict[str, Any]], float]:
        """Áp dụng adaptive threshold để filter articles.
        
        Returns: (filtered_articles, threshold_used)
        """
        # Lấy cấu hình từ environment
        try:
            threshold_start = float(os.getenv("LEGALADVISOR_THRESHOLD_START", "0.60"))
            threshold_min = float(os.getenv("LEGALADVISOR_THRESHOLD_MIN", "0.40"))
            threshold_max = float(os.getenv("LEGALADVISOR_THRESHOLD_MAX", "0.80"))
            threshold_step = float(os.getenv("LEGALADVISOR_THRESHOLD_STEP", "0.05"))
            min_articles = int(os.getenv("LEGALADVISOR_THRESHOLD_MIN_ARTICLES", "3"))
            max_articles = int(os.getenv("LEGALADVISOR_THRESHOLD_MAX_ARTICLES", "15"))
        except Exception:
            threshold_start = 0.60
            threshold_min = 0.40
            threshold_max = 0.80
            threshold_step = 0.05
            min_articles = 3
            max_articles = 15

        # Detail level mapping
        detail_config = {
            "brief": {
                "threshold_start": 0.70,
                "max_articles": 5,
            },
            "moderate": {
                "threshold_start": threshold_start,
                "max_articles": max_articles,
            },
            "comprehensive": {
                "threshold_start": 0.45,
                "max_articles": 30,
            },
        }
        config = detail_config.get(detail_level.lower(), detail_config["moderate"])
        threshold = config["threshold_start"]
        max_target = config["max_articles"]

        # Adaptive threshold: adjust để có 3-15 articles (hoặc tùy detail level)
        for iteration in range(5):
            candidates = [
                e for e in article_entries.values()
                if float(e.get("score", 0.0)) >= threshold
            ]
            
            if min_articles <= len(candidates) <= max_target:
                return candidates, threshold
            
            if len(candidates) < min_articles and threshold > threshold_min:
                threshold = max(threshold_min, threshold - threshold_step)
            elif len(candidates) > max_target and threshold < threshold_max:
                threshold = min(threshold_max, threshold + threshold_step)
            else:
                break

        # Fallback: lấy top-max_target by score nếu không converge
        all_articles = sorted(
            article_entries.values(),
            key=lambda x: float(x.get("score", 0.0)),
            reverse=True
        )
        return all_articles[:max_target], threshold

    def _apply_article_limits(self, doc_infos: List[Dict[str, Any]]) -> None:
        try:
            max_articles = int(os.getenv("LEGALADVISOR_MAX_ARTICLES_PER_DOC", "-1"))
        except Exception:
            max_articles = -1
        try:
            max_segments = int(os.getenv("LEGALADVISOR_MAX_SEGMENTS_PER_ARTICLE", "6"))
        except Exception:
            max_segments = 6
        max_segments = max(1, max_segments)

        for doc in doc_infos:
            articles = sorted(doc.get("articles", []), key=lambda a: a.get("score", 0.0), reverse=True)
            # Nếu max_articles < 0, lấy TẤT CẢ articles (mặc định)
            if max_articles > 0 and len(articles) > max_articles:
                articles = articles[:max_articles]
            for art in articles:
                segs = art.get("segments", [])
                if len(segs) > max_segments:
                    art["segments"] = segs[:max_segments]
            doc["articles"] = articles
            doc["article_count"] = len(articles)

    def _select_top_documents(self, article_entries: Dict[tuple, Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        strategy = os.getenv("LEGALADVISOR_GROUP_STRATEGY", "mean").strip().lower()
        try:
            top_m = int(os.getenv("LEGALADVISOR_ARTICLE_GROUP_TOPM", "3"))
        except Exception:
            top_m = 3
        doc_map: Dict[str, Dict[str, Any]] = {}
        for entry in article_entries.values():
            code = entry["act_code"]
            doc_info = doc_map.setdefault(code, {
                "act_code": code,
                "corpus_id": "",
                "articles": [],
                "scores": [],
                "doc_score": 0.0,
                "best_score": None,
            })
            doc_info["articles"].append(entry)
            doc_info["scores"].append(float(entry.get("score", 0.0)))
            if not doc_info["corpus_id"]:
                first_seg = entry.get("segments", [None])[0]
                if first_seg:
                    cid = first_seg.get("corpus_id")
                    if cid:
                        doc_info["corpus_id"] = cid
        for doc in doc_map.values():
            scores = doc.get("scores", [])
            if not scores:
                doc["doc_score"] = 0.0
            elif strategy == "max":
                doc["doc_score"] = float(max(scores))
            elif strategy == "topm_mean":
                m = max(1, top_m)
                top_scores = sorted(scores, reverse=True)[:m]
                doc["doc_score"] = float(sum(top_scores) / max(1, len(top_scores)))
            else:
                doc["doc_score"] = float(sum(scores) / max(1, len(scores)))
            best = max(scores) if scores else None
            doc["best_score"] = float(best) if best is not None else None
        doc_list = list(doc_map.values())
        doc_list.sort(key=lambda d: (-d.get("doc_score", 0.0), -(d.get("best_score", 0.0) if d.get("best_score") is not None else -1e9)))
        top_docs = doc_list[:max(1, int(top_k))]
        self._apply_article_limits(top_docs)
        return top_docs

    def _select_top_articles_mode(self, article_entries: Dict[tuple, Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        articles = list(article_entries.values())
        articles.sort(key=lambda a: a.get("score", 0.0), reverse=True)
        selected = articles[:max(1, int(top_k))]
        doc_map: Dict[str, Dict[str, Any]] = {}
        for entry in selected:
            code = entry["act_code"]
            doc_info = doc_map.setdefault(code, {
                "act_code": code,
                "corpus_id": "",
                "articles": [],
                "doc_score": 0.0,
                "best_score": None,
            })
            doc_info["articles"].append(entry)
            score = float(entry.get("score", 0.0))
            doc_info["doc_score"] = max(doc_info.get("doc_score", 0.0), score)
            doc_info["best_score"] = max(doc_info.get("best_score", 0.0) or 0.0, score)
            if not doc_info["corpus_id"]:
                first_seg = entry.get("segments", [None])[0]
                if first_seg:
                    cid = first_seg.get("corpus_id")
                    if cid:
                        doc_info["corpus_id"] = cid
        doc_list = list(doc_map.values())
        self._apply_article_limits(doc_list)
        return doc_list

    def _fetch_all_segments_for_articles(self, filtered_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Lấy TẤT CẢ segments của articles được chọn từ database.
        
        Vì retrieve_documents chỉ lấy top-k segments (để scoring),
        nên nếu article dài có 10 segments nhưng chỉ 2 match trong top-k,
        ta cần lấy đầy đủ 10 segments từ database để xử lý toàn bộ nội dung.
        
        Returns: List[Dict] với field 'segments_full' được thêm vào
        """
        if not self.retriever:
            return filtered_articles
        
        import logging
        logger = logging.getLogger("legaladvisor.rag")
        
        result_articles = []
        
        for article in filtered_articles:
            code = article.get("act_code", "")
            article_num = article.get("article")
            
            if not code or article_num is None:
                result_articles.append(article)
                continue
            
            try:
                # Lấy toàn bộ segments của article từ retriever
                # Sử dụng get_article_contents() để lấy tất cả chunks
                full_article_chunks = self.retriever.get_article_contents(code, article_num)
                
                if full_article_chunks:
                    # Cập nhật article với toàn bộ segments từ database
                    article_copy = dict(article)
                    
                    # Convert chunks thành segments format (thêm score nếu cần)
                    full_segments = []
                    for chunk in full_article_chunks:
                        seg = {
                            "chunk_id": chunk.get("chunk_id"),
                            "corpus_id": chunk.get("corpus_id"),
                            "suffix": chunk.get("suffix"),
                            "content": chunk.get("content", ""),
                            "score": 0.0,  # Default score (đã được scoring từ retrieved docs)
                        }
                        full_segments.append(seg)
                    
                    article_copy["segments_full"] = full_segments
                    article_copy["segments_count"] = len(full_segments)
                    result_articles.append(article_copy)
                else:
                    # Fallback: giữ segments cũ nếu không lấy được đầy đủ
                    result_articles.append(article)
            except Exception as e:
                # Fallback: giữ segments cũ nếu có lỗi
                logger.warning(f"Failed to fetch full segments for {code}+{article_num}: {e}")
                result_articles.append(article)
        
        return result_articles

    def _articles_to_doc_infos(self, filtered_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chuyển danh sách articles thành doc_infos (grouped by law)."""
        doc_map: Dict[str, Dict[str, Any]] = {}
        
        for article in filtered_articles:
            code = article.get("act_code", "")
            if not code:
                continue
            
            doc_info = doc_map.setdefault(code, {
                "act_code": code,
                "corpus_id": "",
                "articles": [],
                "doc_score": 0.0,
                "best_score": None,
            })
            
            # Thêm article vào doc_info
            art_score = float(article.get("score", 0.0))
            doc_info["articles"].append({
                "article": article.get("article"),
                "score": art_score,
                "segments": article.get("segments", []),
            })
            
            # Update scores
            doc_info["doc_score"] = max(doc_info.get("doc_score", 0.0), art_score)
            if doc_info["best_score"] is None or art_score > doc_info["best_score"]:
                doc_info["best_score"] = art_score
            
            # Set corpus_id từ segment đầu tiên
            if not doc_info["corpus_id"]:
                segs = article.get("segments", [])
                if segs:
                    doc_info["corpus_id"] = segs[0].get("corpus_id", "")
        
        doc_list = list(doc_map.values())
        # Sort by score
        doc_list.sort(key=lambda d: (-d.get("doc_score", 0.0), -(d.get("best_score", 0.0) if d.get("best_score") is not None else -1e9)))
        
        return doc_list

    def _flatten_segments(self, doc_infos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        flattened: List[Dict[str, Any]] = []
        for doc in doc_infos:
            for art in doc.get("articles", []):
                # Ưu tiên segments_full (toàn bộ segments từ database)
                # Nếu không có, dùng segments cũ (retrieved segments)
                segments = art.get("segments_full") or art.get("segments", [])
                flattened.extend(segments)
        return flattened

    def _build_sources_grouped(self, doc_infos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: List[Dict[str, Any]] = []
        for doc in doc_infos:
            grouped.append({
                "act_code": doc.get("act_code"),
                "corpus_id": doc.get("corpus_id"),
                "articles": [art.get("article") for art in doc.get("articles", []) if art.get("article") is not None],
                "article_count": len(doc.get("articles", [])),
                "group_score": doc.get("doc_score"),
                "best_score": doc.get("best_score"),
            })
        return grouped
    
    def generate_response(self, question: str, context: str = None, **kwargs) -> str:
        """Generate a response using Gemini"""
        try:
            # Ensure Gemini model is initialized at call-time. This allows
            # importing the module (e.g., in tests) without GOOGLE_API_KEY set.
            if not getattr(self, 'model', None):
                try:
                    self._initialize_gemini()
                except Exception as e:
                    # Fail gracefully: return an informative message rather than
                    # raising at import or runtime in user-facing paths.
                    print(f"Error initializing Gemini: {e}")
                    return "Xin lỗi, hệ thống chưa cấu hình mô hình ngôn ngữ. Vui lòng thiết lập GOOGLE_API_KEY."

            # Prepare the prompt
            if context:
                prompt = f"""
                Bạn là trợ lý pháp lý tiếng Việt. Trả lời CHỈ dựa vào ngữ cảnh sau.
                - KHÔNG chèn mã nguồn hay corpus-id vào phần trả lời. KHÔNG dùng ngoặc đơn để liệt kê mã nguồn.
                - Hạn chế suy diễn. Chỉ khi ngữ cảnh không nêu quy định trực tiếp mới nói "Không đủ căn cứ trong nguồn đã trích" và gợi ý văn bản cần tra thêm.
                - Câu trả lời ngắn gọn, 3-5 gạch đầu dòng, dùng ngôn ngữ tự nhiên, dễ hiểu.

                Ngữ cảnh (đã kèm corpus-id):
                {context}

                Câu hỏi: {question}
                """
            else:
                prompt = f"""
                Bạn là một trợ lý pháp lý thông minh. Hãy trả lời câu hỏi sau đây:
                
                Câu hỏi: {question}
                
                Nếu bạn không chắc chắn về câu trả lời, hãy nói rõ điều đó.
                """
            
            # Generate response
            response = self.model.generate_content(prompt)

            # Return the generated text
            return response.text
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại sau."
    
    def ask(self, question: str, top_k: int = 3, detail_level: str = "moderate") -> Dict[str, Any]:
        """Process a question and return the answer with sources
        
        Args:
            question: Câu hỏi
            top_k: (deprecated) Giữ để backward compatible
            detail_level: "brief" | "moderate" (default) | "comprehensive"
                        Quyết định số articles scan để capture toàn bộ articles liên quan
        """
        start_time = time.time()
        
        try:
            # Step 1: Determine target articles to scan (oversample by articles, not segments)
            # Mục đích: lấy đủ segments để tìm toàn bộ articles liên quan
            oversample_by_articles = {
                "brief": 10,           # Scan ~10 articles
                "moderate": 20,        # Scan ~20 articles (MẶC ĐỊNH)
                "comprehensive": 50,   # Scan ~50 articles
            }
            target_articles_to_scan = oversample_by_articles.get(detail_level.lower(), 20)
            
            # Estimate segments: avg article ≈ 3 segments
            avg_segments_per_article = 3
            try:
                avg_segments_per_article = int(os.getenv('LEGALADVISOR_AVG_SEGMENTS_PER_ARTICLE', '3'))
            except Exception:
                avg_segments_per_article = 3
            
            # Calculate chunks to retrieve
            chunk_k = target_articles_to_scan * avg_segments_per_article
            chunk_k = max(15, chunk_k)  # Tối thiểu 15 segments
            
            retrieved_docs = self.retrieve_documents(question, top_k=chunk_k)
            if not retrieved_docs:
                return {
                    'question': question,
                    'answer': "Xin lỗi, không tìm thấy nguồn phù hợp.",
                    'sources': [],
                    'num_sources': 0,
                    'num_segments': 0,
                    'sources_grouped': [],
                    'num_docs': 0,
                    'confidence': 0.0,
                    'status': 'success',
                    'processing_time': time.time() - start_time
                }

            # Step 2: Build article entries từ segments
            article_entries = self._build_article_entries(retrieved_docs)
            if not article_entries:
                return {
                    'question': question,
                    'answer': "Xin lỗi, không tìm thấy nguồn phù hợp.",
                    'sources': [],
                    'num_sources': 0,
                    'num_segments': 0,
                    'sources_grouped': [],
                    'num_docs': 0,
                    'confidence': 0.0,
                    'status': 'success',
                    'processing_time': time.time() - start_time
                }

            # Step 3: Tính toán Article Scores từ segments
            self._calculate_article_scores(article_entries)

            # Step 4: Áp dụng adaptive threshold để filter articles
            filtered_articles, threshold_used = self._apply_adaptive_threshold(article_entries, detail_level)
            
            if not filtered_articles:
                return {
                    'question': question,
                    'answer': "Xin lỗi, không tìm thấy nguồn phù hợp sau khi filter.",
                    'sources': [],
                    'num_sources': 0,
                    'num_segments': 0,
                    'num_articles': 0,
                    'sources_grouped': [],
                    'num_docs': 0,
                    'confidence': 0.0,
                    'status': 'success',
                    'processing_time': time.time() - start_time
                }

            # Step 5: Fetch TẤT CẢ segments của articles được chọn từ database
            # (không chỉ những segments ban đầu retrieve)
            filtered_articles_with_full_segments = self._fetch_all_segments_for_articles(filtered_articles)
            
            # Step 6: Chuyển articles thành document infos (grouped by law)
            doc_infos = self._articles_to_doc_infos(filtered_articles_with_full_segments)
            
            # Step 7: Flatten segments để lấy toàn bộ segments từ articles được chọn
            filtered_docs = self._flatten_segments(doc_infos)
            sources_grouped = self._build_sources_grouped(doc_infos)

            # Step 3: Xây dựng context ưu tiên toàn văn top-k doc + tài liệu trích dẫn
            context = None
            try:
                context = self._build_llm_context(filtered_docs or [], sources_grouped or [])
            except Exception:
                # Fallback an toàn
                context = format_retrieved_docs(filtered_docs) if filtered_docs else None
            
            # Step 4: Generate response using Gemini
            answer = self.generate_response(question, context)

            # Không thêm block tham khảo vào câu trả lời để tránh trùng với UI. UI sẽ hiển thị sources.
            
            # Prepare response
            # Tính confidence sơ bộ từ IP similarity đã chuẩn hóa: quy về [0,1]
            if filtered_docs:
                raw_scores = [float(d.get('score', 0.0)) for d in filtered_docs]
                similarities = [self._score_to_similarity(s) for s in raw_scores]
                confidence = float(np.mean(similarities))
            else:
                confidence = 0.0

            response = {
                'question': question,
                'answer': answer,
                'sources': filtered_docs,
                'num_sources': int(len(sources_grouped)),      # Số tài liệu (laws)
                'num_segments_retrieved': int(len(retrieved_docs)),  # Segments retrieve để scoring
                'num_segments': int(len(filtered_docs)),       # Segments total (toàn bộ của articles chọn)
                'num_articles': int(len(filtered_articles)),   # Số articles được chọn (qua threshold)
                'sources_grouped': sources_grouped,
                'num_docs': int(len(sources_grouped)),         # Alias cho num_sources
                'threshold_used': float(threshold_used),       # Threshold áp dụng
                'detail_level': detail_level,                 # Mức chi tiết
                'confidence': confidence,
                'status': 'success',
                'processing_time': time.time() - start_time
            }
            
            return response
            
        except Exception as e:
            return {
                'question': question,
                'answer': f"Xin lỗi, đã xảy ra lỗi: {str(e)}",
                'sources': [],
                'num_sources': 0,
                'num_segments_retrieved': 0,
                'num_segments': 0,
                'num_articles': 0,
                'threshold_used': 0.0,
                'detail_level': detail_level,
                'confidence': 0.0,
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time
            }

    def get_chunk_content(self, chunk_id: int) -> Optional[str]:
        """Trả về nội dung chunk theo id từ SQLite/Parquet (ưu tiên)."""
        return self._get_chunk_content_by_id(chunk_id)

def test_gemini_rag():
    """Test the GeminiRAG implementation"""
    try:
        print("🚀 Testing GeminiRAG...")
        
        # Initialize RAG
        rag = GeminiRAG(use_gpu=False)
        
        # Test query
        query = "Điều kiện để thành lập doanh nghiệp tư nhân?"
        print(f"\n🤖 Câu hỏi: {query}")
        
        # Get response
        response = rag.ask(query, top_k=3)
        
        # Print results
        print("\n📝 Câu trả lời:")
        print(response['answer'])
        
        print(f"\n🔍 Nguồn tham khảo ({response['num_sources']}):")
        for i, source in enumerate(response['sources'], 1):
            # Tránh KeyError: metadata hiện không có 'title'
            corpus_id = source.get('corpus_id') or '(không có corpus_id)'
            score = source.get('score', 0.0)
            print(f"{i}. {corpus_id} (Điểm: {score:.2f})")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_gemini_rag()
