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

from dotenv import load_dotenv
from ..retrieval.service import RetrievalService

# Do NOT initialize google.generativeai at import time.
# Gemini (google-generativeai) will be imported and configured lazily
# inside GeminiRAG._initialize_gemini() to avoid raising on module import
# when GOOGLE_API_KEY is not present (improves testability and CI).
GEMINI_MODEL = "gemini-2.0-flash-exp"

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
    """Format retrieved documents với tên luật + điều/khoản/điểm và tóm tắt ngắn.

    - Giữ nguyên dấu '_' trong content để khớp embedding, nhưng chỉ khi không ảnh hưởng đọc hiểu.
    - Tăng snippet lên 1200 ký tự để cung cấp ngữ cảnh đầy đủ hơn.
    """
    formatted_docs: List[str] = []
    for i, doc in enumerate(docs, 1):
        corpus_id = doc.get('corpus_id') or ''
        type_code = doc.get('type') or ''
        number = doc.get('number') or ''
        year = doc.get('year') or ''
        suffix = doc.get('suffix') or ''
        dieu = f"Điều {suffix}" if str(suffix).isdigit() else ''

        law_title = _vietnamese_doc_title(type_code, number)

        content = (doc.get('content') or '').strip()
        # Hiển thị thân thiện: thay '_' bằng ' ' chỉ trong phần snippet để dễ đọc
        snippet = content[:1200].replace('_', ' ')
        suffix = '...' if len(content) > 1200 else ''

        formatted_docs.append(
            f"[Nguồn {i}] {law_title}{(' - ' + dieu) if dieu else ''} — `{corpus_id}`\n{snippet}{suffix}\n(điểm: {doc.get('score', 0):.2f})"
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
        
        print("✅ GeminiRAG initialized successfully!")
    
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
                "max_output_tokens": 2048,
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

    def _get_chunk_content_by_id(self, chunk_id: int) -> Optional[str]:
        """Đọc content theo chunk_id từ SQLite hoặc Parquet (lazy)."""
        try:
            if not self.retriever:
                return None
            return self.retriever.get_chunk_content(int(chunk_id))
        except Exception:
            return None
    
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
    
    def ask(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Process a question and return the answer with sources"""
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retrieve_documents(question, top_k=top_k)
            
            # Step 2: Format context from retrieved documents
            context = format_retrieved_docs(retrieved_docs) if retrieved_docs else None
            
            # Step 3: Generate response using Gemini
            answer = self.generate_response(question, context)

            # Không thêm block tham khảo vào câu trả lời để tránh trùng với UI. UI sẽ hiển thị sources.
            
            # Prepare response
            response = {
                'question': question,
                'answer': answer,
                'sources': retrieved_docs,
                'num_sources': int(len(retrieved_docs)),
                'status': 'success',
                'processing_time': time.time() - start_time
            }
            
            return response
            
        except Exception as e:
            return {
                'question': question,
                'answer': f"Xin lỗi, đã xảy ra lỗi: {str(e)}",
                'sources': [],
                'num_sources': int(0),
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
