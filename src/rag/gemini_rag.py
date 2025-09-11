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

import google.generativeai as genai
from dotenv import load_dotenv
from retrieval.service import RetrievalService

# Load environment variables
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
GEMINI_MODEL = "gemini-2.0-flash-exp"

def format_retrieved_docs(docs: List[Dict[str, Any]]) -> str:
    """Format retrieved documents với tên luật + điều/khoản/điểm và tóm tắt ngắn."""
    formatted_docs: List[str] = []
    for i, doc in enumerate(docs, 1):
        base = doc.get('law_title') or doc.get('title') or doc.get('doc_file')
        eff_year = doc.get('effective_year')
        law_title = f"{base} ({int(eff_year)})" if eff_year else base
        article = doc.get('article')
        clause = doc.get('clause')
        point = doc.get('point')
        labels = []
        if article:
            labels.append(f"Điều {article}")
        if clause:
            labels.append(f"Khoản {clause}")
        if point:
            labels.append(f"Điểm {point}")
        label_str = ' - '.join(labels)

        content = (doc.get('content') or '').replace('_', ' ').strip()
        snippet = content[:400]
        suffix = '...' if len(content) > 400 else ''

        formatted_docs.append(
            f"[Nguồn {i}] {law_title}{(' - ' + label_str) if label_str else ''}\n{snippet}{suffix}\n(điểm: {doc.get('score', 0):.2f})"
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
            # Initialize the Gemini model
            generation_config = {
                "temperature": 0.2,  # Lower temperature for more focused answers
                "top_p": 0.95,
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
            # Prepare the prompt
            if context:
                prompt = f"""
                Bạn là trợ lý pháp lý. Dựa trên ngữ cảnh trích dẫn từ các bộ luật dưới đây,
                hãy trả lời ngắn gọn, súc tích, dễ đọc dưới dạng gạch đầu dòng, kèm điều/khoản/điểm liên quan.

                Ngữ cảnh (đã trích nguồn):
                {context}

                Câu hỏi: {question}

                Yêu cầu:
                - Tổng hợp ý chính (tối đa 3-5 gạch đầu dòng), tránh lặp lại nguyên văn dài dòng.
                - Trích dẫn nguồn theo dạng: (Tên luật – Điều X[, Khoản Y[, Điểm Z]]).
                - Nếu không đủ thông tin, nêu rõ cần tham khảo thêm điều nào.
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

            # Bổ sung phần "Nguồn tham khảo" kèm Điều/Khoản/Điểm (đảm bảo luôn có trong câu trả lời)
            if retrieved_docs:
                lines = []
                for i, d in enumerate(retrieved_docs, 1):
                    base = d.get('law_title') or d.get('title') or d.get('doc_file')
                    ey = d.get('effective_year')
                    law_title = f"{base} ({int(ey)})" if ey else base
                    parts = []
                    if d.get('article'):
                        parts.append(f"Điều {d.get('article')}")
                    if d.get('clause'):
                        parts.append(f"Khoản {d.get('clause')}")
                    if d.get('point'):
                        parts.append(f"Điểm {d.get('point')}")
                    label = ' – '.join(parts)
                    if label:
                        lines.append(f"{i}. {law_title} – {label}")
                    else:
                        lines.append(f"{i}. {law_title}")
                citations_block = "\n".join(lines)
                answer = f"{answer}\n\nNguồn tham khảo:\n{citations_block}"
            
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
            print(f"{i}. {source['title']} (Điểm: {source['score']:.2f})")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_gemini_rag()
