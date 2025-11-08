#!/usr/bin/env python3
"""
Zalo-AI-Legal Data Preprocessing Module

Handles preprocessing of Zalo-AI-Legal corpus for legal document retrieval.

Features:
- Parse corpus_id to extract document type, number, year, suffix
- Process JSONL corpus format
- Generate standardized chunk schema with metadata

Schema Output:
    {
        'chunk_id': int,          # Sequential index
        'corpus_id': str,         # Document identifier (e.g., '159/2020/nđ-cp+13')
        'type': str,              # Document type (e.g., 'nđ-cp', 'tt-bnn')
        'number': str,            # Document number (e.g., '159/2020')
        'year': str,              # Year extracted from number
        'suffix': str,            # Suffix after + (e.g., '13')
        'content': str,           # Full document content
        'word_count': int,        # Word count
        'preview': str            # First 200 chars preview
    }

Author: LegalAdvisor Team
Date: 2025-11-07
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


def parse_corpus_id(corpus_id: str) -> Tuple[str, str, str, str]:
    """
    Parse corpus_id to extract document type, number, year, and suffix.
    
    Examples:
        '159/2020/nđ-cp+13' → ('nđ-cp', '159/2020', '2020', '13')
        '47/2011/tt-bca' → ('tt-bca', '47/2011', '2011', '')
        '01/2009/tt-bnn+1' → ('tt-bnn', '01/2009', '2009', '1')
    
    Args:
        corpus_id: Document identifier in format NUMBER/YEAR/TYPE[+SUFFIX]
    
    Returns:
        Tuple of (type, number, year, suffix)
        Returns empty strings if parsing fails
    """
    # Pattern: NUMBER/YEAR/TYPE[+SUFFIX]
    # Example: 159/2020/nđ-cp+13
    pattern = r'(\d{2,4}/\d{4})/([\w-]+)(\+\d+)?'
    match = re.match(pattern, corpus_id)
    
    if match:
        number = match.group(1)  # e.g., '159/2020'
        type_ = match.group(2).lower()  # e.g., 'nđ-cp'
        suffix = match.group(3)[1:] if match.group(3) else ''  # e.g., '13' (remove '+')
        year = number.split('/')[-1]  # e.g., '2020'
        
        return type_, number, year, suffix
    else:
        # Fallback: return empty values
        return '', '', '', ''


def load_and_parse_corpus(
    input_path: str,
    corpus_id_key: str = '_id',
    content_key: str = 'text'
) -> List[Dict]:
    """
    Load and parse JSONL corpus file with progress bar.
    
    Args:
        input_path: Path to input JSONL file
        corpus_id_key: JSON key for corpus ID (default: '_id')
        content_key: JSON key for content (default: 'text')
    
    Returns:
        List of processed chunks with standardized schema
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If JSONL line is invalid
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    result = []

    # Đọc một lần; để tqdm tự đếm nếu cần (không mở file lần 2)
    with open(input_path, encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(
            f,
            desc='Đọc và parse corpus',
            unit='lines'
        )):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  Lỗi parse JSON tại dòng {idx}: {e}")
                continue
            
            # Extract corpus_id and content
            corpus_id = data.get(corpus_id_key, '')
            content = data.get(content_key, '')
            
            if not corpus_id or not content:
                print(f"⚠️  Bỏ qua dòng {idx}: corpus_id hoặc content trống")
                continue
            
            # Parse corpus_id to extract metadata
            type_, number, year, suffix = parse_corpus_id(corpus_id)
            
            # Calculate word count and preview
            word_count = len(content.split()) if content else 0
            preview = content[:200] if content else ''
            
            # Build standardized chunk
            chunk = {
                'chunk_id': len(result),  # Use result length as sequential ID
                'corpus_id': corpus_id,
                'type': type_,
                'number': number,
                'year': year,
                'suffix': suffix,
                'content': content,
                'word_count': word_count,
                'preview': preview
            }
            
            result.append(chunk)
    
    return result


def save_schema_jsonl(chunks: List[Dict], output_path: str) -> int:
    """
    Save processed chunks to JSONL file with standardized schema.
    
    Args:
        chunks: List of chunk dictionaries
        output_path: Path to output JSONL file
    
    Returns:
        Number of chunks written
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    return len(chunks)


def preprocess_corpus(
    input_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    corpus_id_key: str = '_id',
    content_key: str = 'text'
) -> Tuple[str, int]:
    """
    Full preprocessing pipeline: Load → Parse → Save.
    
    Args:
        input_path: Path to input JSONL (default: data/raw/zalo_ai_legal_text_retrieval/corpus.jsonl)
        output_dir: Output directory (default: data/processed/zalo-legal)
        corpus_id_key: JSON key for corpus ID
        content_key: JSON key for content
    
    Returns:
        Tuple of (output_file_path, num_chunks)
    """
    # Set default paths relative to project root
    if input_path is None:
        input_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            'data',
            'raw',
            'zalo_ai_legal_text_retrieval',
            'corpus.jsonl'
        )
    
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            'data',
            'processed',
            'zalo-legal'
        )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build output file path
    output_path = os.path.join(output_dir, 'chunks_schema.jsonl')
    
    # Run pipeline
    print(f"📖 Đọc corpus từ: {input_path}")
    chunks = load_and_parse_corpus(input_path, corpus_id_key, content_key)
    
    print(f"💾 Lưu schema vào: {output_path}")
    num_chunks = save_schema_jsonl(chunks, output_path)
    
    print(f"✅ Đã xử lý thành công: {num_chunks} chunks")
    
    return output_path, num_chunks


if __name__ == '__main__':
    """
    CLI entry point for preprocessing.
    
    Usage:
        python -m src.data_preprocessing.zalo_legal
    """
    try:
        output_file, num_chunks = preprocess_corpus()
        print(f"\n🎉 Kết quả:")
        print(f"   Output: {output_file}")
        print(f"   Chunks: {num_chunks}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
