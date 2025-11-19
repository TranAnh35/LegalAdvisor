#!/usr/bin/env python3
"""
Benchmark script để kiểm tra hiệu suất của Tier 1 optimizations:
1. Indexed JSONL cache
2. Parallel segment fetching

Usage:
    conda activate LegalAdvisor
    python scripts/benchmark_optimization_tier1.py
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.gemini_rag import GeminiRAG
from src.utils.logger import get_logger

logger = get_logger("benchmark")

def benchmark_question(rag: GeminiRAG, question: str, top_k: int = 3) -> Dict[str, Any]:
    """Benchmark một câu hỏi và trả về thời gian xử lý chi tiết."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 BENCHMARKING: {question[:80]}")
    logger.info(f"{'='*60}")
    
    # Phase 1: Retrieve documents
    logger.info("[Phase 1] Retrieving documents...")
    t0 = time.time()
    try:
        retrieved = rag.retrieve_documents(question, top_k=30)  # Retrieve 30 segments
        t_retrieve = time.time() - t0
        num_segments_retrieved = len(retrieved)
        logger.info(f"✅ Retrieved {num_segments_retrieved} segments in {t_retrieve:.2f}s")
    except Exception as e:
        logger.error(f"❌ Retrieval failed: {e}")
        return {"error": str(e)}
    
    # Phase 2: Ask (includes context building with parallel fetching)
    logger.info("[Phase 2] Building context (parallel fetching)...")
    t0 = time.time()
    try:
        response = rag.ask(question, top_k=3, detail_level="moderate")
        t_ask = time.time() - t0
        
        processing_time = response.get('processing_time', 0)
        num_sources = response.get('num_sources', 0)
        num_segments = response.get('num_segments', 0)
        num_articles = response.get('num_articles', 0)
        
        logger.info(f"✅ Ask completed in {t_ask:.2f}s")
        logger.info(f"   - Processing time (internal): {processing_time:.2f}s")
        logger.info(f"   - Sources: {num_sources}")
        logger.info(f"   - Articles: {num_articles}")
        logger.info(f"   - Segments: {num_segments}")
        logger.info(f"   - Confidence: {response.get('confidence', 0):.2%}")
        
        return {
            "question": question,
            "t_retrieve": t_retrieve,
            "t_ask": t_ask,
            "t_total": t_retrieve + t_ask,
            "num_segments_retrieved": num_segments_retrieved,
            "num_sources": num_sources,
            "num_articles": num_articles,
            "num_segments": num_segments,
            "confidence": response.get('confidence', 0),
            "status": response.get('status', 'unknown'),
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"❌ Ask failed: {e}")
        return {"error": str(e)}

def main():
    """Main benchmark function."""
    
    logger.info("\n" + "="*60)
    logger.info("🚀 TIER 1 OPTIMIZATION BENCHMARK")
    logger.info("="*60)
    logger.info("Testing: Indexed JSONL cache + Parallel segment fetching\n")
    
    # Initialize RAG
    logger.info("[Setup] Initializing GeminiRAG...")
    try:
        rag = GeminiRAG(use_gpu=False)
        logger.info("✅ GeminiRAG initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize GeminiRAG: {e}")
        return
    
    # Verify cache is loaded
    if hasattr(rag.retriever, '_all_records_cached'):
        logger.info(f"✅ JSONL indexed cache: {'ENABLED' if rag.retriever._all_records_cached else 'DISABLED'}")
        if hasattr(rag.retriever, '_chunk_cache'):
            logger.info(f"   Cache size: {len(rag.retriever._chunk_cache)} records")
    
    # Test questions (vary complexity)
    test_questions = [
        "Điều kiện để thành lập công ty cổ phần là gì?",
        "Quy định về trách nhiệm của nhân viên công ty?",
        "Thủ tục giải thể công ty như thế nào?",
    ]
    
    results: List[Dict[str, Any]] = []
    total_time = 0
    
    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n[Test {i}/{len(test_questions)}]")
        result = benchmark_question(rag, question, top_k=3)
        results.append(result)
        
        if 'error' not in result:
            total_time += result['t_total']
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📈 SUMMARY")
    logger.info("="*60)
    
    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        logger.error("❌ No valid results")
        return
    
    avg_retrieve = sum(r['t_retrieve'] for r in valid_results) / len(valid_results)
    avg_ask = sum(r['t_ask'] for r in valid_results) / len(valid_results)
    avg_total = sum(r['t_total'] for r in valid_results) / len(valid_results)
    avg_segments = sum(r['num_segments'] for r in valid_results) / len(valid_results)
    avg_articles = sum(r['num_articles'] for r in valid_results) / len(valid_results)
    
    logger.info(f"\n⏱️  TIMINGS (Average across {len(valid_results)} questions):")
    logger.info(f"   - Retrieve: {avg_retrieve:.2f}s")
    logger.info(f"   - Ask (parallel fetch): {avg_ask:.2f}s")
    logger.info(f"   - Total: {avg_total:.2f}s")
    logger.info(f"\n📊 RESULTS:")
    logger.info(f"   - Avg Segments per result: {avg_segments:.0f}")
    logger.info(f"   - Avg Articles per result: {avg_articles:.0f}")
    
    logger.info(f"\n💡 OPTIMIZATION IMPACT:")
    logger.info(f"   ✅ Indexed JSONL cache: O(1) chunk lookup (vs O(n) scan)")
    logger.info(f"   ✅ Parallel fetching: Multi-threaded article loading")
    logger.info(f"   🎯 Expected improvement: -30s (indexed cache) + -15s (parallel) = ~-45s\n")
    
    # Save results to file
    results_file = Path(__file__).parent.parent / "results" / "benchmark_tier1.json"
    results_file.parent.mkdir(exist_ok=True)
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": len(valid_results),
        "avg_retrieve_time": avg_retrieve,
        "avg_ask_time": avg_ask,
        "avg_total_time": avg_total,
        "avg_segments": avg_segments,
        "avg_articles": avg_articles,
        "detailed_results": valid_results,
        "notes": "Tier 1 optimizations: Indexed JSONL cache + Parallel segment fetching"
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📁 Results saved to: {results_file}")

if __name__ == "__main__":
    main()

