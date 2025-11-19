"""
Unit tests for data_preprocessing.zalo_legal module

Tests:
- parse_corpus_id() with various formats
- load_and_parse_corpus() with sample data
- save_schema_jsonl() to JSONL format
- Full pipeline preprocess_corpus()
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.data_preprocessing.zalo_legal import (
    parse_corpus_id,
    load_and_parse_corpus,
    save_schema_jsonl,
    preprocess_corpus
)


class TestParseCorpusId(unittest.TestCase):
    """Test corpus_id parsing"""
    
    def test_full_format_with_suffix(self):
        """Test parsing: NUMBER/YEAR/TYPE+SUFFIX"""
        type_, number, year, suffix = parse_corpus_id('159/2020/nđ-cp+13')
        self.assertEqual(type_, 'nđ-cp')
        self.assertEqual(number, '159/2020')
        self.assertEqual(year, '2020')
        self.assertEqual(suffix, '13')
    
    def test_format_without_suffix(self):
        """Test parsing: NUMBER/YEAR/TYPE (no suffix)"""
        type_, number, year, suffix = parse_corpus_id('47/2011/tt-bca')
        self.assertEqual(type_, 'tt-bca')
        self.assertEqual(number, '47/2011')
        self.assertEqual(year, '2011')
        self.assertEqual(suffix, '')
    
    def test_various_types(self):
        """Test various document types"""
        test_cases = [
            ('01/2009/tt-bnn+1', ('tt-bnn', '01/2009', '2009', '1')),
            ('10/2020/nđ-cp', ('nđ-cp', '10/2020', '2020', '')),
            ('100/2023/qh14+5', ('qh14', '100/2023', '2023', '5')),
        ]
        for corpus_id, expected in test_cases:
            result = parse_corpus_id(corpus_id)
            self.assertEqual(result, expected, f"Failed for {corpus_id}")
    
    def test_invalid_format(self):
        """Test invalid corpus_id formats"""
        invalid_ids = [
            'invalid',
            '2020/nđ-cp',  # Wrong order
            '159/nđ-cp',   # Missing year
            '',
        ]
        for corpus_id in invalid_ids:
            result = parse_corpus_id(corpus_id)
            self.assertEqual(result, ('', '', '', ''), f"Should return empty for {corpus_id}")


class TestLoadAndParseCorpus(unittest.TestCase):
    """Test corpus loading and parsing"""
    
    def setUp(self):
        """Create temporary JSONL file for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_jsonl = os.path.join(self.temp_dir, 'test_corpus.jsonl')
        
        # Create sample corpus data
        sample_data = [
            {
                '_id': '159/2020/nđ-cp+13',
                'text': 'Điều 1: Quy định về quyền của công dân. ' * 10  # ~100 words
            },
            {
                '_id': '47/2011/tt-bca',
                'text': 'Thông tư này quy định về thủ tục hành chính. ' * 8  # ~80 words
            }
        ]
        
        with open(self.test_jsonl, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_valid_corpus(self):
        """Test loading valid JSONL corpus"""
        chunks = load_and_parse_corpus(self.test_jsonl)
        
        self.assertEqual(len(chunks), 2)
        
        # Check first chunk
        self.assertEqual(chunks[0]['chunk_id'], 0)
        self.assertEqual(chunks[0]['corpus_id'], '159/2020/nđ-cp+13')
        self.assertEqual(chunks[0]['type'], 'nđ-cp')
        self.assertEqual(chunks[0]['number'], '159/2020')
        self.assertEqual(chunks[0]['year'], '2020')
        self.assertEqual(chunks[0]['suffix'], '13')
        self.assertGreater(chunks[0]['word_count'], 0)
        self.assertEqual(len(chunks[0]['preview']), 200)
    
    def test_missing_file(self):
        """Test error handling for missing file"""
        with self.assertRaises(FileNotFoundError):
            load_and_parse_corpus('/nonexistent/path/corpus.jsonl')


class TestSaveSchemaJsonl(unittest.TestCase):
    """Test JSONL schema saving"""
    
    def setUp(self):
        """Create temporary directory for output"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_chunks(self):
        """Test saving chunks to JSONL"""
        chunks = [
            {
                'chunk_id': 0,
                'corpus_id': '159/2020/nđ-cp+13',
                'type': 'nđ-cp',
                'number': '159/2020',
                'year': '2020',
                'suffix': '13',
                'content': 'Test content 1',
                'word_count': 2,
                'preview': 'Test content 1'
            },
            {
                'chunk_id': 1,
                'corpus_id': '47/2011/tt-bca',
                'type': 'tt-bca',
                'number': '47/2011',
                'year': '2011',
                'suffix': '',
                'content': 'Test content 2',
                'word_count': 2,
                'preview': 'Test content 2'
            }
        ]
        
        output_path = os.path.join(self.temp_dir, 'output.jsonl')
        num_saved = save_schema_jsonl(chunks, output_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(num_saved, 2)
        
        # Verify content
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 2)
        
        # Verify each line is valid JSON
        for i, line in enumerate(lines):
            data = json.loads(line)
            self.assertEqual(data['chunk_id'], i)


class TestPreprocessPipeline(unittest.TestCase):
    """Test full preprocessing pipeline"""
    
    def setUp(self):
        """Create temporary files for testing"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create input directory and file
        self.input_dir = os.path.join(self.temp_dir, 'raw', 'zalo_ai_legal_text_retrieval')
        os.makedirs(self.input_dir, exist_ok=True)
        
        self.input_file = os.path.join(self.input_dir, 'corpus.jsonl')
        
        # Create sample corpus
        sample_data = [
            {'_id': '159/2020/nđ-cp+13', 'text': 'Content 1 ' * 20},
            {'_id': '47/2011/tt-bca', 'text': 'Content 2 ' * 20}
        ]
        
        with open(self.input_file, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        self.output_dir = os.path.join(self.temp_dir, 'processed', 'zalo-legal')
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline(self):
        """Test complete preprocessing pipeline"""
        output_file, num_chunks = preprocess_corpus(
            input_path=self.input_file,
            output_dir=self.output_dir
        )
        
        # Verify output file exists
        self.assertTrue(os.path.exists(output_file))
        self.assertEqual(num_chunks, 2)
        
        # Verify output format
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                self.assertIn('chunk_id', data)
                self.assertIn('corpus_id', data)
                self.assertIn('type', data)
                self.assertIn('content', data)


if __name__ == '__main__':
    unittest.main()
