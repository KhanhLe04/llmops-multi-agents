#!/usr/bin/env python3
"""
PDF Processor cho Mental Health RAG Agent
Xử lý các tài liệu PDF về tư vấn tâm lý học sinh sinh viên
"""

import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import Config

class PDFProcessor:
    def __init__(self):
        """
        Khởi tạo PDF processor
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=[
                "\n\n\n",  # Section breaks
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentence ends
                "! ",      # Exclamation
                "? ",      # Question
                "; ",      # Semicolon
                ", ",      # Comma
                " "        # Space
            ]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Trích xuất text từ PDF với xử lý encoding cho tiếng Việt
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                print(f"📖 Đang đọc PDF: {pdf_path}")
                print(f"   Số trang: {len(pdf_reader.pages)}")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        
                        # Xử lý encoding và formatting cho tiếng Việt
                        page_text = self.clean_vietnamese_text(page_text)
                        
                        # Thêm page marker để xử lý sau
                        text += f"\n---PAGE_{page_num + 1}---\n{page_text}\n"
                        
                    except Exception as e:
                        print(f"⚠️  Lỗi đọc trang {page_num + 1}: {e}")
                        continue
                
                print(f"✅ Đã trích xuất {len(text)} ký tự từ {len(pdf_reader.pages)} trang")
                return text
                
        except Exception as e:
            print(f"❌ Lỗi đọc PDF {pdf_path}: {e}")
            return ""
    
    def clean_vietnamese_text(self, text: str) -> str:
        """
        Làm sạch và chuẩn hóa text tiếng Việt cho domain tâm lý
        """
        if not text:
            return ""
        
        # Xóa các ký tự không mong muốn
        text = re.sub(r'\x00', '', text)  # Null characters
        text = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)  # Control characters
        
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
        
        # Chuẩn hóa dấu câu tiếng Việt
        text = re.sub(r'\s*\.\s*', '. ', text)
        text = re.sub(r'\s*,\s*', ', ', text)
        text = re.sub(r'\s*;\s*', '; ', text)
        text = re.sub(r'\s*:\s*', ': ', text)
        text = re.sub(r'\s*\?\s*', '? ', text)
        text = re.sub(r'\s*!\s*', '! ', text)
        
        # Xử lý số trang và header/footer (giữ lại page markers để xử lý riêng)
        text = re.sub(r'(?<!---PAGE_)\bTrang \d+\b', '', text)
        text = re.sub(r'(?<!---PAGE_)\bPage \d+\b', '', text)
        
        # Loại bỏ các ký tự lặp lại không cần thiết
        text = re.sub(r'[_-]{3,}', '', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        return text.strip()
    
    def extract_section_from_content(self, content: str) -> str:
        """
        Trích xuất section/mục lục từ nội dung chunk
        """
        # Tìm các pattern section headers
        section_patterns = [
            r'^([IVX]+\.\s*.+?)(?:\n|$)',  # Roman numerals: I. II. III.
            r'^(\d+\.\s*.+?)(?:\n|$)',     # Numbers: 1. 2. 3.
            r'^([A-Z]\.\s*.+?)(?:\n|$)',   # Letters: A. B. C.
            r'^(CHƯƠNG\s+\d+.+?)(?:\n|$)', # Vietnamese chapters
            r'^(Phần\s+\d+.+?)(?:\n|$)',   # Vietnamese parts
            r'^(Mục\s+\d+.+?)(?:\n|$)',    # Vietnamese sections
        ]
        
        for pattern in section_patterns:
            match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: lấy dòng đầu tiên nếu ngắn (có thể là title)
        first_line = content.split('\n')[0].strip()
        if len(first_line) < 100 and first_line:
            return first_line
        
        return "Nội dung chung"
    
    def separate_page_numbers(self, content: str) -> tuple[str, list]:
        """
        Tách page numbers ra khỏi content
        """
        page_numbers = []
        
        # Tìm và extract page markers
        page_pattern = r'---PAGE_(\d+)---'
        matches = re.findall(page_pattern, content)
        page_numbers.extend([int(p) for p in matches])
        
        # Loại bỏ page markers khỏi content
        clean_content = re.sub(page_pattern, '', content)
        
        # Loại bỏ các số trang còn sót lại
        clean_content = re.sub(r'\n\s*---\s*Trang\s+\d+\s*---\s*\n', '\n', clean_content)
        clean_content = re.sub(r'\bTrang\s+\d+\b', '', clean_content)
        
        # Cleanup whitespace
        clean_content = re.sub(r'\n\s*\n\s*\n', '\n\n', clean_content)
        clean_content = clean_content.strip()
        
        return clean_content, page_numbers
    
    def create_chunks(self, text: str, source_file: str) -> List[Dict]:
        """
        Chia text thành chunks với metadata đơn giản
        """
        if not text.strip():
            return []
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Tạo Document object
        doc = Document(
            page_content=text,
            metadata={"source": source_file}
        )
        
        # Chia thành chunks
        chunks = self.text_splitter.split_documents([doc])
        
        # Chuyển đổi thành format dictionary với metadata tối giản
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            # Tách page numbers ra khỏi content
            clean_content, page_numbers = self.separate_page_numbers(chunk.page_content)
            
            # Extract section từ content
            section = self.extract_section_from_content(clean_content)
            
            chunk_dict = {
                "content": clean_content,
                "source": source_file,
                "chunk_index": i,
                "doc_id": doc_id,
                "section": section
            }
            chunk_dicts.append(chunk_dict)
        
        return chunk_dicts
    
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Xử lý hoàn chỉnh một file PDF
        """
        if not os.path.exists(pdf_path):
            print(f"❌ File không tồn tại: {pdf_path}")
            return []
        
        print(f"🔄 Bắt đầu xử lý PDF: {Path(pdf_path).name}")
        
        try:
            # Bước 1: Trích xuất text
            text = self.extract_text_from_pdf(pdf_path)
            
            if not text.strip():
                print(f"⚠️  Không trích xuất được text từ: {pdf_path}")
                return []
            
            # Bước 2: Tạo chunks
            chunks = self.create_chunks(text, Path(pdf_path).name)
            
            if chunks:
                print(f"✅ Đã xử lý xong PDF: {len(chunks)} chunks")
                
                # In thống kê sections
                sections = {}
                for chunk in chunks:
                    section = chunk["section"]
                    sections[section] = sections.get(section, 0) + 1
                
                print(f"📊 Thống kê sections:")
                for section, count in sections.items():
                    print(f"   - {section}: {count} chunks")
            
            return chunks
            
        except Exception as e:
            print(f"❌ Lỗi xử lý PDF {pdf_path}: {e}")
            return []
    
    def get_processing_stats(self, chunks: List[Dict]) -> Dict:
        """
        Lấy thống kê xử lý đơn giản
        """
        if not chunks:
            return {"total_chunks": 0}
        
        stats = {
            "total_chunks": len(chunks),
            "total_characters": sum(len(chunk["content"]) for chunk in chunks),
            "sections": {},
            "sources": list(set(chunk["source"] for chunk in chunks)),
            "doc_ids": list(set(chunk["doc_id"] for chunk in chunks))
        }
        
        # Thống kê theo sections
        for chunk in chunks:
            section = chunk["section"]
            stats["sections"][section] = stats["sections"].get(section, 0) + 1
        
        return stats
