#!/usr/bin/env python3
"""
PDF Processor cho Mental Health RAG Agent
Xử lý các tài liệu PDF về tư vấn tâm lý học sinh sinh viên
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import Config, MENTAL_HEALTH_KEYWORDS

class PDFProcessor:
    def __init__(self):
        """
        Khởi tạo PDF processor với cấu hình tối ưu cho nội dung tâm lý
        """
        print(f"🔧 Khởi tạo PDF Processor cho domain: {Config.DOMAIN}")
        
        # Text splitter với cấu hình tối ưu cho nội dung tâm lý
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            # Separators tối ưu cho văn bản tiếng Việt về tâm lý
            separators=[
                "\n\n\n",  # Section breaks
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentence ends
                "。",      # Vietnamese sentence end
                "! ",      # Exclamation
                "? ",      # Question
                "; ",      # Semicolon
                ", ",      # Comma
                " "        # Space
            ]
        )
        
        print(f"✅ PDF Processor sẵn sàng với chunk size: {Config.CHUNK_SIZE}")
    
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
                        
                        text += f"\n--- Trang {page_num + 1} ---\n{page_text}\n"
                        
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
        
        # Xử lý số trang và header/footer
        text = re.sub(r'Trang \d+', '', text)
        text = re.sub(r'Page \d+', '', text)
        
        # Loại bỏ các ký tự lặp lại không cần thiết
        text = re.sub(r'[_-]{3,}', '', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        return text.strip()
    
    def enhance_mental_health_content(self, text: str) -> str:
        """
        Không thêm keyword enhancement - để semantic model tự học
        """
        # Return text gốc, không thêm artificial markers
        return text
    
    def create_chunks(self, text: str, source_file: str) -> List[Dict]:
        """
        Chia text thành chunks với metadata cho tâm lý học
        """
        if not text.strip():
            return []
        
        print(f"📝 Đang chia text thành chunks...")
        print(f"   Text length: {len(text)} ký tự")
        
        # Tăng cường nội dung trước khi chia chunks
        enhanced_text = self.enhance_mental_health_content(text)
        
        # Tạo Document object
        doc = Document(
            page_content=enhanced_text,
            metadata={
                "source": source_file,
                "domain": Config.DOMAIN,
                "type": "mental_health_document"
            }
        )
        
        # Chia thành chunks
        chunks = self.text_splitter.split_documents([doc])
        
        # Chuyển đổi thành format dictionary với metadata đầy đủ
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            # Phân loại nội dung chunk
            content_type = self.classify_chunk_content(chunk.page_content)
            
            chunk_dict = {
                "content": chunk.page_content,
                "source": source_file,
                "chunk_id": i,
                "content_type": content_type,
                "metadata": {
                    **chunk.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk.page_content),
                    "word_count": len(chunk.page_content.split()),
                    "content_classification": content_type
                }
            }
            chunk_dicts.append(chunk_dict)
        
        print(f"✅ Đã tạo {len(chunk_dicts)} chunks")
        return chunk_dicts
    
    def classify_chunk_content(self, content: str) -> str:
        """
        Phân loại nội dung chunk đơn giản - tránh bias từ keyword matching
        """
        # Tất cả content đều được classify là general để semantic search tự quyết định
        return "general_content"
    
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
                
                # In thống kê
                content_types = {}
                for chunk in chunks:
                    ctype = chunk["content_type"]
                    content_types[ctype] = content_types.get(ctype, 0) + 1
                
                print(f"📊 Thống kê nội dung:")
                for ctype, count in content_types.items():
                    print(f"   - {ctype}: {count} chunks")
            
            return chunks
            
        except Exception as e:
            print(f"❌ Lỗi xử lý PDF {pdf_path}: {e}")
            return []
    
    def get_processing_stats(self, chunks: List[Dict]) -> Dict:
        """
        Lấy thống kê xử lý
        """
        if not chunks:
            return {"total_chunks": 0}
        
        stats = {
            "total_chunks": len(chunks),
            "total_characters": sum(len(chunk["content"]) for chunk in chunks),
            "total_words": sum(chunk["metadata"]["word_count"] for chunk in chunks),
            "content_types": {},
            "sources": list(set(chunk["source"] for chunk in chunks))
        }
        
        # Thống kê theo loại nội dung
        for chunk in chunks:
            ctype = chunk["content_type"]
            stats["content_types"][ctype] = stats["content_types"].get(ctype, 0) + 1
        
        return stats
