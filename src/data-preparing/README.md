# 📚 Data Preparing - Hệ Thống Xử Lý Dữ Liệu Mental Health RAG

## 🎯 Tổng Quan

Module `data-preparing` là thành phần cốt lõi của hệ thống LLMOps Multi-Agent, chuyên xử lý và chuẩn bị dữ liệu cho domain **tư vấn sức khỏe tâm lý học sinh sinh viên**. Module này thực hiện toàn bộ pipeline từ việc xử lý tài liệu PDF gốc đến việc lưu trữ embeddings trong vector database.

## 🏗️ Kiến Trúc Hệ Thống

```
data-preparing/
├── 📄 config.py              # Cấu hình toàn bộ hệ thống
├── 🚀 ingest_data.py          # Pipeline chính xử lý dữ liệu
├── 📊 pyproject.toml          # Dependencies và metadata
├── 🛠️  utils/                 # Các module tiện ích
│   ├── pdf_processor.py       # Xử lý PDF và chunking
│   ├── embedding_manager.py   # Quản lý embedding models
│   └── qdrant_manager.py      # Quản lý vector database
├── 🧪 benchmarks/             # Đánh giá embedding models
│   └── embedding/
│       ├── hit_at_k_benchmark.py      # Benchmark retrieval
│       ├── sts_correlation_benchmark.py # Benchmark semantic similarity
│       └── results/                   # Kết quả benchmark
└── 📁 data/                   # Thư mục chứa PDF nguồn
```

## ⚙️ Cấu Hình Hệ Thống (`config.py`)

### 🔧 Cấu Hình Cơ Bản

```python
class Config:
    # Vector Database
    QDRANT_URL = "http://localhost:6333"
    COLLECTION_NAME = "mental_health_advisor"
    
    # Embedding Model - Tối ưu cho tiếng Việt
    EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
    
    # Chunking Strategy - Tối ưu cho nội dung tâm lý
    CHUNK_SIZE = 800           # Kích thước chunk phù hợp
    CHUNK_OVERLAP = 150        # Overlap để bảo toàn ngữ cảnh
    
    # Retrieval Settings
    TOP_K_DOCUMENTS = 3        # Số document trả về
    SIMILARITY_THRESHOLD = 0.65 # Ngưỡng độ tương đồng
```

### 🎛️ Tùy Chọn Nâng Cao

- **CHUNK_STRATEGY**: `"recursive"` - Chiến lược chia chunk thông minh
- **OVERLAP_METHOD**: `"sentence"` - Overlap theo ranh giới câu
- **NORMALIZE_EMBEDDINGS**: Chuẩn hóa vector embeddings
- **EMBEDDING_BATCH_SIZE**: Kích thước batch cho embedding (tối ưu cho Vietnamese models)

## 🔄 Pipeline Xử Lý Dữ Liệu (`ingest_data.py`)

### 📋 Quy Trình Chính

```python
class MentalHealthDataIngestion:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.embedding_manager = EmbeddingManager()
        self.qdrant_manager = QdrantManager()
```

### 🚀 Các Bước Thực Hiện

1. **📖 Phân Tích PDF** (`analyze_pdf_content`)
   - Kiểm tra khả năng đọc file
   - Đánh giá độ dài nội dung
   - Validation cơ bản

2. **🔄 Xử Lý PDF** (`process_pdfs`)
   - Trích xuất text từ PDF
   - Chia thành chunks thông minh
   - Tạo metadata đơn giản và hiệu quả

3. **💾 Lưu Trữ Vector DB** (`store_in_vector_db`)
   - Tạo embeddings cho từng chunk
   - Lưu vào Qdrant với metadata
   - Thống kê và báo cáo

### 📊 Metadata Structure (Đã Tối Ưu)

```python
{
    "content": "Nội dung chunk",
    "source": "tên_file.pdf",
    "chunk_index": 0,
    "doc_id": "uuid-generated-id",
    "section": "Chương 1: Giới thiệu"
}
```

**✅ Loại bỏ các metadata thừa**: `content_type`, `char_count`, `word_count`, `chunk_id`, `total_chunks`, `domain`, `contains_crisis_keywords`, `contains_student_keywords`, `priority_level`, `tags`

## 📄 PDF Processor (`utils/pdf_processor.py`)

### 🎯 Tính Năng Chính

#### 📖 Trích Xuất Text
- **Encoding UTF-8**: Xử lý hoàn hảo tiếng Việt
- **Page Markers**: Thêm `---PAGE_x---` để theo dõi trang
- **Error Handling**: Xử lý robust các lỗi PDF

#### ✂️ Chunking Thông Minh
```python
separators = [
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
```

#### 🧹 Text Cleaning
- **Vietnamese Text Normalization**: Chuẩn hóa ký tự tiếng Việt
- **Page Number Separation**: Tách số trang khỏi nội dung chính
- **Section Extraction**: Trích xuất tiêu đề section tự động

#### 🏷️ Metadata Generation
```python
def create_chunks(self, documents: List[Document]) -> List[Dict]:
    doc_id = str(uuid.uuid4())  # Unique document ID
    
    for chunk in chunks:
        content, page_info = self.separate_page_numbers(chunk.page_content)
        section = self.extract_section_from_content(content)
        
        chunk_dict = {
            "content": content,
            "source": source_name,
            "chunk_index": chunk_index,
            "doc_id": doc_id,
            "section": section
        }
```

## 🧮 Embedding Manager (`utils/embedding_manager.py`)

### 🎯 Tính Năng Chính

#### 🤖 Model Loading với Error Handling
```python
try:
    self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
except ValueError as ve:
    if "trust_remote_code" in str(ve):
        # Auto-retry với trust_remote_code=True
        self.model = SentenceTransformer(
            Config.EMBEDDING_MODEL, 
            trust_remote_code=True
        )
```

#### 🔄 Batch Processing
- **Progressive Batch Size Reduction**: Tự động giảm batch size khi gặp lỗi
- **Memory Management**: Tối ưu sử dụng GPU/CPU memory
- **Error Recovery**: Xử lý robust các lỗi encoding

#### 📊 Text Preprocessing
```python
def preprocess_text(self, text: str) -> str:
    # Loại bỏ control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate tại sentence boundary
    if len(text) > self.max_length:
        text = self.truncate_at_sentence_boundary(text)
    
    return text
```

## 🗄️ Qdrant Manager (`utils/qdrant_manager.py`)

### 🎯 Tính Năng Chính

#### 🔗 Connection Management
- **Health Check**: Kiểm tra kết nối và collection status
- **Auto Collection Creation**: Tự động tạo collection nếu chưa tồn tại
- **Error Handling**: Xử lý robust các lỗi kết nối

#### 💾 Document Storage
```python
def add_documents(self, chunks: List[Dict], embeddings: List[List[float]]):
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        payload = {
            "content": chunk["content"],
            "source": chunk["source"], 
            "chunk_index": chunk["chunk_index"],
            "doc_id": chunk["doc_id"],
            "section": chunk["section"]
        }
        
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=payload
        )
        points.append(point)
```

#### 🔍 Search Capabilities
- **Semantic Search**: Tìm kiếm dựa trên độ tương đồng vector
- **Filtered Search**: Tìm kiếm theo source, section, doc_id
- **Hybrid Search**: Kết hợp semantic + metadata filtering

## 🧪 Benchmarking System

### 📊 Hit@K Benchmark
**Mục đích**: Đánh giá khả năng retrieval của embedding models

**Dataset**: BEIR ViHealthQA - Chuyên biệt cho domain y tế tiếng Việt

**Metrics**:
- **Hit@1**: Accuracy ở top-1 result
- **Hit@4**: Accuracy ở top-4 results  
- **Hit@10**: Accuracy ở top-10 results

### 📈 STS Correlation Benchmark
**Mục đích**: Đánh giá độ tương đồng ngữ nghĩa

**Dataset**: ViSTS (Vietnamese Semantic Textual Similarity)

**Metrics**:
- **Pearson Correlation**: Tương quan tuyến tính
- **Spearman Correlation**: Tương quan thứ hạng

### 🏆 Model Recommendations

| Model | Use Case | Performance | Speed |
|-------|----------|-------------|-------|
| `intfloat/multilingual-e5-base` | **Production** | ⭐⭐⭐⭐ | 🚀🚀🚀 |
| `keepitreal/vietnamese-sbert` | **Vietnamese Specialized** | ⭐⭐⭐⭐⭐ | 🚀🚀 |
| `Alibaba-NLP/gte-multilingual-base` | **High Accuracy** | ⭐⭐⭐⭐⭐ | 🚀🚀 |
| `intfloat/multilingual-e5-large-instruct` | **Best Overall** | ⭐⭐⭐⭐⭐ | 🚀 |

## 🚀 Cách Sử Dụng

### 📦 Cài Đặt Dependencies

```bash
cd src/data-preparing
pip install uv
uv sync
```

### 🔄 Chạy Data Ingestion

```bash
# Phân tích PDF (không lưu vào DB)
python ingest_data.py --analyze-only --data-dir ./data

# Xử lý và lưu vào vector DB
python ingest_data.py --data-dir ./data

# Xử lý với custom collection
python ingest_data.py --data-dir ./data --collection-name my_collection
```

### 🧪 Chạy Benchmarks

```bash
cd benchmarks/embedding

# Hit@K Benchmark
uv run hit_at_k_benchmark.py

# STS Correlation Benchmark  
uv run sts_correlation_benchmark.py
```

## 📊 Monitoring và Logging

### 📈 Processing Stats
```python
{
    "total_chunks": 1250,
    "total_characters": 2500000,
    "sections": {
        "Chương 1": 45,
        "Chương 2": 38,
        # ...
    },
    "sources": ["file1.pdf", "file2.pdf"],
    "doc_ids": ["uuid1", "uuid2"]
}
```

### 🔍 Health Checks
- **Qdrant Connection**: Kiểm tra kết nối vector DB
- **Embedding Model**: Validate model loading
- **Collection Status**: Kiểm tra collection và index

## 🛠️ Troubleshooting

### ❌ Lỗi Thường Gặp

1. **"index out of range in self"**
   - **Nguyên nhân**: Text quá dài hoặc có ký tự đặc biệt
   - **Giải pháp**: Text preprocessing và batch size reduction

2. **"trust_remote_code=True required"**
   - **Nguyên nhân**: Model yêu cầu trust remote code
   - **Giải pháp**: Auto-retry với trust_remote_code=True

3. **Qdrant Connection Failed**
   - **Nguyên nhân**: Qdrant server chưa khởi động
   - **Giải pháp**: `docker run -p 6333:6333 qdrant/qdrant`

### 🔧 Performance Tuning

1. **Tăng tốc Embedding**:
   - Giảm `EMBEDDING_BATCH_SIZE` nếu gặp OOM
   - Sử dụng GPU nếu có sẵn
   - Chọn model nhỏ hơn cho production

2. **Tối ưu Chunking**:
   - Điều chỉnh `CHUNK_SIZE` theo domain
   - Tăng `CHUNK_OVERLAP` để bảo toàn context
   - Sử dụng separators phù hợp với tiếng Việt

3. **Vector DB Performance**:
   - Tăng `TOP_K_DOCUMENTS` cho recall cao hơn
   - Điều chỉnh `SIMILARITY_THRESHOLD` theo use case
   - Sử dụng filtered search khi có thể

## 🔮 Roadmap

- [ ] **Multi-modal Support**: Xử lý hình ảnh trong PDF
- [ ] **Advanced Chunking**: Semantic chunking với LLM
- [ ] **Real-time Updates**: Incremental data ingestion
- [ ] **Quality Metrics**: Tự động đánh giá chất lượng chunks
- [ ] **A/B Testing**: So sánh performance các embedding models

---

## 📞 Liên Hệ Hỗ Trợ

Nếu gặp vấn đề khi sử dụng hệ thống xử lý dữ liệu, vui lòng:

1. Kiểm tra logs chi tiết
2. Xem phần Troubleshooting
3. Chạy health checks
4. Liên hệ team phát triển với thông tin lỗi cụ thể

**🎯 Hệ thống Data Preparing được thiết kế để xử lý robust, scalable và tối ưu cho domain tư vấn sức khỏe tâm lý tiếng Việt!**
