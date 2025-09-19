# RAG Agent - Self-Contained Mental Health Chatbot

## Tổng quan

RAG Agent là một chatbot tự hoàn chỉnh sử dụng Retrieval-Augmented Generation (RAG) để hỗ trợ sức khỏe tâm thần cho học sinh, sinh viên Việt Nam. Agent này tích hợp trực tiếp với Qdrant vector database và sử dụng embedding model tiếng Việt để truy xuất thông tin từ các tài liệu chuyên môn.

## Tính năng chính

### 🤖 RAG-powered Chatbot
- Tích hợp LangChain với model gemma-3n-e2b-it
- **BAAI/bge-m3** embedding model cho multi-lingual support
- Vector search từ Qdrant database
- Context-aware responses dựa trên tài liệu tham khảo
- Conversation memory cho session management

### 📚 Self-Contained Knowledge Base
- **Local embedding generation** với dangvantuan/vietnamese-embedding model
- **Direct Qdrant integration** - không cần external services
- Truy xuất thông tin từ tài liệu MOET & UNICEF
- Score-based relevance filtering với advanced query filters
- Multi-source context aggregation
- **Complete independence** - no external API dependencies cho retrieval

### 🔒 Safety Features
- Emergency response handling
- Professional support recommendations
- Fallback responses for technical issues
- Vietnamese cultural context awareness

## Cài đặt

### 1. Cài đặt dependencies
```bash
cd src/agents/rag-agent
pip install -r requirements.txt
```

### 2. Cấu hình Environment Variables
Tạo file `.env` với nội dung:
```env
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=mental_health_vi

# Service URLs
CONTEXT_RETRIEVAL_URL=http://localhost:5005
```

### 3. Chạy ứng dụng
```bash
python main.py
```

Hoặc với Docker:
```bash
docker build -t rag-agent .
docker run -p 7005:7005 --env-file .env rag-agent
```

## API Endpoints

### Chat Endpoints
- `POST /chat` - Gửi tin nhắn tới RAG chatbot
  ```json
  {
    "message": "Tôi đang cảm thấy lo lắng về kỳ thi",
    "session_id": "user_123"
  }
  ```

### Search & Retrieval Endpoints
- `POST /search` - Advanced search với simplified filters
  ```json
  {
    "query": "stress management techniques",
    "limit": 5,
    "score_threshold": 0.7,
    "filters": {
      "source_name": "MOET_SoTay_ThucHanh_CTXH",
      "page": "15",
      "section_title": "Quản lý căng thẳng"
    }
  }
  ```
- `GET /search?q=query&source_name=...&page=...` - Simple search via GET
- `POST /retrieve` - Simplified context retrieval
- `POST /embed` - Generate embedding cho text

### Document Structure
Mỗi document chunk bao gồm:
- `source_name`: Tên nguồn tài liệu (VD: "MOET_SoTay_ThucHanh_CTXH")
- `page`: Trang tài liệu (VD: "15" hoặc "15-17" cho nhiều trang)
- `section_title`: Tiêu đề section (VD: "Quản lý căng thẳng học tập")
- `content`: Nội dung văn bản thực tế
- `score`: Điểm tương đồng với query

**Lưu ý**: RAG Agent tương thích với dữ liệu hiện tại và sẽ tự động map từ các fields cũ (`title`, `context`, `doc_id`) sang structure mới. Để tối ưu hóa filters, nên cập nhật indexing pipeline để include các fields `source_name`, `page`, `section_title` trực tiếp.

### Collection Management
- `GET /collections` - List all Qdrant collections
- `GET /collection/{name}/info` - Collection details

### Session Management  
- `GET /sessions` - Liệt kê các sessions đang hoạt động
- `DELETE /sessions/{session_id}` - Xóa session cụ thể

### Health & Monitoring
- `GET /` - Health check cơ bản
- `GET /health` - Health check chi tiết với Qdrant status
- `GET /docs` - FastAPI documentation

## Kiến trúc

### RAG Pipeline
1. **User Input** → Session Management
2. **Local Embedding** → Generate query embedding với BAAI/bge-m3
3. **Vector Search** → Direct Qdrant similarity search
4. **Context Augmentation** → Combine user query với relevant documents
5. **Generation** → gemma-3n-e2b-it với enriched context
6. **Response** → Formatted output với source references

### Vector Search Flow
```
User Query → BAAI/bge-m3 Embedding → Qdrant Search → Context Chunks → LLM → Response
```

### Integration Architecture
```
RAG Agent ←→ Direct Qdrant Connection
    ↓              ↑
LangChain LLM   BAAI/bge-m3 Embeddings
    ↓              
Fallback: Context Retrieval Service (if needed)
```

## Configuration

### Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key
- `QDRANT_URL`: Qdrant server URL (default: http://localhost:6333) **[Required]**
- `QDRANT_COLLECTION`: Collection name (default: mental_health_vi) **[Required]**
- `EMBEDDING_MODEL_NAME`: HuggingFace model name (default: dangvantuan/vietnamese-embedding)
- `DEVICE`: Compute device (default: cpu, options: cpu/cuda)
- `MAX_RESULTS`: Maximum search results (default: 10)

### RAG Parameters
- `limit`: Number of context chunks to retrieve (default: 5)
- `score_threshold`: Minimum similarity score (default: 0.7)
- `temperature`: LLM temperature (default: 0.3)
- `max_tokens`: Maximum response tokens (default: 1000)

## Development

### Testing
```bash
# Test basic functionality
curl http://localhost:7005/health

# Test chat endpoint
curl -X POST http://localhost:7005/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tôi cần giúp đỡ về căng thẳng học tập", "session_id": "test"}'

# Test advanced search with filters
curl -X POST http://localhost:7005/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "quản lý căng thẳng",
    "limit": 3,
    "filters": {
      "source_name": "MOET_SoTay_ThucHanh_CTXH"
    }
  }'

# Test simple GET search
curl "http://localhost:7005/search?q=stress&source_name=MOET&limit=2"
```

### Debugging
- Set `log_level="debug"` trong uvicorn.run()
- Monitor Qdrant queries trong logs
- Check context retrieval service connectivity

## Troubleshooting

### Common Issues
1. **Qdrant Connection Error**: Kiểm tra QDRANT_URL và đảm bảo Qdrant server đang chạy
2. **Context Retrieval Service Error**: Verify CONTEXT_RETRIEVAL_URL
3. **Gemini API Error**: Kiểm tra GEMINI_API_KEY
4. **No Context Retrieved**: Kiểm tra collection tồn tại trong Qdrant
5. **Memory Issues**: Điều chỉnh conversation buffer size

### Performance Tuning
- Adjust `score_threshold` cho relevance filtering
- Tune `limit` để cân bằng context vs response time
- Configure conversation memory size
- Optimize embedding model selection

## Security Considerations

- API key security và rotation
- Input validation và sanitization
- Rate limiting cho production deployment
- Session management và cleanup
- Logging sensitive data protection

## Contributing

1. Fork repository
2. Create feature branch
3. Implement changes với comprehensive tests
4. Ensure proper error handling
5. Submit pull request với detailed description

## License

MIT License - see LICENSE file for details
