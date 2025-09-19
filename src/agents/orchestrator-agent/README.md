# Orchestrator Agent - Mental Health Chatbot

## Tổng quan

Orchestrator Agent là thành phần trung tâm của hệ thống LLMOps Multi-Agent, được tối ưu hóa với LangChain và RAG (Retrieval-Augmented Generation) để hỗ trợ sức khỏe tâm thần cho học sinh, sinh viên Việt Nam.

## Tính năng chính

### 🤖 RAG-powered Chatbot
- Tích hợp LangChain với model gemma-3n-e2b-it
- Conversation memory cho session management
- Hỗ trợ tiếng Việt chuyên biệt cho tâm lý học

### 🎯 Agent Orchestration
- Điều phối requests tới các agents chuyên biệt
- Routing thông minh dựa trên task type
- Load balancing và error handling

### 📊 Monitoring & Management
- Health check endpoints
- Session management
- Logging và metrics
- FastAPI documentation

## Cài đặt

### 1. Cài đặt dependencies
```bash
cd src/agents/orchestrator-agent
pip install -r requirements.txt
```

### 2. Cấu hình Environment Variables
Tạo file `.env` với nội dung:
```env
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Service URLs
RAG_AGENT_URL=http://localhost:7005
CONTEXT_RETRIEVAL_URL=http://localhost:5005
EMBEDDING_SERVICE_URL=http://localhost:5000
```

### 3. Chạy ứng dụng
```bash
python main.py
```

Hoặc với Docker:
```bash
docker build -t orchestrator-agent .
docker run -p 7000:7000 --env-file .env orchestrator-agent
```

## API Endpoints

### Chat Endpoints
- `POST /chat` - Gửi tin nhắn tới chatbot

### Session Management  
- `GET /sessions` - Liệt kê các sessions đang hoạt động
- `DELETE /sessions/{session_id}` - Xóa session cụ thể

### Health & Monitoring
- `GET /` - Health check cơ bản
- `GET /health` - Health check chi tiết
- `GET /docs` - FastAPI documentation

## Kiến trúc

### RAG Pipeline
1. **User Input** → Embedding với Google Generative AI
2. **Vector Search** → Qdrant similarity search  
3. **Context Retrieval** → Top-k relevant documents
4. **Generation** → gemma-3n-e2b-it với context
5. **Response** → Formatted output với sources

### Agent Orchestration
```
User Request → Orchestrator → Task Classification → Agent Routing
                          ↓
                      Response Aggregation ← Specialized Agent
```

## Tối ưu hóa

### Performance
- Async/await cho non-blocking operations
- Connection pooling cho Qdrant
- Memory-efficient conversation management
- Caching cho frequent queries

## Development

### Debugging
- Set `log_level="debug"` trong uvicorn.run()
- Enable LangChain debug mode
- Monitor Qdrant queries

## Triển khai

### Docker Compose
```yaml
version: '3.8'
services:
  orchestrator:
    build: .
    ports:
      - "7000:7000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    depends_on:
      - qdrant
```

### Kubernetes
- Helm charts available trong `/k8s`
- Auto-scaling configurations
- Service mesh integration

## Troubleshooting

### Common Issues
1. **Qdrant Connection Error**: Kiểm tra QDRANT_HOST và QDRANT_PORT
2. **Gemini API Error**: Verify GEMINI_API_KEY
3. **Memory Issues**: Điều chỉnh conversation buffer size
4. **Slow Responses**: Optimize retriever search_kwargs

### Performance Tuning
- Adjust `k` value trong retriever (default: 5)
- Tune `temperature` cho creativity/consistency balance  
- Optimize chunk size và overlap trong indexing
- Configure connection pooling

## Contributing

1. Fork repository
2. Create feature branch
3. Implement changes với tests
4. Submit pull request
5. Code review process

## License

MIT License - see LICENSE file for details
