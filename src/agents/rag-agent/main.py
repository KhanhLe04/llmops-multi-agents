import uvicorn
import logging
import os
import time
from contextlib import asynccontextmanager

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, AIMessage
# Simple session management instead of LangChain memory
from collections import defaultdict, deque
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mental_health_vi")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
DEVICE = os.getenv("DEVICE", "cpu")  # or "cuda" if GPU available
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "10"))

# Global variables
qdrant_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown events"""
    global qdrant_client
    
    # Startup
    logger.info("Starting Self-Contained RAG Agent...")
    
    # Initialize Qdrant client
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL)
        logger.info(f"Connected to Qdrant at {QDRANT_URL}")
        
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        if QDRANT_COLLECTION not in collection_names:
            logger.warning(f"Collection '{QDRANT_COLLECTION}' not found. Available collections: {collection_names}")
        else:
            logger.info(f"Collection '{QDRANT_COLLECTION}' is available")
            # Get collection info
            try:
                collection_info = qdrant_client.get_collection(QDRANT_COLLECTION)
                # Use collection_info.dict() to access all properties safely
                info_dict = collection_info.dict() if hasattr(collection_info, 'dict') else vars(collection_info)
                points_count = info_dict.get('points_count', 'unknown')
                vectors_count = info_dict.get('vectors_count', 'unknown')
                logger.info(f"Collection info: {points_count} points, {vectors_count} vectors")
            except Exception as e:
                logger.warning(f"Could not get detailed collection info: {e}")
            
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise RuntimeError(f"Cannot start RAG Agent without Qdrant connection: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Agent...")
    if qdrant_client:
        qdrant_client.close()

app = FastAPI(
    title="RAG Agent - Mental Health Chatbot",
    description="",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: Optional[List[str]] = None

class SearchQuery(BaseModel):
    query: str
    limit: Optional[int] = 5
    score_threshold: Optional[float] = 0.7
    filters: Optional[Dict[str, Any]] = None

class DocumentChunk(BaseModel):
    id: str
    chunk_id: str
    source_name: str
    page: Optional[str] = None  # Can be "1-3" for multiple pages
    section_title: str
    content: str  # The actual text content
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[DocumentChunk]
    total_found: int
    search_time_ms: float

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int

class RAGAgent:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Initialize components
        self._init_llm()
        self._init_embeddings()
        
        # Store conversation sessions (simple implementation)
        self.sessions = defaultdict(lambda: deque(maxlen=20))  # Keep last 20 messages per session
        
        logger.info("RAG Agent initialized successfully")

    def _init_llm(self):
        """Initialize the LLM"""
        self.llm = ChatGoogleGenerativeAI(
            model="gemma-3n-e4b-it",
            google_api_key=self.gemini_api_key,
            temperature=0.7,
            max_tokens=1024
        )
        logger.info("LLM initialized")
    
    def _init_embeddings(self):
            model_kwargs = {
                'device': DEVICE,
                'trust_remote_code': True
            }
            
            # Encoding configuration
            encode_kwargs = {
                'normalize_embeddings': True,  # Normalize embeddings for better similarity
                'batch_size': 16  # Adjust based on memory constraints
            }
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                show_progress=True
            )
            
            logger.info(f"HuggingFace embeddings model initialized: {EMBEDDING_MODEL_NAME}")
            logger.info(f"Device: {DEVICE}")
            
            # Test embedding to ensure it works
            test_text = "Test embedding initialization"
            test_embedding = self.embeddings.embed_query(test_text)
            logger.info(f"Embedding dimension: {len(test_embedding)}")

    def get_or_create_session(self, session_id: str):
        """Get or create conversation session"""
        # Sessions are automatically created by defaultdict
        return self.sessions[session_id]
    
    def add_message_to_session(self, session_id: str, role: str, content: str):
        """Add message to session"""
        session = self.get_or_create_session(session_id)
        session.append({"role": role, "content": content})
    
    def get_session_history(self, session_id: str, limit: int = 6):
        """Get recent conversation history"""
        session = self.get_or_create_session(session_id)
        # Return recent messages
        recent_messages = list(session)[-limit:] if len(session) > limit else list(session)
        return recent_messages

    def build_qdrant_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """Build Qdrant filter from search filters"""
        if not filters:
            return None
        
        conditions = []
        
        # Simplified filter fields
        for field, value in filters.items():
            if field in ["source_name", "page", "section_title"]:
                if isinstance(value, list):
                    # Multiple values - use should match any
                    for v in value:
                        conditions.append(FieldCondition(key=field, match=MatchValue(value=v)))
                else:
                    conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
        
        return Filter(must=conditions) if conditions else None

    def format_search_result(self, hit, query: str) -> DocumentChunk:
        """Format Qdrant search result into simplified DocumentChunk"""
        payload = hit.payload
        
        # Extract source name from various possible fields
        source_name = (
            payload.get("source_name") or 
            payload.get("source") or 
            payload.get("doc_id", "").replace("_vi", "") or
            "Unknown Source"
        )
        
        # Extract section title from title or section fields
        section_title = (
            payload.get("section_title") or 
            payload.get("section") or 
            payload.get("title", "").split('\n')[0] or  # Take first line of title
            "Untitled Section"
        )
        
        # Extract content from various possible fields
        content = (
            payload.get("content") or 
            payload.get("text") or 
            payload.get("context") or  # For current data structure
            ""
        )
        
        return DocumentChunk(
            id=payload.get("id", ""),
            chunk_id=payload.get("chunk_id", str(payload.get("id", ""))),
            source_name=source_name,
            page=payload.get("page"),  # May be None for current data
            section_title=section_title,
            content=content,
            score=hit.score
        )

    async def search_context(self, search_query: SearchQuery) -> SearchResponse:
        """Search for relevant context based on query"""
        start_time = time.time()
        
        try:
            if not qdrant_client:
                raise HTTPException(status_code=500, detail="Qdrant client not initialized")
            
            if not self.embeddings:
                raise HTTPException(status_code=500, detail="Embeddings model not initialized")
            
            # Get embedding for the query
            query_embedding = self.embeddings.embed_query(search_query.query)
            
            # Build filter
            qdrant_filter = self.build_qdrant_filter(search_query.filters)
            
            # DEBUG: Log search parameters
            logger.info(f"🔍 Search parameters:")
            logger.info(f"   Query: '{search_query.query[:50]}...'")
            logger.info(f"   Collection: {QDRANT_COLLECTION}")
            logger.info(f"   Embedding dim: {len(query_embedding)}")
            logger.info(f"   Limit: {min(search_query.limit, MAX_RESULTS)}")
            logger.info(f"   Score threshold: {search_query.score_threshold}")
            logger.info(f"   Filter: {qdrant_filter}")
            
            # Use old reliable search API
            search_results = qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=min(search_query.limit, MAX_RESULTS),
                score_threshold=search_query.score_threshold,
                with_payload=True,
                with_vectors=False
            )
            
            logger.info(f"📊 Raw search results: {len(search_results)}")
            
            # If no results with current threshold, try with lower threshold
            if not search_results and search_query.score_threshold > 0.1:
                logger.info(f"🔄 No results with threshold {search_query.score_threshold}, trying 0.1")
                search_results = qdrant_client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=query_embedding,
                    query_filter=qdrant_filter,
                    limit=min(search_query.limit, MAX_RESULTS),
                    score_threshold=0.1,
                    with_payload=True,
                    with_vectors=False
                )
                logger.info(f"📊 Results with threshold 0.1: {len(search_results)}")
            
            # If still no results, try without any threshold
            if not search_results:
                logger.info(f"🔄 Still no results, trying without threshold...")
                search_results = qdrant_client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=query_embedding,
                    query_filter=qdrant_filter,
                    limit=min(search_query.limit, MAX_RESULTS),
                    score_threshold=0.0,  # No threshold
                    with_payload=True,
                    with_vectors=False
                )
                logger.info(f"📊 Results without threshold: {len(search_results)}")
            
            # Format results
            results = [
                self.format_search_result(hit, search_query.query) 
                for hit in search_results
            ]
            
            search_time_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Search completed: {len(results)} results in {search_time_ms:.2f}ms")
            
            return SearchResponse(
                query=search_query.query,
                results=results,
                total_found=len(results),
                search_time_ms=round(search_time_ms, 2)
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

    async def retrieve_context(self, query: str, limit: int = 5, score_threshold: float = 0.7, filters: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Retrieve relevant context chunks for RAG"""
        try:
            search_query = SearchQuery(
                query=query,
                limit=limit,
                score_threshold=score_threshold,
                filters=filters
            )
            
            search_response = await self.search_context(search_query)
            return search_response.results
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text"""
        try:
            if not self.embeddings:
                raise HTTPException(status_code=500, detail="Embeddings model not initialized")
            
            embedding = self.embeddings.embed_query(text)
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding generation error: {str(e)}")

    async def chat(self, message: str, session_id: str) -> ChatResponse:
        """Chat with RAG capabilities"""
        try:
            # Retrieve relevant context
            context_chunks = await self.retrieve_context(message)
            
            # Build context string
            context_text = ""
            sources = []
            if context_chunks:
                context_text = "\n\n".join([
                    f"[{chunk.source_name}] {chunk.content}" 
                    for chunk in context_chunks
                ])
                sources = list(set([chunk.source_name for chunk in context_chunks]))
            
            # Get chat history
            chat_history = self.get_session_history(session_id, limit=6)
            
            # Build conversation context
            conversation_context = """
            Bạn là một trợ lý AI chuyên về sức khỏe tâm thần cho học sinh, sinh viên Việt Nam. 
            Hãy trả lời dựa trên thông tin được cung cấp và kinh nghiệm của bạn. 
            Luôn ưu tiên sự an toàn và khuyến khích tìm kiếm sự hỗ trợ chuyên nghiệp khi cần thiết.\n\n
            """
            
            if context_text:
                conversation_context += f"Thông tin tham khảo:\n{context_text}\n\n"
            
            # Add chat history
            if chat_history:
                conversation_context += "Lịch sử trò chuyện:\n"
                for msg in chat_history:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    if role == 'user':
                        conversation_context += f"Người dùng: {content}\n"
                    elif role == 'assistant':
                        conversation_context += f"Trợ lý: {content}\n"
            
            # Add current message
            conversation_context += f"\nCâu hỏi hiện tại: {message}\n\nTrả lời:"
            
            # Generate response
            response = self.llm.invoke(conversation_context)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Update session
            self.add_message_to_session(session_id, 'user', message)
            self.add_message_to_session(session_id, 'assistant', response_text)
            
            return ChatResponse(
                response=response_text,
                session_id=session_id,
                sources=sources if sources else None
            )
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            # Fallback response
            fallback_response = """
            Xin lỗi, tôi đang gặp một số vấn đề kỹ thuật. 

            Tuy nhiên, tôi muốn nhắc nhở bạn rằng:
            - Mọi cảm xúc và lo lắng đều là bình thường
            - Hãy tìm kiếm sự hỗ trợ từ gia đình, bạn bè hoặc chuyên gia tâm lý
            - Đừng ngần ngại liên hệ với đường dây tư vấn tâm lý nếu cần

            Nếu bạn đang gặp khủng hoảng tâm lý, hãy liên hệ ngay với các dịch vụ hỗ trợ khẩn cấp.
            """

            return ChatResponse(
                response=fallback_response,
                session_id=session_id,
                sources=["emergency_fallback"]
            )


# Initialize the agent
rag_agent = RAGAgent()

# FastAPI Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "RAG Agent is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check for self-contained RAG Agent"""
    qdrant_status = "unknown"
    embedding_status = "unknown"
    collection_info = {}
    
    # Check Qdrant connection
    try:
        if qdrant_client:
            collections = qdrant_client.get_collections()
            qdrant_status = "connected"
            
            # Get collection info
            if QDRANT_COLLECTION in [col.name for col in collections.collections]:
                try:
                    collection_data = qdrant_client.get_collection(QDRANT_COLLECTION)
                    # Safely extract collection info
                    info_dict = collection_data.dict() if hasattr(collection_data, 'dict') else vars(collection_data)
                    
                    collection_info = {
                        "name": QDRANT_COLLECTION,
                        "points_count": info_dict.get('points_count', 'unknown'),
                        "vectors_count": info_dict.get('vectors_count', 'unknown'),
                        "status": info_dict.get('status', 'unknown')
                    }
                except Exception as e:
                    collection_info = {
                        "name": QDRANT_COLLECTION,
                        "error": f"Could not get collection details: {str(e)}"
                    }
        else:
            qdrant_status = "not_connected"
    except Exception as e:
        qdrant_status = f"error: {str(e)}"
    
    # Check embedding model status
    try:
        if hasattr(rag_agent, 'embeddings') and rag_agent.embeddings:
            # Test embedding generation
            test_embedding = rag_agent.embeddings.embed_query("test")
            embedding_status = f"loaded: {EMBEDDING_MODEL_NAME} (dim: {len(test_embedding)})"
        else:
            embedding_status = "not_loaded"
    except Exception as e:
        embedding_status = f"error: {str(e)}"
    
    overall_status = "healthy" if qdrant_status == "connected" and "loaded" in embedding_status else "unhealthy"
    
    return {
        "status": overall_status,
        "qdrant": qdrant_status,
        "embedding_model": embedding_status,
        "collection": collection_info,
        "device": DEVICE,
        "max_results": MAX_RESULTS,
        "sessions": len(rag_agent.sessions),
        "mode": "self_contained"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """Main chat endpoint with RAG"""
    try:
        response = await rag_agent.chat(
            message=chat_message.message,
            session_id=chat_message.session_id
        )
        return response
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation session"""
    if session_id in rag_agent.sessions:
        rag_agent.sessions[session_id].clear()
        return {"message": f"Session {session_id} cleared"}
    else:
        return {"message": f"Session {session_id} not found"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "sessions": list(rag_agent.sessions.keys()),
        "total": len(rag_agent.sessions)
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=7005,
        log_level="info"
    )

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(search_query: SearchQuery):
    """Search for relevant context based on query"""
    try:
        response = await rag_agent.search_context(search_query)
        return response
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search", response_model=SearchResponse)
async def search_endpoint_get(
    q: str,
    limit: int = 5,
    score_threshold: float = 0.7,
    source_name: Optional[str] = None,
    page: Optional[str] = None,
    section_title: Optional[str] = None
):
    """Search for relevant context using GET method with simplified parameters"""
    # Build filters from query parameters
    filters = {}
    if source_name:
        filters["source_name"] = source_name.split(",") if "," in source_name else source_name
    if page:
        filters["page"] = page.split(",") if "," in page else page
    if section_title:
        filters["section_title"] = section_title.split(",") if "," in section_title else section_title
    
    search_query = SearchQuery(
        query=q,
        limit=limit,
        score_threshold=score_threshold,
        filters=filters if filters else None
    )
    
    return await search_endpoint(search_query)

@app.post("/embed", response_model=EmbeddingResponse)
async def embed_endpoint(embedding_request: EmbeddingRequest):
    """Generate embedding for given text"""
    try:
        embedding = await rag_agent.generate_embedding(embedding_request.text)
        return EmbeddingResponse(
            embedding=embedding,
            dimension=len(embedding)
        )
    except Exception as e:
        logger.error(f"Embedding endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve", response_model=List[DocumentChunk])
async def retrieve_endpoint(search_query: SearchQuery):
    """Retrieve relevant context chunks (simplified interface)"""
    try:
        chunks = await rag_agent.retrieve_context(
            query=search_query.query,
            limit=search_query.limit,
            score_threshold=search_query.score_threshold,
            filters=search_query.filters
        )
        return chunks
    except Exception as e:
        logger.error(f"Retrieve endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def get_collections():
    """Get available collections in Qdrant"""
    try:
        if not qdrant_client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        
        collections = qdrant_client.get_collections()
        collection_list = []
        
        for col in collections.collections:
            try:
                # Safely extract collection properties
                col_dict = col.dict() if hasattr(col, 'dict') else vars(col)
                collection_list.append({
                    "name": col_dict.get('name', col.name if hasattr(col, 'name') else 'unknown'),
                    "status": col_dict.get('status', 'unknown'),
                    "vectors_count": col_dict.get('vectors_count', 'unknown'),
                    "points_count": col_dict.get('points_count', 'unknown')
                })
            except Exception as e:
                collection_list.append({
                    "name": getattr(col, 'name', 'unknown'),
                    "error": f"Could not get collection info: {str(e)}"
                })
        
        return {"collections": collection_list}
    except Exception as e:
        logger.error(f"Failed to get collections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collections: {str(e)}")

@app.get("/collection/{collection_name}/info")
async def get_collection_info(collection_name: str):
    """Get detailed information about a specific collection"""
    try:
        if not qdrant_client:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        
        collection_info = qdrant_client.get_collection(collection_name)
        
        # Safely extract collection info
        info_dict = collection_info.dict() if hasattr(collection_info, 'dict') else vars(collection_info)
        
        # Get actual count
        try:
            count_result = qdrant_client.count(collection_name=collection_name)
            actual_count = count_result.count if hasattr(count_result, 'count') else 0
        except:
            actual_count = 'unknown'
        
        return {
            "name": collection_name,
            "status": info_dict.get('status', 'unknown'),
            "vectors_count": info_dict.get('vectors_count', 'unknown'),
            "points_count": info_dict.get('points_count', 'unknown'),
            "actual_count": actual_count,
            "config": info_dict.get('config', None)
        }
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")

