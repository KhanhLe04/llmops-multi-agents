import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

# Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Orchestrator Agent - Mental Health Chatbot",
    description="An intelligent orchestrator agent with RAG capabilities for Vietnamese mental health support",
    version="2.0.0",
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: Optional[List[str]] = None

class OrchestratorAgent:
    def __init__(self):
        """Initialize the Orchestrator Agent with RAG capabilities"""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = os.getenv("QDRANT_COLLECTION", "mental_health_docs")
        
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Initialize components
        self._init_llm()
        self._init_qdrant_client()
        self._init_embeddings()
        self._init_vector_store()
        self._init_rag_chain()
        
        # Store conversation sessions
        self.sessions = {}
        
        logger.info("Orchestrator Agent initialized successfully")
    
    def _init_llm(self):
        """Initialize the LLM"""
        self.llm = ChatGoogleGenerativeAI(
            model="gemma-3n-e2b-it",
            google_api_key=self.gemini_api_key,
            temperature=0.3,
            max_tokens=1000
        )
        logger.info("LLM initialized")
    
    def _init_qdrant_client(self):
        """Initialize Qdrant client"""
        try:
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port
            )
            # Test connection
            collections = self.qdrant_client.get_collections()
            logger.info(f"Connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _init_embeddings(self):
        """Initialize embeddings model"""
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.gemini_api_key
        )
        logger.info("Embeddings model initialized")
    
    def _init_vector_store(self):
        """Initialize vector store connection to Qdrant"""
        try:
            self.vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
            logger.info(f"Vector store connected to collection: {self.collection_name}")
            
            # Test retrieval to check data quality
            try:
                test_results = self.vector_store.similarity_search("test", k=1)
                logger.info("Vector store test retrieval successful")
            except Exception as e:
                logger.warning(f"Vector store test failed: {e}")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            # Don't raise here, allow fallback mode
            self.vector_store = None
    
    def _init_rag_chain(self):
        """Initialize RAG chain with custom prompt for Vietnamese mental health"""
        # Check if vector store is available
        if self.vector_store is None:
            logger.warning("Vector store not available, initializing fallback mode")
            self.rag_chain = None
            return
            
        # Create custom prompt template
        system_template = """
        Bạn là một chuyên gia tư vấn tâm lý được đào tạo chuyên sâu về sức khỏe tâm thần của học sinh, sinh viên Việt Nam.

        Nhiệm vụ của bạn:
        1. Cung cấp lời khuyên chuyên nghiệp, empathetic và phù hợp với văn hóa Việt Nam
        2. Sử dụng thông tin từ các tài liệu chuyên môn được cung cấp trong context
        3. Luôn khuyến khích tìm kiếm sự giúp đỡ chuyên nghiệp khi cần thiết
        4. Trả lời bằng tiếng Việt một cách tự nhiên và dễ hiểu

        Context từ tài liệu chuyên môn:
        {context}

        Lịch sử cuộc trò chuyện:
        {chat_history}

        Hãy trả lời câu hỏi một cách chu đáo và hữu ích.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        try:
            # Create custom retriever with validation
            retriever = self._create_safe_retriever()
            
            # Create retrieval chain
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": prompt,
                    "memory": ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True
                    )
                }
            )
            logger.info("RAG chain initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG chain: {e}")
            self.rag_chain = None
    
    def _create_safe_retriever(self):
        """Create a retriever with document validation"""
        from langchain.schema import Document
        
        class SafeRetriever:
            def __init__(self, vector_store):
                self.vector_store = vector_store
            
            def get_relevant_documents(self, query: str):
                """Get documents with validation"""
                try:
                    docs = self.vector_store.similarity_search(query, k=5)
                    
                    # Filter out invalid documents
                    valid_docs = []
                    for doc in docs:
                        if hasattr(doc, 'page_content') and doc.page_content is not None:
                            # Ensure page_content is string
                            if isinstance(doc.page_content, str) and doc.page_content.strip():
                                valid_docs.append(doc)
                            else:
                                logger.warning(f"Skipping document with invalid content: {type(doc.page_content)}")
                        else:
                            # Create a fallback document
                            logger.warning("Found document with None page_content, creating fallback")
                            fallback_doc = Document(
                                page_content="Tài liệu hỗ trợ về sức khỏe tâm thần học sinh, sinh viên.",
                                metadata=getattr(doc, 'metadata', {})
                            )
                            valid_docs.append(fallback_doc)
                    
                    return valid_docs
                    
                except Exception as e:
                    logger.error(f"Error in document retrieval: {e}")
                    # Return fallback documents
                    return [Document(
                        page_content="Thông tin về sức khỏe tâm thần và hỗ trợ tâm lý cho học sinh, sinh viên.",
                        metadata={"source": "fallback"}
                    )]
        
        return SafeRetriever(self.vector_store)
    
    def get_or_create_session(self, session_id: str):
        """Get or create conversation session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        return self.sessions[session_id]
    
    async def chat(self, message: str, session_id: str = "default") -> ChatResponse:
        """Process chat message with RAG or fallback"""
        try:
            # Get session memory
            memory = self.get_or_create_session(session_id)
            
            # Check if RAG is available
            if self.rag_chain is None:
                logger.warning("RAG chain not available, using fallback mode")
                return await self._fallback_chat(message, session_id, memory)
            
            # Get chat history
            chat_history = memory.chat_memory.messages
            
            # Process with RAG
            try:
                result = self.rag_chain(
                    {
                        "query": message,
                        "chat_history": chat_history
                    }
                )
                
                response_text = result["result"]
                source_docs = result.get("source_documents", [])
                
                # Extract sources
                sources = []
                for doc in source_docs:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        sources.append(doc.metadata['source'])
                
                # Update memory
                memory.chat_memory.add_user_message(message)
                memory.chat_memory.add_ai_message(response_text)
                
                return ChatResponse(
                    response=response_text,
                    session_id=session_id,
                    sources=sources[:3]  # Limit to top 3 sources
                )
                
            except Exception as rag_error:
                logger.error(f"RAG processing failed: {rag_error}")
                logger.info("Falling back to direct LLM response")
                return await self._fallback_chat(message, session_id, memory)
            
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")
    
    async def _fallback_chat(self, message: str, session_id: str, memory) -> ChatResponse:
        """Fallback chat without RAG"""
        try:
            # Create a simple prompt for direct LLM usage
            system_message = """Bạn là một chuyên gia tư vấn tâm lý chuyên về sức khỏe tâm thần của học sinh, sinh viên Việt Nam.

Hãy cung cấp lời khuyên hữu ích, empathetic và phù hợp với văn hóa Việt Nam. 
Luôn khuyến khích tìm kiếm sự giúp đỡ chuyên nghiệp khi cần thiết.
Trả lời bằng tiếng Việt một cách tự nhiên và dễ hiểu."""

            # Get chat history
            chat_history = memory.chat_memory.messages
            
            # Build conversation context
            conversation = [{"role": "system", "content": system_message}]
            
            # Add chat history
            for msg in chat_history[-10:]:  # Limit to recent messages
                if hasattr(msg, 'content'):
                    if msg.__class__.__name__ == 'HumanMessage':
                        conversation.append({"role": "user", "content": msg.content})
                    elif msg.__class__.__name__ == 'AIMessage':
                        conversation.append({"role": "assistant", "content": msg.content})
            
            # Add current message
            conversation.append({"role": "user", "content": message})
            
            # Use LLM directly
            response = self.llm.invoke(message)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Update memory
            memory.chat_memory.add_user_message(message)
            memory.chat_memory.add_ai_message(response_text)
            
            return ChatResponse(
                response=response_text + "\n\n⚠️ Lưu ý: Đang hoạt động ở chế độ cơ bản (không có tài liệu tham khảo).",
                session_id=session_id,
                sources=["fallback_mode"]
            )
            
        except Exception as e:
            logger.error(f"Fallback chat failed: {e}")
            # Last resort fallback
            default_response = """
            Xin lỗi, tôi đang gặp một số vấn đề kỹ thuật. 

            Tuy nhiên, tôi muốn nhắc nhở bạn rằng:
            - Mọi cảm xúc và lo lắng đều là bình thường
            - Hãy tìm kiếm sự hỗ trợ từ gia đình, bạn bè hoặc chuyên gia tâm lý
            - Đừng ngần ngại liên hệ với đường dây tư vấn tâm lý nếu cần

            Nếu bạn đang gặp khủng hoảng tâm lý, hãy liên hệ ngay với các dịch vụ hỗ trợ khẩn cấp.
            """

            return ChatResponse(
                response=default_response,
                session_id=session_id,
                sources=["emergency_fallback"]
            )

# Initialize the agent
orchestrator = OrchestratorAgent()

# FastAPI Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Orchestrator Agent - Mental Health Chatbot is running",
        "version": "2.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test Qdrant connection
        collections = orchestrator.qdrant_client.get_collections()
        qdrant_status = "connected"
    except Exception as e:
        qdrant_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "qdrant": qdrant_status,
        "collection": orchestrator.collection_name,
        "sessions": len(orchestrator.sessions)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """Main chat endpoint with RAG"""
    try:
        response = await orchestrator.chat(
            message=chat_message.message,
            session_id=chat_message.session_id
        )
        return response
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/orchestrate")
# async def orchestrate_request(request: dict):
#     """Orchestration endpoint to coordinate with other agents"""
#     try:
#         task_type = request.get("task_type", "general")
#         message = request.get("message", "")
#         session_id = request.get("session_id", "default")
        
#         # Determine which agent to route to based on task type
#         if task_type == "emergency":
#             # Route to emergency agent (port 7010)
#             return {"agent": "emergency", "status": "routed", "message": "Routing to emergency agent"}
#         elif task_type == "empathy":
#             # Route to empathy agent (port 7015)  
#             return {"agent": "empathy", "status": "routed", "message": "Routing to empathy agent"}
#         else:
#             # Handle with RAG agent or local RAG
#             response = await orchestrator.chat(message, session_id)
#             return {
#                 "agent": "rag",
#                 "status": "processed",
#                 "response": response.response,
#                 "sources": response.sources
#             }
            
#     except Exception as e:
#         logger.error(f"Orchestration error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation session"""
    if session_id in orchestrator.sessions:
        del orchestrator.sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    else:
        return {"message": f"Session {session_id} not found"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "sessions": list(orchestrator.sessions.keys()),
        "total": len(orchestrator.sessions)
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=7000,
        log_level="info"
    )