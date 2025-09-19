import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from system_prompt import prompt as system_message

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Orchestrator Agent - Mental Health Chatbot",
    description="",
    version="1.0.0",
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
        
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Initialize components
        self._init_llm()
        self._init_embeddings()
        
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
    
    def _init_embeddings(self):
        """Initialize embeddings model"""
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.gemini_api_key
        )
        logger.info("Embeddings model initialized")
      
    def get_or_create_session(self, session_id: str):
        """Get or create conversation session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        return self.sessions[session_id]
    
    async def chat(self, message: str, session_id: str) -> ChatResponse:
        """Response without RAG, direct LLM usage"""
        try:
            # Get session memory
            memory = self.get_or_create_session(session_id)

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
                response=response_text + "⚠️ Lưu ý: Đang hoạt động ở chế độ cơ bản (không có tài liệu tham khảo).",
                session_id=session_id
            )
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
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
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
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