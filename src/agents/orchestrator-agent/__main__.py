# Main - FastAPI app cho orchestrator agent

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from agent import OrchestratorAgent
from config import Config
import uvicorn
from redis_memory import RedisManager, ChatHistoryStore
app = FastAPI(
    title="Orchestrator Agent",
    description="Orchestrator Agent for managing and coordinating tasks.",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    selected_agent: Optional[str] = None
    response: str
    sources: Optional[List[str]] = None
    error: Optional[str] = None


agent = OrchestratorAgent()
redis_manager = RedisManager()
chat_store: Optional[ChatHistoryStore] = None




@app.get("/health")
async def health():
    # include redis health if available
    redis = await redis_manager.health_check() if redis_manager else {"connected": False}
    return {"status": "ok", "redis": redis}


@app.on_event("startup")
async def on_startup():
    try:
        await redis_manager.initialize()
    except Exception:
        pass
    global chat_store
    chat_store = ChatHistoryStore(redis_manager)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Ensure minimal identifiers
    user_id = req.user_id or "anonymous"
    session_id = req.session_id or "default"

    # Append user message to chat history
    if chat_store and redis_manager.is_ready():
        await chat_store.append_message(user_id, session_id, role="user", content=req.message)

    result = await agent.process_message(req.message)
    # Normalize output shape
    response_text = result.get("response")

    # Append assistant message (with optional sources)
    if chat_store and redis_manager.is_ready():
        sources = result.get("sources") or []
        await chat_store.append_message(
            user_id,
            session_id,
            role="assistant",
            content=response_text or "",
            agent_used=result.get("selected_agent") or "Orchestrator",
            source=sources if isinstance(sources, list) else [],
        )

    return {
        "selected_agent": result.get("selected_agent"),
        "response": response_text,
        "sources": result.get("sources"),
        "error": result.get("error")
    }


@app.get("/history/{user_id}/{session_id}")
async def get_history(user_id: str, session_id: str):
    if not chat_store or not redis_manager.is_ready():
        return {"error": "Redis not available", "messages": [], "created_at": None, "last_updated": None}
    chat = await chat_store.load(user_id, session_id)
    return {
        "messages": chat.messages,
        "created_at": chat.created_at.isoformat(),
        "last_updated": chat.last_updated.isoformat(),
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=7010)

if __name__ == "__main__":
    main()