# Main - FastAPI app cho orchestrator agent

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from agent import OrchestratorAgent
from config import Config
import uvicorn
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




@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    result = await agent.process_message(req.message)
    # Normalize output shape
    response_text = result.get("response")
    return {
        "selected_agent": result.get("selected_agent"),
        "response": response_text,
        "sources": result.get("sources"),
        "error": result.get("error")
    }

def main():
    uvicorn.run(app, host=Config.ORCHESTRATOR_AGENT_HOST, port=Config.ORCHESTRATOR_AGENT_PORT)

if __name__ == "__main__":
    main()