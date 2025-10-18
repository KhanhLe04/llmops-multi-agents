import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Google API Key
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your_google_api_key")
    GOOGLE_LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemma-3n-e4b-it")
    GOOGLE_LLM_TEMPERATURE = os.getenv("GOOGLE_LLM_TEMPERATURE", 0.2)
    GOOGLE_LLM_MAX_OUTPUT_TOKENS = os.getenv("GOOGLE_LLM_MAX_OUTPUT_TOKENS", 2048)


    # Server
    ORCHESTRATOR_AGENT_HOST = os.getenv("ORCHESTRATOR_AGENT_HOST", "0.0.0.0")
    ORCHESTRATOR_AGENT_PORT = int(os.getenv("ORCHESTRATOR_AGENT_PORT", 8000))
    RAG_AGENT_URL = os.getenv("RAG_AGENT_URL", "http://localhost:7005")


    #REDIS
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = os.getenv("REDIS_PORT", 6379)
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB = os.getenv("REDIS_DB", 0)

    # Short-term memory settings
    HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", 50))
    REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", 86400))  # 1 day
    MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", 2000))