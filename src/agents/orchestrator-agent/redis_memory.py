"""
Redis Memory Utilities

Mục tiêu
--------
- Cung cấp lớp quản lý kết nối Redis (async) dùng chung cho Orchestrator Agent.
- Chuẩn hoá cách lưu/đọc short-term chat history lên Redis với schema rõ ràng.
- Cung cấp tiện ích health-check, key helpers và store lớp mỏng để thao tác lịch sử.

Thiết kế chính
--------------
- RedisManager: Quản lý vòng đời kết nối, set/get JSON có TTL, health check.
- ChatHistory: Mô hình hoá dữ liệu lịch sử trò chuyện (messages + metadata thời gian).
- ChatHistoryStore: Lớp thao tác lịch sử dựa trên RedisManager (load/save/append/clear).

Key & TTL
---------
- Key chat history: "chat_history:{user_id}:{session_id}"
- TTL mặc định: Config.REDIS_TTL_SECONDS (ví dụ 86400 = 1 ngày; có thể chỉnh sửa).

Quy ước message
----------------
- Mỗi message gồm: role (user|assistant), content, timestamp (ISO-8601)
  và tuỳ chọn agent_used, user_id, source (list[str] các nguồn tham chiếu).
"""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from langchain.schema import BaseMessage, HumanMessage, AIMessage

import redis.asyncio as aioredis

from config import Config


logger = logging.getLogger(__name__)


class RedisManager:
    """
    Quản lý kết nối Redis (async) và tiện ích thao tác dữ liệu mức thấp.

    Trách nhiệm:
    - Khởi tạo client theo Config (host/port/db/password) và kiểm tra kết nối (PING).
    - Cung cấp set/get JSON (có TTL), xoá key, liệt kê keys theo pattern.
    - Cung cấp health_check để giám sát trạng thái kết nối.
    """

    def __init__(self) -> None:
        self.client: Optional[aioredis.Redis] = None
        self.redis_config = {
            "host": Config.REDIS_HOST,
            "port": Config.REDIS_PORT,
            "password": Config.REDIS_PASSWORD or None,
            "db": Config.REDIS_DB,
        }

    async def initialize(self) -> None:
        if self.client is not None:
            return
        try:
            self.client = aioredis.from_url(
                f"redis://{self.redis_config['host']}:{self.redis_config['port']}/{self.redis_config['db']}",
                password=self.redis_config["password"],
                decode_responses=True,
            )
            await self.client.ping()
            logger.info("✅ RedisManager: kết nối Redis thành công")
        except Exception as e:
            logger.error(f"❌ RedisManager: không thể kết nối Redis: {e}")
            self.client = None
            raise

    async def close(self) -> None:
        if self.client is not None:
            try:
                await self.client.close()
                logger.info("✅ RedisManager: đã đóng kết nối")
            finally:
                self.client = None

    def is_ready(self) -> bool:
        return self.client is not None

    # -------- JSON helpers --------
    async def set_json(self, key: str, value: Dict[str, Any], ttl_seconds: Optional[int] = None) -> None:
        if not self.client:
            raise RuntimeError("RedisManager chưa được initialize")
        data = json.dumps(value)
        await self.client.set(key, data, ex=ttl_seconds)

    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.client:
            raise RuntimeError("RedisManager chưa được initialize")
        data = await self.client.get(key)
        if not data:
            return None
        try:
            return json.loads(data)
        except Exception:
            return None

    async def delete(self, key: str) -> None:
        if not self.client:
            raise RuntimeError("RedisManager chưa được initialize")
        await self.client.delete(key)

    async def keys(self, pattern: str) -> List[str]:
        if not self.client:
            raise RuntimeError("RedisManager chưa được initialize")
        return await self.client.keys(pattern)

    # -------- Health --------
    async def health_check(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "connected": False,
            "host": self.redis_config["host"],
            "port": self.redis_config["port"],
            "db": self.redis_config["db"],
        }
        if not self.client:
            status["error"] = "Client not initialized"
            return status
        try:
            pong = await self.client.ping()
            status["connected"] = bool(pong)
            return status
        except Exception as e:
            status["error"] = str(e)
            return status

    # -------- chat_history key helpers (tuỳ chọn) --------
    @staticmethod
    def chat_history_key(user_id: str, session_id: str) -> str:
        return f"chat_history:{user_id}:{session_id}"
    
    @staticmethod
    def chat_history_user_pattern(user_id: str) -> str:
        return f"chat_history:{user_id}:*"

    @staticmethod
    def langchain_history_key(user_id: str, session_id: str) -> str:
        return f"langchain_history:{user_id}:{session_id}"

    @staticmethod
    def langchain_history_pattern (user_id: str) -> str:
        return f"langchain_history:{user_id}:*"


class ChatHistory:
    """
    Mô hình lịch sử trò chuyện (in-memory) tương thích với payload lưu trên Redis.

    Lưu ý schema message:
    - role: "user" | "assistant"
    - content: nội dung văn bản gốc
    - timestamp: thời điểm ghi nhận (ISO-8601)
    - agent_used: (tuỳ chọn) tên agent thực thi trả lời (vd. "RAG Agent", "Orchestrator")
    - user_id: (tuỳ chọn) id người dùng (đồng bộ hoá khi cần truy vấn theo user)
    - source: list[str] các nguồn tham chiếu (nếu không có -> lưu [])
    """

    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []
        self.created_at: datetime = datetime.now()
        self.last_updated: datetime = datetime.now()

    def add_message(
        self,
        role: str,
        content: str,
        *,
        agent_used: Optional[str] = None,
        user_id: Optional[str] = None,
        source: Optional[List[str]] = None,
    ) -> None:
        message: Dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if agent_used:
            message["agent_used"] = agent_used
        if user_id:
            message["user_id"] = user_id
        message["source"] = source or []

        self.messages.append(message)
        self.last_updated = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": self.messages,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatHistory":
        inst = cls()
        inst.messages = data.get("messages", [])
        try:
            inst.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
            inst.last_updated = datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat()))
        except Exception:
            # fallback nếu parse lỗi
            inst.created_at = datetime.now()
            inst.last_updated = datetime.now()
        return inst


class ChatHistoryStore:
    """
    Lớp làm việc với Redis để lưu/đọc/ghi bổ sung chat history theo key chuẩn hoá.

    Dùng trường hợp:
    - Load lịch sử trước khi xử lý để dựng context ngắn hạn.
    - Append message người dùng trước khi gọi orchestrator/RAG.
    - Append message assistant sau khi có kết quả, kèm nguồn tham chiếu.
    - Clear theo session khi cần reset trạng thái.
    """

    def __init__(self, redis_manager: RedisManager) -> None:
        self.redis = redis_manager

    async def load(self, user_id: str, session_id: str) -> ChatHistory:
        key = RedisManager.chat_history_key(user_id, session_id)
        data = await self.redis.get_json(key) if self.redis.is_ready() else None
        if data:
            return ChatHistory.from_dict(data)
        return ChatHistory()

    async def save(self, user_id: str, session_id: str, chat: ChatHistory) -> None:
        if not self.redis.is_ready():
            return
        key = RedisManager.chat_history_key(user_id, session_id)
        await self.redis.set_json(key, chat.to_dict(), ttl_seconds=Config.REDIS_TTL_SECONDS)

    async def append_message(
        self,
        user_id: str,
        session_id: str,
        *,
        role: str,
        content: str,
        agent_used: Optional[str] = None,
        source: Optional[List[str]] = None,
    ) -> ChatHistory:
        chat = await self.load(user_id, session_id)
        chat.add_message(role, content, agent_used=agent_used, user_id=user_id, source=source)
        await self.save(user_id, session_id, chat)
        return chat

    async def clear(self, user_id: str, session_id: str) -> None:
        if not self.redis.is_ready():
            return
        key = RedisManager.chat_history_key(user_id, session_id)
        await self.redis.delete(key)

    async def list_sessions(self, user_id: str) -> List[str]:
        if not self.redis.is_ready():
            return []
        keys = await self.redis.keys(RedisManager.chat_history_user_pattern(user_id))
        sessions: List[str] = []
        for key in keys:
            parts = key.split(":")
            if len(parts) >= 3:
                sessions.append(":".join(parts[2:]))
        return sessions
    

class LangChainHistory:
    def __init__(self) -> None:
        self.turns: List[Dict[str, str]] = []

    def add_turn(self, turn_type: str, content: str) -> None:
        kind = (turn_type or "").lower()
        if kind not in ("human", "ai"):
            kind = "human" if kind in ("user", "human") else "ai"
        self.turns.append({"type": kind, "content": content})

    def to_list(self) -> List[Dict[str, str]]:
        return list(self.turns)

    @classmethod
    def from_list(cls, payload: Optional[List[Dict[str, str]]]) -> "LangChainHistory":
        inst = cls()
        if payload:
            for item in payload:
                if isinstance(item, dict) and "type" in item and "content" in item:
                    inst.add_turn(str(item["type"]), str(item["content"]))
        return inst


class LangChainHistoryStore:
    """Lớp thao tác Redis với key `langchain_history:{user_id}:{session_id}`."""

    def __init__(self, redis_manager: RedisManager) -> None:
        self.redis = redis_manager

    async def load(self, user_id: str, session_id: str) -> LangChainHistory:
        if not self.redis.is_ready():
            return LangChainHistory()
        key = RedisManager.langchain_history_key(user_id, session_id)
        raw = await self.redis.client.get(key)  # type: ignore[attr-defined]
        try:
            turns = json.loads(raw) if raw else []
        except Exception:
            turns = []
        return LangChainHistory.from_list(turns)

    async def save(
        self,
        user_id: str,
        session_id: str,
        history: LangChainHistory,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if not self.redis.is_ready():
            return
        key = RedisManager.langchain_history_key(user_id, session_id)
        ttl = ttl_seconds or Config.REDIS_TTL_SECONDS
        payload = json.dumps(history.to_list(), ensure_ascii=False)
        await self.redis.client.set(key, payload, ex=ttl)  # type: ignore[attr-defined]

    async def append_turn(
        self,
        user_id: str,
        session_id: str,
        *,
        turn_type: str,
        content: str,
        trim_to: Optional[int] = None,
    ) -> LangChainHistory:
        history = await self.load(user_id, session_id)
        history.add_turn(turn_type, content)
        if trim_to and trim_to > 0:
            history.turns = history.turns[-trim_to:]
        await self.save(user_id, session_id, history)
        return history

    async def get(self, user_id: str, session_id: str) -> List[Dict[str, str]]:
        history = await self.load(user_id, session_id)
        return history.to_list()

    async def clear(self, user_id: str, session_id: str) -> None:
        if not self.redis.is_ready():
            return
        key = RedisManager.langchain_history_key(user_id, session_id)
        await self.redis.delete(key)

    async def list_sessions(self, user_id: str) -> List[str]:
        if not self.redis.is_ready():
            return []
        keys = await self.redis.keys(RedisManager.langchain_history_user_pattern(user_id))
        sessions: List[str] = []
        for key in keys:
            parts = key.split(":")
            if len(parts) >= 3:
                sessions.append(":".join(parts[2:]))
        return sessions

    async def get_history_context(
        self,
        user_id: str,
        session_id: str,
        limit: int = 20,
    ) -> List[Dict[str, str]]:
        """Lấy N lượt hội thoại gần nhất để dựng context (mặc định 20)."""
        history_list = await self.get(user_id, session_id)
        if not history_list:
            return []
        if limit is None or limit <= 0:
            return history_list
        return history_list[-limit:] if len(history_list) > limit else history_list
    
    async def convert_history(self, user_id: str, session_id: str) -> List[BaseMessage]:
        history = await self.get(user_id, session_id)
        converted = []
        for item in history:
            role = (item.get("type") or "").lower()
            content = item.get("content") or ""
            if role in ("human", "user"):
                converted.append(HumanMessage(content=content))
            elif role in ("ai", "assistant"):
                converted.append(AIMessage(content=content))
        return converted

