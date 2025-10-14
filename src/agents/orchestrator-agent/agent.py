from a2a.types import A2A
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain.prompts import PromptTemplate
from config import Config
from typing import Dict, Any, Optional
import logging
import re
import json
from root_prompt import ROOT_INSTRUCTION
from a2a_client import RAGAgentA2AClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrchestratorAgent:
    def __init__(self):
        self.llm = None
        # Initialize A2A client with configured RAG agent URL
        self.a2a_client = RAGAgentA2AClient(base_url=Config.RAG_AGENT_URL)
        self.prompt_template = None
        self._initialized = False
        
    
    def _init_llm(self):
        if not self.llm:
            print("Đang khởi tạo LLM Model...")
            if Config.GOOGLE_API_KEY == "your_google_api_key":
                raise ValueError("Vui lòng thiết lập biến GOOGLE_API_KEY trong file .env")
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=Config.GOOGLE_LLM_MODEL,
                    temperature=Config.GOOGLE_LLM_TEMPERATURE,
                    max_output_tokens=Config.GOOGLE_LLM_MAX_OUTPUT_TOKENS,
                    google_api_key=Config.GOOGLE_API_KEY
                )
                logger.info(f"Khởi tạo LLM Model {Config.GOOGLE_LLM_MODEL} thành công")
            except ChatGoogleGenerativeAIError as e:
                logger.error(f"Lỗi khi khởi tạo LLM Model {Config.GOOGLE_LLM_MODEL}: {e}")
                exit(1)

    async def _initialize(self):
        if self._initialized:
            return
        try:
            if not self.a2a_client._initialized:
                await self.a2a_client._initialize()
            self._init_llm()
            self._setup_prompt()
            self._initialized = True
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo Orchestrator Agent: {e}")
            exit(1)
        

    def _setup_prompt(self):
        input_vars = ["user_message", "available_agents"]
        self.prompt_template = PromptTemplate(
            input_variables=input_vars,
            template=f"""{ROOT_INSTRUCTION}
            **Message từ người dùng**: 
            {{user_message}}
            **Nhiệm vụ của bạn**:
            1. Phân tích, làm rõ message từ người dùng để xác định context chính xác.
            2. Xác định agent nào phù hợp nhất để xử lý message này dựa trên context đã xác định.
            3. Trả về phản hồi ở định dạng JSON như sau:
            {{{{
                "selected_agent": "Tên Agent, có thể là RAG Agent nếu sử dụng RAG Agent hoặc null nếu trả lời trực tiếp",
                "response": "Câu trả lời cuối cùng cho người dùng",
                "sources": "Các tài liệu liên quan từ RAG Agent"
            }}}}
            **Lưu ý:**
            - Nếu là các câu hỏi đơn giản hoặc chitchat, hãy trả lời trực tiếp mà không cần sử dụng agent nào khác.
            - Nếu câu hỏi yêu cầu thông tin chuyên sâu, hãy chọn RAG Agent.
            - Chỉ chọn Agent khi thực sự cần thiết.
            - Nếu trả lời sử dụng Agent, hãy để trường "direct_response" = null, trường "response" sẽ là câu trả lời của RAG Agent.
            - Nếu có thể trả lời trực tiếp, hãy để trường "selected_agent" = null, sources = null và cung cấp câu trả lời trong trường "direct_response".
            """
        )

    async def process_message(self, message: str) -> Dict[str, Any]:
        # Ensure components are initialized
        if not self._initialized:
            await self._initialize()
        formatted_query = self.prompt_template.format(
            user_message=message,
            available_agents="RAG Agent",
        )
        try:
            result = await self.llm.ainvoke(formatted_query)
            content = getattr(result, "content", str(result))
            try:
                match = re.search(r"\{[\s\S]*\}", content)
                if match:
                    decision = json.loads(match.group(0))
                    # Nếu chọn RAG Agent, gọi A2A server để lấy câu trả lời
                    if isinstance(decision, dict) and decision.get("selected_agent") == "RAG Agent":
                        # ensure A2A client initialized
                        if not self.a2a_client._initialized:
                            await self.a2a_client._initialize()
                        rag_result = await self.a2a_client.send_message(message, stream=False)
                        return {
                            "selected_agent": "RAG Agent",
                            "response": rag_result.get("content", ""),
                            "sources": rag_result.get("sources", []),
                        }
                    elif isinstance(decision, dict) and decision.get("selected_agent") is None:
                        return {
                            "selected_agent": None,
                            "response": decision.get("response", ""),
                            "sources": None,
                        }
                    return decision
            except Exception as e:
                return {"selected_agent": None, "direct_response": "Xin lỗi, hiện tôi không thể xử lý yêu cầu.", "error": str(e)}
        except Exception as e:
            return {"selected_agent": None, "direct_response": "Xin lỗi, hiện tôi không thể xử lý yêu cầu.", "error": str(e)}


# if __name__ == "__main__":
#     import asyncio

#     async def _dev_test():
#         agent = OrchestratorAgent()
#         result = await agent.process_message("Xin chào")
#         print(f"Agent response: {result}")

#     asyncio.run(_dev_test())
