from a2a.client import A2ACardResolver, A2AClient
from a2a.client.base_client import BaseClient
from a2a.client.transports import RestTransport
from a2a.types import (
    SendMessageRequest, 
    SendStreamingMessageRequest, 
    MessageSendParams,
    Message
)
import httpx
import asyncio
from uuid import uuid4
from typing import Dict, Any

class RAGAgentA2AClient:
    def __init__(self, base_url: str = "http://localhost:7005"):
        self.base_url = base_url
        self.httpx_client = None
        self.client = None
        self.agent_card = None
        self._initialized = None

    async def _initialize(self):
        if self._initialized:
            return
        
        self.httpx_client = httpx.AsyncClient()

        card_resolver = A2ACardResolver(
            base_url= self.base_url,
            httpx_client= self.httpx_client,
            agent_card_path="/.well-known/agent-card.json"
        )

        self.agent_card = await card_resolver.get_agent_card()

        self.client = A2AClient(
            httpx_client=self.httpx_client,
            agent_card=self.agent_card,
            url=self.base_url,
        )

        print(f"Kết nối tới Agent thành công: {self.agent_card.name}")
        print(f"Thông tin: {self.agent_card.description}")
        print(f"URL: {self.agent_card.url}")
        if hasattr(self.agent_card, 'skills') and self.agent_card.skills:
            print(f"Các skill: {len(self.agent_card.skills)}")
            for skill in self.agent_card.skills:
                print(f"   - {skill.name}: {skill.description}")
        else:
            print("🛠️ Không có skill cụ thể nào được liệt kê")
        print("─" * 50)

        self._initialized = True
    
    async def close(self):
        if self.httpx_client:
            await self.httpx_client.aclose()
    
    async def send_message(self, message: str, stream: bool = False) -> dict:
        if not self.client:
            await self.initialize()

        send_message_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': message}
                ],
                'messageId': uuid4().hex,
            },
        }
        
        if stream:
            print(f"Stream request tới RAG Agent")
            
            # Tạo streaming request
            streaming_request = SendStreamingMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload)
            )
            
            stream_response = self.client.send_message_streaming(streaming_request)
            result_parts = []
            
            async for chunk in stream_response:
                chunk_data = chunk.model_dump(mode='json', exclude_none=True)
                print(f" Chunk: {chunk_data}")
                
                # Lấy content từ chunk
                if 'result' in chunk_data:
                    result = chunk_data['result']
                    if 'parts' in result:
                        for part in result['parts']:
                            if part.get('type') == 'text':
                                result_parts.append(part.get('text', ''))
            
            return {
                "status": "success",
                "content": "\n".join(result_parts),
                "task_id": streaming_request.id,
                "raw_response": chunk_data
            }
                    
        else:
            print(f"Gửi message thông thường tới RAG Agent...")

            # Tạo message
            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload)
            )

            # Gửi message
            response = await self.client.send_message(request=request, http_kwargs={"timeout": None})
            response_data = response.model_dump(mode='json', exclude_none=True)
            
            # Lấy content từ response data (theo artifacts/parts/text)
            content = ""
            if 'result' in response_data:
                result = response_data['result']
                artifacts = result.get('artifacts', [])
                for i, artifact in enumerate(artifacts):
                    parts = artifact.get('parts', [])
                    for j, part in enumerate(parts):
                        if part.get('kind') == 'text':
                            text_content = part.get('text', '')
                            content += text_content + "\n"
            return {
                "status": "success",
                "content": content,
                "task_id": request.id,
                "raw_response": response_data
            }

async def test_query():
    demo_question = "Stress là gì?"

    client = RAGAgentA2AClient()
    try:
        await client._initialize()
        print("A2A Client được khởi tạo thành công")
        print(f"Thử query với câu hỏi: {demo_question}")

        result = await client.send_message(demo_question, stream=False)
        if result["status"] == "success":
            print(f"Agent trả lời: {result['raw_response']}")
        else:
            print(f"Lỗi: {result['error']}")

        await asyncio.sleep(1)
    finally:
        await client.close()
        
async def main():
    await test_query()


if __name__ == "__main__":
    asyncio.run(main())