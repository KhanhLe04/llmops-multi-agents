from agent import RAGAgent
from datetime import datetime
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.utils.task import new_task
from a2a.utils.errors import ServerError
from a2a.types import Part, TextPart, InternalError, UnsupportedOperationError
from typing import Any, Dict
import logging
import uuid
logger = logging.getLogger(__name__)

class RAGAgentExecutor(AgentExecutor):
    def __init__(self):
        try:
            self.agent=RAGAgent()
            print("Khởi tạo RAG Agent thành công")
        except Exception as e:
            logger.error(f"Failed to initialize RAGAgent: {e}")
            exit(1)

    async def execute(
        self, 
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        
        # Kiểm tra context hợp lệ
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            raise ValueError("RequestContext must have a message")
        
        # Tạo task mới nếu không có sẵn
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=context.context_id
        )

        try:
            query = context.get_user_input()
            start_time = datetime.now()
            result = self.agent.invoke(query)
            total_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Processed query in {total_time} seconds")
            logger.info(f"Retrieved documents: {result.get('relevant_documents_count'), 0}")
            logger.info(f"Used {result.get('relevant_documents_count'), 0} relevant documents")
            
            answer = result.get("answer", "Tôi không thể trả lời câu hỏi của bạn vào lúc này.")
            sources = result.get("sources", [])

            metadata = {
                "sources": sources,
                "relevant_documents_count": result.get("relevant_documents_count", 0),
                "total_retrieved_count": result.get("total_retrieved_count", 0),
                "processing_time": total_time,
                "status": result.get("status", "completed")
            }

            # Tạo response, thêm artifact và hoàn thành task
            parts = [Part(root=TextPart(text=answer))]
            await updater.add_artifact(
                parts=parts,
                metadata=metadata
            )

            # Cập nhật trạng thái task thành completed
            await event_queue.enqueue_event(task)
            await updater.complete()
            logger.info(f"Completed task: {context.task_id}")
        
        except Exception as e:
            logger.error(f"RAG Agent execution failed: {e}")
            await updater.update_status("failed", error=str(e))
            try:
                await updater.failed(str(e))
            except Exception as fail_error:
                logger.error(f"Error calling updater.fail: {fail_error}")
            raise ServerError(error=InternalError()) from e

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.info(f"Hủy yêu cầu cho task {context.task_id}")
        raise ServerError(error=UnsupportedOperationError("RAG Agent không hỗ trợ hủy yêu cầu"))
    
    async def execute_sync(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            start_time = datetime.now()
            result = self.agent.invoke(user_message)
            total_time = (datetime.now() - start_time).total_seconds()
            return {
                "type": "message",
                "role": "agent",
                "parts": [
                    {
                        "type": "text",
                        "text": result['answer']
                    }
                ],
                "messageId": str(uuid.uuid4()),
                "metadata": {
                    "sources": result.get("sources", []),
                    "relevant_documents_count": result.get("relevant_documents_count", 0),
                    "total_retrieved_count": result.get("total_retrieved_count", 0),
                    "processing_time": total_time,
                    "status": result.get("status", "completed")
                }
            }
        except Exception as e:
            return {
                "type": "error",
                "error": f"Lỗi: {str(e)}",
                "messageId": str(uuid.uuid4())
            }
    
    def health_check(self) -> Dict[str, Any]:
        try:
            health = self.agent.health_check()
            return health
        except Exception as e:
            logger.error(f"RAG Agent health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }