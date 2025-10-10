from agent import RAGAgent
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
import logging
logger = logging.getLogger(__name__)

class RAGAgentExecutor(AgentExecutor):
    def __init__(self):
        self.agent=RAGAgent()

    async def execute(
        self, 
        context: RequestContext,
        event_queue: EventQueue  ):
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            raise ValueError("RequestContext must have a message")
        
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.contextId)

        try:
            query = context.get_user_input()
            print(f"Processing Query: {query} ...")
            # Thực hiện yêu cầu của người dùng
            # TODO: Tạo hàm invoke ở class RAGAgent để reply yêu cầu
            # result = self.agent.invoke(query)
        except Exception as e:
            