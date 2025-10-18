from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler, RESTHandler
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotificationConfigStore, BasePushNotificationSender
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from agent_executor import RAGAgentExecutor
from config import Config
import logging
import click
import uvicorn
import sys
import httpx
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--host', default=Config.HOST)
@click.option('--port', default=Config.PORT)
def main(host, port):
    try:

        logger.info(f"Starting server on {host}:{port}")
        capabilities = AgentCapabilities(streaming=True, push_notifications=True)
        skills = AgentSkill(
            id="mental_health_consultation",
            name="Tư vấn sức khỏe tinh thần",
            description="Hỗ trợ tư vấn và cung cấp thông tin về sức khỏe tinh thần.",
            tags=["sức khỏe", "tinh thần", "tư vấn"],
            examples=[
                "Tôi cảm thấy căng thẳng và lo lắng, bạn có thể giúp tôi không?",
                "Làm thế nào để tôi có thể cải thiện giấc ngủ của mình?",
                "Bạn có thể cung cấp cho tôi một số kỹ thuật thư giãn không?"
            ]
        )

        agent_card = AgentCard(
            name="RAG Mental Health Agent",
            description="Một trợ lý ảo sử dụng Retrieval-Augmented Generation (RAG) để cung cấp thông tin và hỗ trợ về sức khỏe tinh thần.",
            url=f'http://{host}:{port}',
            version="1.0.0",
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            capabilities=capabilities,
            skills=[skills]
        )
        httpx_client = httpx.AsyncClient()
        push_config_store=InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store
        )
        request_handler = DefaultRequestHandler(
            agent_executor=RAGAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender
        )

        server = A2AFastAPIApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )

        uvicorn.run(server.build(), host=host, port=port)
    except Exception as e:
        logger.error(f"Không thể khởi tạo server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()