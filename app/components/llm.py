import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from app.common.logger import get_logger

logger = get_logger(__name__)

def load_llm(huggingface_repo_id=None, hf_token=None):
    try:
        token = hf_token or os.environ.get("HF_TOKEN")
        if token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = str(token)

        # Qwen идеально подходит для чата
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        logger.info(f"Connecting to Chat API with model: {model_id}")
        
        # 1. Базовый эндпоинт с принудительной задачей conversational
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=model_id,
            task="conversational", 
            temperature=0.2,
            max_new_tokens=256
        )
        
        # 2. Оборачиваем в чат-модель (для совместимости с Together AI)
        chat_model = ChatHuggingFace(llm=llm_endpoint)
        return chat_model
        
    except Exception as e:
        logger.error(f"failed to load cloud chat llm | Error: {e}")
        return None
