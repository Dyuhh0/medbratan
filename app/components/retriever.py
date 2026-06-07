from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store

from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException


logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """Answer the following medical question in 2-3 lines maximum using information provided in the context

Context:
{context}

Question:
{input}

Answer:
"""

def set_custom_prompt():
    return ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)

def create_qa_chain():
    try:
        logger.info("Loading vector store for context")
        db = load_vector_store()

        if db is None:
            raise CustomException("vector store is not present or empty")

        llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, hf_token=HF_TOKEN)

        if llm is None:
            raise CustomException("llm is not present or empty")

        combine_docs_chain = create_stuff_documents_chain(llm, set_custom_prompt())
        qa_chain = create_retrieval_chain(
            db.as_retriever(search_kwargs={'k': 1}),
            combine_docs_chain
        )

        logger.info("successfully created qa chain")
        return qa_chain

    except Exception as e:
        error_message = CustomException("failed to make a qa chain", e)
        logger.error(str(error_message))


