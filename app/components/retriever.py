from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

# ПЕРЕВОДИМ ПРОМПТ В СТАНДАРТ ЧАТА (Роли system и human)
def set_custom_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "Answer the following medical question in 2-3 lines maximum using information provided in the context.\n\nContext:\n{context}"),
        ("human", "{input}")
    ])

def _format_docs(docs):
    if not docs:
        return "No relevant medical context found."
    return "\n\n".join(doc.page_content for doc in docs)

def create_qa_chain():
    try:
        logger.info("Loading vector store for context")
        db = load_vector_store()

        if db is None:
            raise CustomException("vector store is not present or empty")

        llm = load_llm()

        if llm is None:
            raise CustomException("llm is not present or empty")

        retriever = db.as_retriever(search_kwargs={'k': 1})

        # Конвейер LCEL с чат-моделью и парсером строк
        qa_chain = (
            RunnablePassthrough.assign(
                context=lambda x: _format_docs(retriever.invoke(x["input"]))
            )

            | set_custom_prompt()
            | llm
            | StrOutputParser()
        )

        logger.info("successfully created qa chain")
        return qa_chain

    except Exception as e:
        error_message = CustomException("failed to make a qa chain", e)
        logger.error(str(error_message))
        return None
