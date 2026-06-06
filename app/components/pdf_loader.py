import os
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH,CHUNK_SIZE,CHUNK_OVERLAP

logger = get_logger(__name__)

def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException("Data path doesn't exist")
        
        logger.info(f"Loading files from {DATA_PATH}")

        # Добавлен параметр silent_errors=True
        loader = DirectoryLoader(
            DATA_PATH, 
            glob="*.pdf", 
            loader_cls=PyPDFLoader,
            silent_errors=True  # Пропускает поврежденные файлы и логирует их
        )

        documents = loader.load()

        if not documents:
            logger.warning("no pdfs were found")
            return []

        logger.info(f"Succesfully fetched {len(documents)} documents")

        non_empty = [d for d in documents if d.page_content.strip()]
        skipped = len(documents) - len(non_empty)
        if skipped:
            logger.warning(f"Skipped {skipped} pages with no extractable text (corrupted/encrypted/image-only)")

        return non_empty
    
    except Exception as e:
        error_message = CustomException("failed to load pdf", e)
        logger.error(str(error_message))
        return []

    

def create_text_chunks(documents):
    try:
        if not documents:
            raise CustomException("No documents were found")
        
        logger.info(f"Splitting {len(documents)} documents into chunks")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)

        text_chunks = text_splitter.split_documents(documents)

        logger.info(f"Generated {len(text_chunks)} text chunks")
        return text_chunks
    
    except Exception as e:
        error_message = CustomException("Failed to generate chunks" , e)
        logger.error(str(error_message))
        return []




