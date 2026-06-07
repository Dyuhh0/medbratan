import os
from dotenv import load_dotenv

load_dotenv()

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HF_TOKEN = os.environ.get("HF_TOKEN")

HUGGINGFACE_REPO_ID="Qwen/Qwen2.5-7B-Instruct"
DB_FAISS_PATH=os.path.join(_BASE_DIR, "vectorstore", "db_faiss")
DATA_PATH=os.path.join(_BASE_DIR, "data")
CHUNK_SIZE=500
CHUNK_OVERLAP=50