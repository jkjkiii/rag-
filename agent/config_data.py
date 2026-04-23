
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

md5_path = os.path.join(BASE_DIR, "md5.txt")

collection_name = "rag"
persist_directory = os.path.join(BASE_DIR, "chroma_db")

chunk_size = 1000
chunk_overlap = 200
separators = ["\n\n", "\n", " ", ""]

max_split_char_number = 1000

similarity_top_k = 2

embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"


session_config = {
        "configurable": {
            "session_id": "user_001"
        }
    }