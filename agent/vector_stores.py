from langchain_chroma import Chroma
import config_data as config

class VectorStoreService:
    def __init__(self, embedding):
        
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory
        )
        
    def get_retriever(self):
        """获取向量存储的检索器对象"""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": config.similarity_top_k})
        return retriever
            

    