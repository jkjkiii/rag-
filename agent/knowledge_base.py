import os

import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from datetime import datetime
    
def check_md5(md5_str: str):
    """
    检查md5值是否已处理过，避免重复处理同一数据
    """
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding='utf-8').close()  # 创建空的md5文件
        return False
    else:
        for line in open(config.md5_path, 'r', encoding='utf-8'):
            if line.strip() == md5_str:
                return True
        return False

    
def save_md5(md5_str: str):
    """
    保存md5值到文件中
    """
    with open(config.md5_path, 'a', encoding='utf-8') as f:
        f.write(md5_str + '\n')


def get_string_md5(input_str: str, encoding='utf-8'):
    str_bytes = input_str.encode(encoding)   
    md5_hash = hashlib.md5(str_bytes)
    return md5_hash.hexdigest() 

class KnowledgeBase:
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)  # 确保持久化目录存在
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model = "text-embedding-v4"),
            persist_directory=config.persist_directory
            )      #向量存储的实例对象
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len
        )    #文本分割的实例对象
        
    def upload_by_str(self, data, file_name):
        #将字符串数据向量化上传到向量存储中
        md5_str = get_string_md5(data)
        if check_md5(md5_str):
            return f"{file_name} 已经上传过了，无需重复上传"
            
        if len(data) > config.max_split_char_number:
            chunks = self.splitter.split_text(data)
        else:
            chunks = [data]
            
        metadata = {
            "source": file_name, 
            "create_time": datetime.now().isoformat(),
            "operater": "lzp"}
        self.chroma.add_texts(
            chunks, 
            metadatas = [metadata for _ in chunks]
            )
            
        save_md5(md5_str)
        return f"{file_name} 上传成功，分成 {len(chunks)} 块文本"