# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_core.embeddings import DashScopeEmbeddings
# from langchain_community.document_loaders import CSVLoader,JSONLoader,TextLoader,PyPDFLoader
# from langchain_chromadb import Chroma

# # vectorstore = InMemoryVectorStore(
# #     embedding_function=DashScopeEmbeddings()
# #     )

# vectorstore = Chroma(
#     collection_name="my_collection",        #指定集合名称
#     embedding_function=DashScopeEmbeddings(),
#     persist_directory="./chroma_db"         #指定Chroma数据库的存储目录
#     )
    

# loader  = CSVLoader(
#     file_path="./data/sample.csv",      
#     encoding="utf-8",
#     source_column= "source",
# )

# document = loader.load()

# vectorstore.add_documents(
#     documents = document,       #类型,list[Document]
#     ids = ["id_" + str(i) for i in range(len(document))]
#     )

# vectorstore.delete(ids=["id_0"])

# result = vectorstore.similarity_search(
#     query="What is the capital of France?", 
#     k=3,
#     filter = ""
#     )



from xml.dom.minidom import Document

from langchain_community.chat_models import ChatTongyi
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

model = ChatTongyi(model="qwen3-max")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的文档内容为基础，回答用户的问题。参考资料:{context}"),
        ("user", "用户问题:{input}"),
    ]
)

vectorstore = InMemoryVectorStore(
    embedding=DashScopeEmbeddings(model="text-embedding-v4")
)

vectorstore.add_texts(
    texts=["巴黎是法国的首都。", "伦敦是英国的首都。", "巴黎很美丽。"],
)
input_text = "法国的首都是什么？"
result = similar_docs = vectorstore.similarity_search(input_text, k=2)
reference_text = "["
for doc in similar_docs:
    reference_text += doc.page_content + ";"
    
reference_text += "]"


# chain = prompt | model | StrOutputParser()

# res = chain.invoke({"input": input_text, "context": reference_text})
# print(res)

retrieve = vectorstore.as_retriever(search_kwargs={"k": 2})

def format_func(retrieve : list[Document]) -> str:
    reference_text = "["
    for doc in retrieve:
        reference_text += doc.page_content + ";"
    
    reference_text += "]"
    return reference_text
chain = (
    {"input": RunnablePassthrough(), "context": retrieve | format_func} | prompt | model | StrOutputParser()
)

res = chain.invoke(input_text)
print(res)