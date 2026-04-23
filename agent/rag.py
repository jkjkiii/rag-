from vector_stores import VectorStoreService
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from file_history_store import get_file_chat_history

class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model = config.embedding_model_name)
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "以我提供的资料为主,简洁、准确、专业的回答我的问题。参考资料:{context}"),
            ("system","并且我提供用户的历史信息，作为补充参考:"),
            MessagesPlaceholder(variable_name="history"),
            ("user", "请回答我的问题: {input}"),
        ])
        
        self.chat_model = ChatTongyi(model=config.chat_model_name)  # 聊天模型实例对象
        
        self.chain = self.__get_chain()
        
    def __get_chain(self):
        retriever = self.vector_service.get_retriever()

        def format_document(documents:list[Document]):
            if not documents:
                return "无相关资料"
            formatted_docs = ""
            for doc in documents:
                formatted_docs += f"文档片段: {doc.page_content}\n文档单元数据: {doc.metadata}\n\n"
            return formatted_docs
        
        def format_for_retriever(value):
            print(f"Input type: {type(value)}")
            return value["input"]
        def format_for_prompt(value):
            print(f"Retriever output type: {type(value)}")
            new_value = {}
            new_value["context"] = value["context"]
            new_value["input"] = value["input"]["input"]
            new_value["history"] = value["input"]["history"]

            print("\n========== Prompt Debug ==========")
            print("[参考资料]")
            print(new_value["context"])

            print("\n[历史信息]")
            if not new_value["history"]:
                print("无历史信息")
            else:
                for idx, msg in enumerate(new_value["history"], start=1):
                    role = getattr(msg, "type", msg.__class__.__name__)
                    content = getattr(msg, "content", str(msg))
                    print(f"{idx}. ({role}) {content}")

            print("\n[当前问题]")
            print(new_value["input"])
            print("========== End Debug ==========\n")
            return new_value
        
        
        chain = (
            {
                "input":RunnablePassthrough(),
                "context":RunnableLambda(format_for_retriever) |retriever | format_document
            } | RunnableLambda(format_for_prompt) | self.prompt_template | self.chat_model | StrOutputParser()
        )
        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_file_chat_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        
        return conversation_chain


if __name__ == "__main__":  
    session_config = {
        "configurable": {
            "session_id": "user_001"
        }
    }
    res = RagService().chain.invoke({"input": "春天穿什么颜色衣服？我的体重是180斤"}, config=session_config)
    print(res)