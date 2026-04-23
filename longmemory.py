import json
import os
from typing import Sequence

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

class FileChatMessageHistory(BaseChatMessageHistory):
    """File-based chat message history."""

    def __init__(self, session_id: str, storage_path: str):
        """Initialize the file chat message history."""
        self.session_id = session_id
        self.file_path = os.path.join(storage_path, f"{session_id}.json")
        os.makedirs(storage_path, exist_ok=True)
        # Ensure the file exists
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history."""
        self.add_messages([message])

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add multiple messages and persist the full history."""
        all_messages = list(self.messages)
        all_messages.extend(messages)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(messages_to_dict(all_messages), f, ensure_ascii=False, indent=2)

    @property
    def messages(self) -> list[BaseMessage]:
        """Get all messages in the history."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
            return messages_from_dict(messages_data)

        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def clear(self) -> None:
        """Clear the chat history."""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)


def run_long_memory_qa_test(session_id: str = "long-memory-demo") -> None:
    """Run a simple two-turn QA test to verify long-term memory behavior."""
    storage_path = "./chat_memory_store"

    model = ChatTongyi(model="qwen-max")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个记忆稳定的中文助手。你会结合历史对话回答问题，"
                "当用户询问个人信息时，优先从历史中提取。",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    chain = prompt | model | StrOutputParser()

    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history=lambda sid: FileChatMessageHistory(
            session_id=sid, storage_path=storage_path
        ),
        input_messages_key="question",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": session_id}}

    print("=== 长期记忆问答测试开始 ===")

    q1 = "我叫小李，我喜欢打羽毛球和喝拿铁。请你记住。"
    a1 = with_history.invoke({"question": q1}, config=config)
    print(f"\n[第1轮-用户] {q1}")
    print(f"[第1轮-助手] {a1}")

    q2 = "你还记得我叫什么、喜欢做什么吗？"
    a2 = with_history.invoke({"question": q2}, config=config)
    print(f"\n[第2轮-用户] {q2}")
    print(f"[第2轮-助手] {a2}")

    print("\n记忆文件:", os.path.join(storage_path, f"{session_id}.json"))
    print("=== 测试结束 ===")


if __name__ == "__main__":
    run_long_memory_qa_test()


