from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
import os
from langchain_core.chat_history import BaseChatMessageHistory
import json
from typing import Sequence

def get_file_chat_history(session_id: str):
    """Get a file-based chat message history instance."""
    return FileChatMessageHistory(session_id=session_id, storage_path="./chat_histories")


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
