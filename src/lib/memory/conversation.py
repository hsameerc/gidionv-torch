from collections import deque
from typing import List, Dict


class ConversationHistory:
    """
    A simple, in-memory ring buffer for short-term conversation history.
    It now includes a helper to format the history for use in a prompt.
    """

    def __init__(self, max_turns: int = 5):
        """
        Initializes the history buffer.

        Args:
            max_turns: The number of user/AI turn pairs to remember.
        """
        self.history = deque(maxlen=max_turns * 2)

    def add(self, role: str, text: str):
        """
        Adds a new entry to the conversation history.

        Args:
            role: The speaker, e.g., "User" or "Gideon".
            text: The content of the message.
        """
        self.history.append({"role": role, "content": text})

    def get_history_as_string(self):
        return "\n".join(f"{item['role']}: {item['content']}" for item in self.history)

    def get_recent_turns_as_list(self) -> List[Dict[str, str]]:
        """Returns the history as a list of dictionaries."""
        return list(self.history)

    def get_as_prompt_string(self) -> str:
        """
        Formats the entire conversation history into a single string,
        suitable for inclusion in a model's prompt.

        Returns:
            A formatted string of the conversation, or a default message if empty.
        """
        if not self.history:
            return "The conversation has just begun."
        return "\n".join([f"{turn['role'].upper()}: {turn['content']}" for turn in self.history])

    def clear(self):
        """Clears all entries from the history."""
        self.history.clear()
