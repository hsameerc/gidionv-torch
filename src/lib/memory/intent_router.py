import re

from src.config.templates import get_intent_rules


class IntentRouter:
    def __init__(self):
        self.intent_order = [
            "json_conversion",
            "keyword_extraction",
            "translation",
            "code_generation",
            "summarization",
            "cot_reasoning",
            "story_generation",
            "creative_writing",
            "qa_general"
        ]
        self.intent_rules = get_intent_rules()

    def detect_intent(self, query: str, has_context: bool) -> str:
        """
        Detects user's intent based on the query and whether context was found.

        :param query: The user's input text.
        :param has_context: A boolean, True if the memory search found relevant documents.
        :return: The name of the detected intent.
        """
        query_lower = query.lower()
        for intent in self.intent_order:
            if intent == "qa_with_context":
                continue

            rules = self.intent_rules.get(intent, [])
            for rule in rules:
                if re.search(rule, query_lower):
                    print(f"[IntentRouter] Matched rule '{rule}' for intent '{intent}'.")
                    return intent

        if has_context:
            print("[IntentRouter] Context found, defaulting to 'qa_with_context'.")
            return "qa_with_context"

        # If all else fails, treat it as a general instruction or conversation.
        print("[IntentRouter] No specific intent matched, using 'general_conversation'.")
        return "general_conversation"
