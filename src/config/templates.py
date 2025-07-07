def get_prompt_templates():
    return {
        "qa_with_context": {
            "template": (
                "### Instruction:\n"
                "Answer the question based on the provided context. If the answer is not found in the context, state that you don't know.\n\n"
                "### Context:\n{context}\n\n"
                "### Question:\n{input}\n\n"
                "### Response:\n"
            ),
            "requires_context": True,
            "description": "Used for factual questions that must be answered from a specific document (RAG)."
        },
        "qa_without_context": {
            "template": (
                "### Instruction:\n"
                "Answer the following question using your general knowledge.\n\n"
                "### Question:\n{input}\n\n"
                "### Response:\n"
            ),
            "requires_context": False,
            "description": "Used for open-domain, general knowledge questions."
        },
        "multi_turn_chat": {
            "template": (
                "### Instruction:\n"
                "You are Gideon, a friendly and helpful AI assistant. Continue the conversation with the user, using the previous turns as context.\n\n"
                "### Context:\n{context}\n\n"
                "### User Input:\n{input}\n\n"
                "### Response:\n"
            ),
            "requires_context": True,
            "description": "For ongoing conversations where chat history is important."
        },

        "summarization": {
            "template": (
                "### Instruction:\n"
                "Summarize the following text in a clear and concise manner, capturing the main points.\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n"
            ),
            "requires_context": False,
            "description": "Used for summarization tasks."
        },
        "keyword_extraction": {
            "template": (
                "### Instruction:\n"
                "Extract the most relevant keywords from the following text. List them as a comma-separated list.\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n"
            ),
            "requires_context": False,
            "description": "Extracts keywords or tags from a block of text."
        },
        "json_conversion": {
            "template": (
                "### Instruction:\n"
                "Extract the key information from the text below and format it as a JSON object with the following keys: {json_keys}.\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n"
            ),
            "requires_context": False,
            "description": "Transforms unstructured text into a structured JSON format."
        },
        "creative_writing": {
            "template": (
                "### Instruction:\n{instruction}\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n"
            ),
            "requires_context": False,
            "description": "Flexible template for creative tasks like writing poems, stories, or marketing copy. The main instruction is passed in the {instruction} field."
        },
        "code_generation": {
            "template": (
                "### Instruction:\n"
                "You are an expert programmer. Write a code snippet that accomplishes the following task. Make sure to use best practices and include comments.\n\n"
                "### Task:\n{input}\n\n"
                "### Response:\n"
            ),
            "requires_context": False,
            "description": "Generates code based on a natural language description."
        },
        "translation": {
            "template": (
                "### Instruction:\n"
                "Translate the following text from {source_language} to {target_language}.\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n"
            ),
            "requires_context": False,
            "description": "Translates text between languages."
        },
        "cot_reasoning": {
            "template": (
                "### Instruction:\n"
                "Solve the following problem by thinking step-by-step. First, lay out your reasoning, then provide the final answer.\n\n"
                "### Problem:\n{input}\n\n"
                "### Response:\n"
            ),
            "requires_context": False,
            "description": "Chain-of-Thought reasoning for complex problems (math, logic puzzles)."
        },
        "general_conversation": {
            "template": "### INSTRUCTION:\nAct as Gideon, a friendly and engaging AI assistant. Below is the recent conversation history and some relevant context from your long-term memory. Use them to answer the user's latest message.\n\n### CONVERSATION HISTORY:\n{chat_history}\n\n### RELEVANT CONTEXT:\n{context}\n\n### USER:\n{input}\n\n### ASSISTANT:\n",
            "requires_memory_search": True,
            "description": "Default conversational fallback."
        },
        "general_instruction": {
            "template": (
                "### Instruction:\n{instruction}\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n"
            ),
            "requires_context": False,
            "description": "A general-purpose, Alpaca-style template for instructions that don't fit other categories."
        },
        "story_generation": {
            "template": "### INSTRUCTION:\nYou are a creative storyteller. Your task is to continue the story provided below in a coherent and imaginative way.\n\n### STORY SO FAR:\n{input}\n\n### CONTINUATION:\n",
            "requires_memory_search": False,
            "description": "Task for teaching narrative continuation. Used for pre-training on story datasets."
        }
    }


def get_intent_rules():
    return {
        "story_generation": [
            "\\b(tell|write|create|generate) me a story\\b",
            "\\bmake up a story\\b",
            "\\btell me a tale\\b",
            "\\bcan you write a story\\b"
        ],
        "json_conversion": [
            r"\bextract to json\b", r"\bformat as json\b", r"\b(into|as) a json\b"
        ],
        "keyword_extraction": [
            r"\b(extract|find|get|list) the keywords\b", r"\b(what are the )?key topics\b",
            r"\b(find|suggest) tags for\b"
        ],
        "translation": [
            r"\btranslate\b", r"\b(in|to) (spanish|french|german|japanese|chinese|italian)\b"
        ],
        "code_generation": [
            r"\bwrite code\b", r"\b(code|script|snippet) for\b", r"\b(python|javascript|java|c\+\+) function\b"
        ],
        "summarization": [
            r"\bsummarize\b", r"\bsummary\b", r"\btldr\b", r"\bgive me the gist\b", r"\b(short|brief) version\b",
            r"\bkey points of\b", r"\bcondense this\b"
        ],
        "cot_reasoning": [
            r"\bsolve this\b", r"\bcalculate\b", r"\bthink step-by-step\b", r"what is the logic",
            r"\b(math|logic) problem\b", r"puzzle"
        ],
        "creative_writing": [
            r"\bwrite a\b", r"\bcreate a\b", r"\bcompose a\b", r"\bdraft an?\b", r"\btell me a story\b",
            r"\b(poem|story|essay|song)\b"
        ],

        "qa_with_context": [
            r"\bwhat\b", r"\bwho\b", r"\bwhy\b", r"\bwhere\b", r"\bwhen\b", r"\bhow\b", r"\?$"
        ],
        "qa_without_context": [
            r"\bwhat is\b", r"\bwho is\b", r"\bwhy do\b", r"\bexplain\b", r"\btell me about\b", r"\?$"
        ],
        "multi_turn_chat": [],
        "general_instruction": [],
    }
