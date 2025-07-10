from typing import List

import numpy as np
import torch

from .conversation import ConversationHistory
from .fiass_memory_core import FaissMemoryCore
from .intent_router import IntentRouter


class MemoryController:
    """
    The cognitive controller for the multi-modal, multi-memory transformer.
    This controller manages memory, detects intent, prepares inputs dynamically,
    and orchestrates the model's generation process.
    """

    def __init__(self, model: torch.nn.Module, factual_memory: FaissMemoryCore):
        print("[INIT] PyTorch Memory Controller starting...")
        self.model = model
        self.device = next(model.parameters()).device
        self.factual_memory = factual_memory
        self.conversation_history = ConversationHistory(max_turns=5)
        self.intent_router = IntentRouter()
        # self.prompt_templates = get_prompt_templates() # This logic will be more integrated now

        self.num_memory_slots = self.model.config['model']['num_memory_streams']
        print(f"[INIT] Controller configured for {self.num_memory_slots} memory slots.")

        self.reset_conversation()
        print("[INIT] Memory Controller is online.")

    def reset_conversation(self):
        """Resets the conversational state."""
        print("[CONTROLLER] Conversation state has been reset.")
        self.conversation_history.clear()

    @torch.no_grad()
    def _get_embedding(self, text: str) -> 'np.ndarray':
        """Helper to get a single embedding for FAISS, which uses NumPy."""
        if hasattr(self.model, 'encode_text_to_vectors'):
            return self.model.encode_text_to_vectors([text], pooling_strategy='mean')[0].cpu().numpy()
        else:
            print("Warning: `encode_text_to_vectors` not found on model. Using dummy embeddings.")
            return np.random.rand(int(self.model.d_model))

    def _prepare_memory_streams(self, query: str, intent: str, top_k: int) -> List[List[int]]:
        """
         Gathers factual and conversational memories and prepares
        them as a list of token ID lists, respecting the model's memory slot count.
        """
        memory_slots = []

        # Factual Memory
        if intent in ["qa_with_context", "general_qa"]:
            relevant_memories = self.factual_memory.find_similar_memories(
                query, self._get_embedding, top_k
            )
            # Combining all retrieved facts into a single context string
            factual_text = " ".join(mem[2] for mem in relevant_memories)
            memory_slots.append(self.model.tokenizer.encode(factual_text))

        # Conversational History
        history_text = self.conversation_history.get_history_as_string()
        if history_text:
            memory_slots.append(self.model.tokenizer.encode(history_text))

        while len(memory_slots) < self.num_memory_slots:
            memory_slots.append([])
        return memory_slots[:self.num_memory_slots]

    def _prepare_prompt(self, query: str, intent: str) -> str:
        """Builds the final prompt string for the decoder."""
        special_tokens = {"USER": "<USER>", "ASSISTANT": "<ASSISTANT>"}
        prompt = f"{special_tokens['USER']}INSTRUCTION: {query}\n\n{special_tokens['ASSISTANT']}"
        return prompt

    @torch.no_grad()
    def process_query(self, query: str, top_k: int = 3, top_p: float = 0.9, temperature: float = 0.7,
                      max_new_tokens: int = 256) -> str:
        """The main thinking loop for processing a user query."""
        self.model.eval()
        print(f"\n>>> Query: '{query}'")

        has_context_potential = self.factual_memory.find_similar_memories(query, self._get_embedding)
        intent = self.intent_router.detect_intent(query, bool(has_context_potential))
        print(f"[CONTROLLER] Detected Intent: '{intent}'")

        memory_streams_ids_list = self._prepare_memory_streams(query, intent, top_k)
        prompt_text = self._prepare_prompt(query, intent)

        prompt_ids = torch.tensor([self.model.tokenizer.encode(prompt_text)], dtype=torch.long, device=self.device)

        generated_ids = self.model.generate(
            prompt_ids=prompt_ids,
            memory_streams_ids=memory_streams_ids_list,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.model.tokenizer.eos_token_id
        )

        prompt_len = prompt_ids.shape[1]
        newly_generated_ids = generated_ids[0, prompt_len:]
        response = self.model.tokenizer.decode(newly_generated_ids.tolist(), skip_special_tokens=True).strip()

        self.conversation_history.add("USER", query)
        self.conversation_history.add('ASSISTANT', response)

        print(f"\n[Gideon] Response: {response}")
        return response

    def learn_from_qa(self, question: str, answer: str):
        """Adds a new piece of factual knowledge to the memory."""
        print(f"\n[LEARN] Teaching Gideon that '{question}' -> '{answer}'")
        # This logic can stay largely the same as FAISS uses NumPy
        self.factual_memory.add_memory(question, self._get_embedding)
        self.factual_memory.add_memory(answer, self._get_embedding)
        print("[LEARN] Knowledge successfully stored.")

    def shutdown(self):
        """Safely shuts down the memory system."""
        print("\n[SHUTDOWN] Closing controller...")
