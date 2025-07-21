import argparse
import json
from typing import Dict

import torch

from src.data.saver_loader import load_checkpoint
from src.lib.memory.conversation import ConversationHistory
from src.loaders.finetune_loader import format_prompt

# Global device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InteractiveChat:
    """
    A stateful, interactive chat session with the Multi-Memory Transformer.
    It manages conversation history and prepares inputs for the model on each turn.
    """

    def __init__(self, config: Dict):
        print("Initializing Interactive Chat")
        self.config = config

        # Loading the fine-tuned model and tokenizer
        print("Loading fine-tuned model...")
        self.model, _, self.tokenizer, _ = load_checkpoint(config, device)
        self.model.eval()

        # Initializing state
        self.conversation_history = ConversationHistory(max_turns=5)
        self.special_tokens = {"USER": "<USER>", "ASSISTANT": "<ASSISTANT>", "INSTRUCTION": "<INST>", "RESPONSE": "<RESPONSE>"}
        self.num_memory_slots = self.config['model']['num_memory_streams']

    @torch.no_grad()
    def generate_response(self, user_input: str):
        """
        Takes a single user input, prepares all contexts, and generates a response.
        """
        # Preparing Memory Streams
        # The conversation history is now a primary memory source.
        history_text = self.conversation_history.get_history_as_string()

        # We will use the history as the first memory stream.
        memory_token_ids = [self.tokenizer.encode(history_text)]

        # Pad the remaining memory slots with empty streams
        empty_stream = []
        memory_token_ids.extend([empty_stream] * (self.num_memory_slots - 1))

        # The prompt should include the user's latest message.
        # The 'context' for the prompt can be a hint from the history.
        context_hint = history_text[:100] + "..." if history_text else ""
        prompt_text = format_prompt(user_input, context_hint, self.special_tokens)

        prompt_ids = torch.tensor([self.tokenizer.encode(prompt_text)], dtype=torch.long, device=device)

        # Generating
        print("\nGideon is thinking...")
        generated_ids = self.model.generate(
            prompt_ids=prompt_ids,
            memory_streams_ids=memory_token_ids,
            max_new_tokens=256,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.15,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Decoding and return
        prompt_len = prompt_ids.shape[1]
        newly_generated_ids = generated_ids[0, prompt_len:]
        response = self.tokenizer.decode(newly_generated_ids.tolist(), skip_special_tokens=True).strip()
        return response

    def start(self):
        """Starts the main interactive chat loop."""
        print("\nGideon is ready. Type 'exit' or 'quit' to end the conversation.")
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['exit', 'quit']:
                    break
                if not user_input:
                    continue

                # Generating a response
                gideon_response = self.generate_response(user_input)

                print(f"Gideon: {gideon_response}")
                self.conversation_history.add("User", user_input)
                self.conversation_history.add("Assistant", gideon_response)

            except (KeyboardInterrupt, EOFError):
                break

        print("\nGideon has gone to sleep.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an interactive chat with the Multi-Memory Transformer.")
    parser.add_argument('--config', default="configs/gidionv_multi_memory.json",
                        help="Path to the model's JSON config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    # Start the chat
    chat_session = InteractiveChat(cfg)
    chat_session.start()
