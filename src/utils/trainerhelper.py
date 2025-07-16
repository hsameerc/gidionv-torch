import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


def get_learning_rate(step: int, config: Dict) -> float:
    """Implements linear warmup and cosine decay LR schedule."""
    warmup = config['WARMUP_STEPS']
    decay_steps = config['TOTAL_DECAY_STEPS']
    peak_lr = config['PEAK_LEARNING_RATE']
    min_lr = config['MIN_LEARNING_RATE']

    if step < warmup:
        return peak_lr * (step / warmup)
    if step > decay_steps:
        return min_lr

    # Cosine decay phase
    progress = (step - warmup) / (decay_steps - warmup)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine_decay


@torch.no_grad()
def calculate_validation_loss(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
                              device: torch.device) -> Tuple[float, float]:
    """
    Calculates validation loss and perplexity on a validation set.
    """
    print("Calculating validation loss...")
    model.eval()

    total_loss_sum = 0.0
    total_active_tokens = 0

    for batch in val_loader:
        # Move data to the correct device
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        # Unbind it along the `num_streams` dimension (dim=1).
        # This creates a tuple of 3 tensors.
        batched_memory_tensor = batch['memory_streams_ids'].to(device)
        unbound_streams = torch.unbind(batched_memory_tensor, dim=1)
        memory_streams_ids = list(unbound_streams)
        # Forward Pass
        logits, _, _ = model(input_ids=input_ids, memory_streams_ids=memory_streams_ids)

        # Loss Calculation
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        # Accumulate loss and token counts
        num_active_tokens_in_batch = (target_ids.view(-1) != -100).sum().item()

        if num_active_tokens_in_batch > 0:
            total_loss_sum += loss.item() * num_active_tokens_in_batch
            total_active_tokens += num_active_tokens_in_batch

    model.train()

    if total_active_tokens == 0:
        print("Warning: No active tokens found in validation set. Returning 0 loss.")
        return 0.0, float('inf')

    # Calculate the final average loss and perplexity
    average_loss = total_loss_sum / total_active_tokens
    perplexity = math.exp(average_loss)

    return average_loss, perplexity
