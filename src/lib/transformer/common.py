import torch
import torch.nn.functional as F


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k == 0:
        return logits
    values, _ = torch.topk(logits, top_k)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_filtering(logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('inf')
    return logits
