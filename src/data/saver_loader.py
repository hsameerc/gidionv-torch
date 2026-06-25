import os
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.lib.transformer.memory_of_experts_transformer import MemoryOfExpertsTransformer
from src.lib.transformer.multi_memory_transformer import MultiMemoryTransformer


def save_checkpoint(model: nn.Module, optimizer: Optimizer, config: Dict[str, Any], total_steps: int,
                    best_val_loss: float, epoch: int, is_best: bool = False):
    """
    Saves a comprehensive training checkpoint.

    Args:
        model: The PyTorch model to save.
        optimizer: The optimizer whose state needs to be saved.
        config: The configuration dictionary.
        total_steps: The current total number of training steps.
        best_val_loss: The best validation loss achieved so far.
        epoch: The current epoch number.
        is_best: If True, saves the model as '..._best.pth' without optimizer state.
                 If False, saves a full checkpoint as '..._latest.pth'.
    """
    model_dir = config['MODEL_DIR']
    model_name = config['MODEL_NAME']
    os.makedirs(model_dir, exist_ok=True)

    # State dictionary to save.
    checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                  'config': config, 'total_steps': total_steps, 'best_val_loss': best_val_loss,
                  'last_trained_epoch': epoch}

    if is_best:
        save_path = os.path.join(model_dir, f"{model_name}_best.pth")
        resume_save_path = os.path.join(model_dir, f"{model_name}_best_resume.pth")
        torch.save(model.state_dict(), save_path)
        torch.save(checkpoint, resume_save_path)
        print(f"[SUCCESS] New best model saved to {save_path}")
    else:
        save_path = os.path.join(model_dir, f"{model_name}_latest.pth")
        torch.save(checkpoint, save_path)
        print(f"Regular checkpoint saved to {save_path}")


def load_checkpoint(config: Dict[str, Any], device: torch.device, use_best: bool = False) -> Tuple[
    nn.Module, Optimizer, HFTokenizerWrapper, Dict[str, Any]]:
    """
    Loads a model and optimizer state from a checkpoint. If no checkpoint exists,
    it initializes a new model and returns default training progress.

    Args:
        config: The configuration dictionary.
        device: The device to load the model onto.
        use_best: Use Best Modal saved checkpoint path to load
    Returns:
        A tuple containing:
        - model: The initialized or loaded model.
        - optimizer: The initialized or loaded optimizer.
        - tokenizer: The tokenizer instance.
        - train_state: A dictionary with 'total_steps', 'best_val_loss', 'start_epoch'.
    """
    model_dir = config['MODEL_DIR']
    model_name = config['MODEL_NAME']
    checkpoint_path = os.path.join(model_dir, f"{model_name}_latest.pth")
    best_checkpoint_path = os.path.join(model_dir, f"{model_name}_best.pth")

    # Initializing empty model and optimizer
    print("Initializing model architecture...")
    tokenizer = HFTokenizerWrapper(config['TOKENIZER_PATH'])
    architecture = config.get('MODEL_ARCHITECTURE', 'multi_memory')
    if architecture == 'moe':
        print("Using MemoryOfExpertsTransformer architecture.")
        model = MemoryOfExpertsTransformer(config, tokenizer).to(device)
    else:
        print("Using MultiMemoryTransformer architecture.")
        model = MultiMemoryTransformer(config, tokenizer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['PEAK_LEARNING_RATE'],
                                  betas=(config['ADAM_BETA1'], config['ADAM_BETA2']), weight_decay=config['WEIGHT_DECAY'])

    # Loading state from checkpoint if it exists
    if use_best and os.path.exists(best_checkpoint_path):
        print(f"Loading checkpoint from {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("Successfully loaded model.")
        train_state = {'total_steps': 0, 'best_val_loss': float('inf'), 'start_epoch': 0}
        return model, optimizer, tokenizer, train_state
    elif os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Successfully loaded model and optimizer state.")

        # Creating the training state dictionary from the checkpoint
        train_state = {'total_steps': checkpoint.get('total_steps', 0),
                       'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
                       'start_epoch': checkpoint.get('last_trained_epoch', 0)}
        # Returning model, optimizer, tokenizer and train state
        return model, optimizer, tokenizer, train_state

    else:
        print("No checkpoint found. Initializing a new model from scratch.")
        if hasattr(model, 'init_weights'):
            print("Applying custom weight initialization...")
            model.apply(model.init_weights)

        #  Creating a default training state for a new run
        train_state = {'total_steps': 0, 'best_val_loss': float('inf'), 'start_epoch': 0}

        return model, optimizer, tokenizer, train_state
