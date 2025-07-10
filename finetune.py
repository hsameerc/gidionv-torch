import argparse
import csv
import json
import os
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.config import get_config
from src.data.saver_loader import save_checkpoint, load_checkpoint
from src.streamers.finetune_external_streamer import FinetuneDatasetStream, FinetuneValidationDataset
from src.utils.trainerhelper import get_learning_rate, calculate_validation_loss


def train(config: Dict[str, Any]):
    """[V4-PyTorch] The main, production-ready training script using the PyTorch framework."""
    # Initial Setup
    if config['RANDOM_SEED']:
        torch.manual_seed(config['RANDOM_SEED'])
    os.makedirs(config['MODEL_DIR'], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model_path = os.path.join(config['MODEL_DIR'], f"{config['MODEL_NAME']}.pth")

    # Starting or Resuming Training
    model, optimizer, tokenizer, train_state = load_checkpoint(config, device)
    if config['resume_training']:
        total_steps = train_state['total_steps']
        best_val_loss = train_state['best_val_loss']
        start_epoch = train_state['start_epoch']
    else:
        total_steps = 0
        best_val_loss = float('inf')
        start_epoch = 0

    # Logging Setup
    log_file_path = os.path.join(config['MODEL_DIR'], config['LOG_FILE_NAME'])
    log_file = open(log_file_path, 'a', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    if log_file.tell() == 0:
        log_writer.writerow(["step", "epoch", "loss", "val_loss", "perplexity", "lr", "grad_norm"])

    use_amp = config.get('use_amp', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    special_tokens = {"USER": "<USER>", "ASSISTANT": "<ASSISTANT>", "INST": "<INST>", "END_INST": "</INST>"}
    # Validation Data loader
    val_dataset = FinetuneValidationDataset(tokenizer=tokenizer, config=config,
                                  special_tokens=special_tokens)
    val_data_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], num_workers=config.get('NUM_WORKERS', 1),
                                 persistent_workers=True)
    for epoch in range(start_epoch, config['EPOCHS']):
        print(f"\n{'=' * 25} Epoch {epoch + 1}/{config['EPOCHS']} {'=' * 25}")

        # Training Data loader
        dataset = FinetuneDatasetStream(tokenizer=tokenizer, config=config,
                                  special_tokens=special_tokens)
        data_loader = DataLoader(dataset, batch_size=config['BATCH_SIZE'], num_workers=config.get('NUM_WORKERS', 1),
                                 persistent_workers=True)
        model.train()
        accum_loss = 0.0
        for i, batch in enumerate(data_loader):
            batch: Dict[str, torch.Tensor]
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            batched_memory_tensor = batch['memory_streams_ids'].to(device)
            unbound_streams = torch.unbind(batched_memory_tensor, dim=1)
            memory_streams_ids = list(unbound_streams)
            batched_mask_tensor = batch['memory_padding_masks'].to(device)
            unbound_masks = torch.unbind(batched_mask_tensor, dim=1)
            memory_padding_masks = list(unbound_masks)

            # Forward Pass
            use_amp = config.get('use_amp', False) and device.type == 'cuda'
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, _ = model(input_ids=input_ids, memory_streams_ids=memory_streams_ids,
                                  memory_padding_masks=memory_padding_masks)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                accum_loss += loss.item()
                loss_for_backward = loss / config['GRADIENT_ACCUMULATION_STEPS']

            # Backward Pass & Gradient Accumulation
            scaler.scale(loss_for_backward).backward()

            if (i + 1) % config['GRADIENT_ACCUMULATION_STEPS'] == 0:
                total_steps += 1
                scaler.unscale_(optimizer)

                # Gradient Clipping
                if config['CLIP_THRESHOLD'] > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['CLIP_THRESHOLD'])
                else:
                    grad_norm = torch.sqrt(
                        torch.sum(torch.stack([p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None])))

                scaler.step(optimizer)
                scaler.update()

                # Learning Rate Update
                lr = get_learning_rate(total_steps, config)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # Optimizer Step
                optimizer.zero_grad()
                grad_norm_val = grad_norm.item() if config['CLIP_THRESHOLD'] > 0 else 0.0
                avg_loss = accum_loss / config['GRADIENT_ACCUMULATION_STEPS']
                print(
                    f"Epoch {epoch + 1} | Step {total_steps: >6} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Grad Norm: {grad_norm_val:.4f}")
                accum_loss = 0.0

                # Logging & Validation
                if total_steps % config['LOG_EVERY_N_STEPS'] == 0:
                    log_data = [total_steps, epoch + 1, f"{avg_loss:.4f}", "N/A", "N/A", f"{lr:.2e}",
                                f"{grad_norm_val:.4f}"]
                    log_writer.writerow(log_data)
                    log_file.flush()
                if total_steps % config['EVAL_EVERY_N_STEPS'] == 0:
                    val_loss, val_ppl = calculate_validation_loss(model=model, val_loader=val_data_loader,
                                                                  criterion=criterion, device=device)
                    print(f"VALIDATION @ Step {total_steps: >6} | Val Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
                    log_data = [total_steps, epoch + 1, "N/A", f"{val_loss:.4f}", f"{val_ppl:.2f}", f"{lr:.2e}", "N/A"]
                    log_writer.writerow(log_data)
                    log_file.flush()

                    # Save a "best" model checkpoint
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print(f"🎉 New best validation loss! Saving best model to {model_path}... 🎉")
                        save_checkpoint(model, optimizer, config, total_steps, best_val_loss, epoch, is_best=True)

                if total_steps % config['SAVE_EVERY_N_STEPS'] == 0:
                    # Saving Checkpoint
                    save_checkpoint(model, optimizer, config, total_steps, best_val_loss, epoch, is_best=False)

    log_file.close()
    print("\nTraining Finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the PyTorch Multi Memory Transformer.")
    parser.add_argument('--config', default='configs/gidionv_multi_memory_finetune.json', type=str)
    args = parser.parse_args()

    cfg = get_config()
    if args.config:
        with open(args.config, 'r') as f:
            cfg.update(json.load(f))

    train(cfg)
