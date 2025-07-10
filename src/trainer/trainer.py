import csv
import os
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.saver_loader import save_checkpoint, load_checkpoint
from src.trainer.datafactory import get_data_components
from src.utils.trainerhelper import get_learning_rate, calculate_validation_loss


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def train(self):
        """
            The main, production-ready training script using the PyTorch framework.
        """
        # Initial Setup
        if self.config['RANDOM_SEED']:
            torch.manual_seed(self.config['RANDOM_SEED'])
        os.makedirs(self.config['MODEL_DIR'], exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # CrossEntropyLoss
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        model_path = os.path.join(self.config['MODEL_DIR'], f"{self.config['MODEL_NAME']}.pth")

        # Starting or Resuming Training
        model, optimizer, tokenizer, train_state = load_checkpoint(self.config, device)
        dataset_factory = get_data_components(self.config, tokenizer=tokenizer)

        total_steps = train_state['total_steps']
        best_val_loss = train_state['best_val_loss']
        start_epoch = train_state['start_epoch']

        # Logging Setup
        log_file_path = os.path.join(self.config['MODEL_DIR'], self.config['LOG_FILE_NAME'])
        log_file = open(log_file_path, 'a', newline='', encoding='utf-8')
        log_writer = csv.writer(log_file)
        if log_file.tell() == 0:
            log_writer.writerow(["step", "epoch", "loss", "val_loss", "perplexity", "lr", "grad_norm"])

        use_amp = self.config.get('use_amp', False) and device.type == 'cuda'
        scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

        # Validation Data loader
        val_dataset = dataset_factory.create_validation_dataset()
        val_data_loader = DataLoader(val_dataset, batch_size=self.config['BATCH_SIZE'],
                                     num_workers=self.config.get('NUM_WORKERS', 1),
                                     persistent_workers=True)
        for epoch in range(start_epoch, self.config['EPOCHS']):
            print(f"\n{'=' * 25} Epoch {epoch + 1}/{self.config['EPOCHS']} {'=' * 25}")

            # Training Data loader
            stream_dataset = dataset_factory.create_training_dataset()
            data_loader = DataLoader(stream_dataset, batch_size=self.config['BATCH_SIZE'],
                                     num_workers=self.config.get('NUM_WORKERS', 1), persistent_workers=True,
                                     pin_memory=True)
            model.train()
            accum_loss = 0.0
            for i, batch in enumerate(data_loader):
                batch: Dict[str, torch.Tensor]
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                memory_streams_ids = [s.to(device) for s in batch['memory_streams_ids']]
                memory_padding_masks = [s.to(device) for s in batch['memory_padding_masks']]

                # Forward Pass
                use_amp = self.config.get('use_amp', False) and device.type == 'cuda'
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits, _ = model(input_ids=input_ids, memory_streams_ids=memory_streams_ids,
                                      memory_padding_masks=memory_padding_masks)
                    loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                    accum_loss += loss.item()
                    loss_for_backward = loss / self.config['GRADIENT_ACCUMULATION_STEPS']

                # Backward Pass & Gradient Accumulation
                scaler.scale(loss_for_backward).backward()

                if (i + 1) % self.config['GRADIENT_ACCUMULATION_STEPS'] == 0:
                    total_steps += 1
                    scaler.unscale_(optimizer)

                    # Gradient Clipping
                    if self.config['CLIP_THRESHOLD'] > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['CLIP_THRESHOLD'])
                    else:
                        grad_norm = torch.sqrt(
                            torch.sum(
                                torch.stack([p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None])))

                    scaler.step(optimizer)
                    scaler.update()

                    # Learning Rate Update
                    lr = get_learning_rate(total_steps, self.config)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    # Optimizer Step
                    optimizer.zero_grad()
                    grad_norm_val = grad_norm.item() if self.config['CLIP_THRESHOLD'] > 0 else 0.0
                    avg_loss = accum_loss / self.config['GRADIENT_ACCUMULATION_STEPS']
                    print(
                        f"Epoch {epoch + 1} | Step {total_steps: >6} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Grad Norm: {grad_norm_val:.4f}")
                    accum_loss = 0.0

                    # Logging & Validation
                    if total_steps % self.config['LOG_EVERY_N_STEPS'] == 0:
                        log_data = [total_steps, epoch + 1, f"{avg_loss:.4f}", "N/A", "N/A", f"{lr:.2e}",
                                    f"{grad_norm_val:.4f}"]
                        log_writer.writerow(log_data)
                        log_file.flush()
                    if total_steps % self.config['EVAL_EVERY_N_STEPS'] == 0:
                        val_loss, val_ppl = calculate_validation_loss(model=model, val_loader=val_data_loader,
                                                                      criterion=criterion, device=device)
                        print(
                            f"VALIDATION @ Step {total_steps: >6} | Val Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
                        log_data = [total_steps, epoch + 1, "N/A", f"{val_loss:.4f}", f"{val_ppl:.2f}", f"{lr:.2e}",
                                    "N/A"]
                        log_writer.writerow(log_data)
                        log_file.flush()

                        # Save a "best" model checkpoint
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            print(f"🎉 New best validation loss! Saving best model to {model_path}... 🎉")
                            save_checkpoint(model, optimizer, self.config, total_steps, best_val_loss, epoch,
                                            is_best=True)

                    if total_steps % self.config['SAVE_EVERY_N_STEPS'] == 0:
                        # Saving Checkpoint
                        save_checkpoint(model, optimizer, self.config, total_steps, best_val_loss, epoch, is_best=False)

        log_file.close()
        print("\nTraining Finished")
