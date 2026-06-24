import csv
import math
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
        The main, training script.
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
        num_mem_streams = self.config['model']['num_memory_streams']
        log_headers = ["step", "epoch", "loss", "val_loss", "perplexity", "lr", "grad_norm"]
        for i in range(num_mem_streams):
            log_headers.append(f"mem_weight_{i}")
        # Logging Setup
        log_file_path = os.path.join(self.config['MODEL_DIR'], self.config['LOG_FILE_NAME'])
        log_file = open(log_file_path, 'a', newline='', encoding='utf-8')
        log_writer = csv.DictWriter(log_file, fieldnames=log_headers)
        if log_file.tell() == 0:
            log_writer.writeheader()

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
            optimizer.zero_grad()
            accum_loss = 0.0
            accum_steps = 0
            for i, batch in enumerate(data_loader):
                accum_steps += 1
                batch: Dict[str, torch.Tensor]
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)

                # Unbind it along the `num_streams` dimension (dim=1).
                batched_memory_tensor = batch['memory_streams_ids'].to(device)
                unbound_streams = torch.unbind(batched_memory_tensor, dim=1)
                memory_streams_ids = list(unbound_streams)

                # Forward Pass
                use_amp = self.config.get('use_amp', False) and device.type == 'cuda'
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits, _, fusion_weights_per_layer = model(input_ids=input_ids,
                                                                memory_streams_ids=memory_streams_ids, )
                    loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                    accum_loss += loss.item()
                    loss_for_backward = loss / self.config['GRADIENT_ACCUMULATION_STEPS']

                # Backward Pass & Gradient Accumulation
                scaler.scale(loss_for_backward).backward()

                if accum_steps == self.config['GRADIENT_ACCUMULATION_STEPS']:
                    total_steps += 1
                    scaler.unscale_(optimizer)

                    # Gradient Clipping
                    if self.config['CLIP_THRESHOLD'] > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['CLIP_THRESHOLD'])
                    else:
                        grad_norm = torch.sqrt(
                            torch.sum(
                                torch.stack([p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None])))

                    grad_norm_val = grad_norm.item()
                    avg_loss = accum_loss / self.config['GRADIENT_ACCUMULATION_STEPS']

                    # Safety check for NaN/Inf in gradients or loss
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm) or math.isnan(avg_loss) or math.isinf(avg_loss):
                        print(f"[WARNING] Step {total_steps}: NaN or Inf detected in gradients (norm: {grad_norm_val:.4f}) or loss (loss: {avg_loss:.4f}). Skipping optimizer step.")
                        optimizer.zero_grad()
                        accum_loss = 0.0
                        accum_steps = 0

                        # Recovery: reload the latest clean checkpoint to restore model weights
                        checkpoint_path = os.path.join(self.config['MODEL_DIR'], f"{self.config['MODEL_NAME']}_latest.pth")
                        if os.path.exists(checkpoint_path):
                            print(f"[RECOVERY] Reloading model and optimizer state from {checkpoint_path}...")
                            checkpoint = torch.load(checkpoint_path, map_location=device)
                            model.load_state_dict(checkpoint['model_state_dict'])
                            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                            total_steps = checkpoint.get('total_steps', 0)
                            print(f"[RECOVERY] Successfully recovered model weights. Rolled back total_steps to {total_steps}.")
                        else:
                            print("[RECOVERY] No latest checkpoint found to reload! Continuing training from current state.")
                            total_steps -= 1
                        continue

                    scaler.step(optimizer)
                    scaler.update()

                    # Learning Rate Update
                    lr = get_learning_rate(total_steps, self.config)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    # Optimizer Step
                    optimizer.zero_grad()
                    print(
                        f"Epoch {epoch + 1} | Step {total_steps: >6} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Grad Norm: {grad_norm_val:.4f}")
                    accum_loss = 0.0
                    accum_steps = 0

                    # Logging & Validation
                    if total_steps % self.config['LOG_EVERY_N_STEPS'] == 0:
                        log_data = {
                            "step": total_steps,
                            "epoch": epoch + 1,
                            "loss": f"{avg_loss:.4f}",
                            "val_loss": "N/A",
                            "perplexity": "N/A",
                            "lr": f"{lr:.2e}",
                            "grad_norm": f"{grad_norm_val:.4f}"
                        }
                        if fusion_weights_per_layer:
                            stacked_weights = torch.stack(fusion_weights_per_layer)
                            avg_fusion_weights = stacked_weights.mean(dim=0)
                            avg_weights_list = avg_fusion_weights.cpu().numpy().tolist()
                            for mi, weight in enumerate(avg_weights_list):
                                log_data[f"mem_weight_{mi}"] = f"{weight:.4f}"
                        log_writer.writerow(log_data)
                        log_file.flush()
                    if total_steps % self.config['EVAL_EVERY_N_STEPS'] == 0:
                        val_loss, val_ppl = calculate_validation_loss(model=model, val_loader=val_data_loader,
                                                                      criterion=criterion, device=device)
                        print(f"VALIDATION @ Step {total_steps: >6} | Val Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
                        log_data = {
                            "step": total_steps,
                            "epoch": epoch + 1,
                            "loss": "N/A",
                            "val_loss": f"{val_loss:.4f}",
                            "perplexity": f"{val_ppl:.2f}",
                            "lr": f"{lr:.2e}",
                            "grad_norm": "N/A"
                        }
                        for msi in range(num_mem_streams):
                            log_data[f"mem_weight_{msi}"] = f"N/A"
                        log_writer.writerow(log_data)
                        log_file.flush()
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            print(f"[SUCCESS] New best validation loss! Saving best model to {model_path}...")
                            save_checkpoint(model, optimizer, self.config, total_steps, best_val_loss, epoch, is_best=True)

                    if total_steps % self.config['SAVE_EVERY_N_STEPS'] == 0:
                        save_checkpoint(model, optimizer, self.config, total_steps, best_val_loss, epoch, is_best=False)

            # Check for trailing batches
            if accum_steps > 0:
                total_steps += 1
                scaler.unscale_(optimizer)
                if self.config['CLIP_THRESHOLD'] > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['CLIP_THRESHOLD'])
                else:
                    grad_norm = torch.sqrt(torch.sum(torch.stack([p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None])))
                
                grad_norm_val = grad_norm.item()
                avg_loss = accum_loss / accum_steps  # Normalize by actual trailing steps

                # Safety check for NaN/Inf in gradients or loss (trailing)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm) or math.isnan(avg_loss) or math.isinf(avg_loss):
                    print(f"[WARNING] Step {total_steps} (Trailing): NaN or Inf detected in gradients or loss. Skipping optimizer step.")
                    optimizer.zero_grad()
                    accum_loss = 0.0

                    # Recovery: reload the latest clean checkpoint to restore model weights
                    checkpoint_path = os.path.join(self.config['MODEL_DIR'], f"{self.config['MODEL_NAME']}_latest.pth")
                    if os.path.exists(checkpoint_path):
                        print(f"[RECOVERY] Reloading model and optimizer state from {checkpoint_path}...")
                        checkpoint = torch.load(checkpoint_path, map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        total_steps = checkpoint.get('total_steps', 0)
                        print(f"[RECOVERY] Successfully recovered model weights. Rolled back total_steps to {total_steps}.")
                    else:
                        print("[RECOVERY] No latest checkpoint found to reload!")
                        total_steps -= 1
                else:
                    scaler.step(optimizer)
                    scaler.update()
                    lr = get_learning_rate(total_steps, self.config)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    optimizer.zero_grad()
                    print(f"Epoch {epoch + 1} | Step {total_steps: >6} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Grad Norm: {grad_norm_val:.4f} (Trailing)")
                    accum_loss = 0.0


        # Save final checkpoint at the end of training
        save_checkpoint(model, optimizer, self.config, total_steps, best_val_loss, epoch, is_best=False)

        log_file.close()
        print("\nTraining Finished")
