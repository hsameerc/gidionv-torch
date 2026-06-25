import argparse
import json
import os
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.config import get_config
from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.lib.transformer.multi_memory_avt_transformer import MultiModalMemoryTransformer
from src.loaders.vision_loader import VisionLanguageDataset, vision_language_collate_fn
from src.utils.trainerhelper import get_learning_rate


class TrainVision:

    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        model_dir = config['MODEL_DIR'] if False else ""
        model_name = config['MODEL_NAME']
        checkpoint_path = os.path.join(model_dir, f"{model_name}_latest.pth")
        best_checkpoint_path = os.path.join(model_dir, f"{model_name}_best.pth")

    def test(self, model, config, tokenizer,optimizer, device):
        # CrossEntropyLoss
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        pad_id = tokenizer.pad_token_id
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        start_epoch = 0
        use_amp = config.get('use_amp', False) and device.type == 'cuda'
        scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
        for epoch in range(start_epoch, config['EPOCHS']):
            print(f"\n{'=' * 25} Epoch {epoch + 1}/{config['EPOCHS']} {'=' * 25}")

            train_dataset = VisionLanguageDataset(annotations_path=config['TRAIN_FILE_PATH'],
                                                  image_dir=self.config['IMAGE_DIR'], tokenizer=tokenizer,
                                                  image_size=self.config['vision_encoder']['image_size'])

            collate_fn = partial(vision_language_collate_fn, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id,
                                 config=self.config)

            train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4,
                                      collate_fn=collate_fn)
            model.train()
            accum_loss = 0.0
            total_steps = 0

            for i, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                image_ids = batch['image_input'].to(self.device)
                use_amp = config.get('use_amp', False) and device.type == 'cuda'
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits, _ = model(input_ids=input_ids, image_input=image_ids)
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

    def train(self):
        print(f"Using device: {self.device}")
        tokenizer = HFTokenizerWrapper(self.config['TOKENIZER_PATH'])
        # Initializing empty model and optimizer
        print("Initializing model architecture...")
        tokenizer = HFTokenizerWrapper(self.config['TOKENIZER_PATH'])
        # model = MultiMemoryTransformer(config, tokenizer).to(device)
        model = MultiModalMemoryTransformer(self.config, tokenizer).to(self.device)
        model.apply(model.init_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['PEAK_LEARNING_RATE'],
                                      betas=(self.config['ADAM_BETA1'], self.config['ADAM_BETA2']),
                                      weight_decay=self.config['WEIGHT_DECAY'])

        self.test(model, self.config, tokenizer, optimizer, self.device)
        print(self.config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the PyTorch Multi Memory Transformer.")
    parser.add_argument('--config', default='configs/test-vision.json', type=str)
    args = parser.parse_args()

    cfg = get_config()
    if args.config:
        with open(args.config, 'r') as f:
            cfg.update(json.load(f))

    train_vison = TrainVision(cfg)
    train_vison.train()
