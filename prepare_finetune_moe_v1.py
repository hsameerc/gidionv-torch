"""
prepare_finetune_moe_v1.py
===========================
Copies the best pretrained MoE checkpoint into the fine-tune model directory
as the starting point for instruction fine-tuning.

Run this ONCE after pre-training has plateaued, before starting fine-tuning:
    python prepare_finetune_moe_v1.py
"""

import os
import shutil
import torch

PRETRAIN_DIR  = "research/models/gidion_moe_v1"
PRETRAIN_NAME = "gidion_moe_v1"

FINETUNE_DIR  = "research/models/gidion_moe_v1_finetune"
FINETUNE_NAME = "gidion_moe_v1_finetune"

def main():
    os.makedirs(FINETUNE_DIR, exist_ok=True)

    # Prefer the best checkpoint; fall back to latest
    src_path = os.path.join(PRETRAIN_DIR, f"{PRETRAIN_NAME}_best_resume.pth")
    if not os.path.exists(src_path):
        src_path = os.path.join(PRETRAIN_DIR, f"{PRETRAIN_NAME}_latest.pth")
    
    if not os.path.exists(src_path):
        print(f"[ERROR] No checkpoint found at {src_path}")
        print("Make sure pre-training has started and saved at least one checkpoint.")
        return

    dst_path = os.path.join(FINETUNE_DIR, f"{FINETUNE_NAME}_latest.pth")
    print(f"Source:      {src_path}")
    print(f"Destination: {dst_path}")

    # Load and patch the checkpoint metadata so the trainer starts fresh
    checkpoint = torch.load(src_path, map_location='cpu')
    checkpoint['total_steps']        = 0
    checkpoint['best_val_loss']      = float('inf')
    checkpoint['last_trained_epoch'] = 0

    # Clear the optimizer state so it re-initializes cleanly for the new LR
    if 'optimizer_state_dict' in checkpoint:
        checkpoint['optimizer_state_dict'] = None

    torch.save(checkpoint, dst_path)
    print(f"\n[SUCCESS] Fine-tune checkpoint ready at: {dst_path}")
    print("You can now run:")
    print("  python train.py --config configs/finetune_moe_v1.json --source online --type finetune")


if __name__ == '__main__':
    main()
