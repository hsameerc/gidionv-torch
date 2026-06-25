import os
import torch
import json
from src.lib.transformer.multi_memory_transformer import MultiMemoryTransformer
from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper

def main():
    # Load fine-tune config
    config_path = 'configs/finetune_v2.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Re-create model and tokenizer structure
    tokenizer = HFTokenizerWrapper(config['TOKENIZER_PATH'])
    device = torch.device('cpu')
    model = MultiMemoryTransformer(config, tokenizer).to(device)

    # Load pre-trained state dict
    pretrain_best_path = 'research/models/gidionv_pretrain_v2/gidionv_pretrain_v2_best.pth'
    pretrain_latest_path = 'research/models/gidionv_pretrain_v2/gidionv_pretrain_v2_latest.pth'
    
    if os.path.exists(pretrain_best_path):
        target_path = pretrain_best_path
    elif os.path.exists(pretrain_latest_path):
        target_path = pretrain_latest_path
    else:
        raise FileNotFoundError(f"Could not find pre-trained weights in research/models/gidionv_pretrain_v2/")
    
    print(f"Loading weights from {target_path}")
    pretrained_state = torch.load(target_path, map_location='cpu')

    # Unpack if it's a full checkpoint
    if 'model_state_dict' in pretrained_state:
        pretrained_state = pretrained_state['model_state_dict']

    # Remove the pre-computed PE buffers to avoid size mismatch (they are sinusoidal, not learnable)
    pretrained_state.pop('positional_encoding.pe', None)
    pretrained_state.pop('positional_encoding.div_term', None)

    # Load the weights into the model
    model.load_state_dict(pretrained_state, strict=False)
    print("Pre-trained weights loaded successfully.")

    # Create the optimizer with fine-tuning parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['PEAK_LEARNING_RATE'],
        betas=(config['ADAM_BETA1'], config['ADAM_BETA2']),
        weight_decay=config['WEIGHT_DECAY']
    )

    # Prepare checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'total_steps': 0,
        'best_val_loss': float('inf'),
        'last_trained_epoch': 0
    }

    # Save checkpoint to the fine-tuning directory
    model_dir = config['MODEL_DIR']
    model_name = config['MODEL_NAME']
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"{model_name}_latest.pth")
    torch.save(checkpoint, save_path)
    print(f"Fine-tuning latest checkpoint successfully created at: {save_path}")
    print("You can now run: python train.py --config configs/finetune_v2.json --source online")

if __name__ == '__main__':
    main()
