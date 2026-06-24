import datasets
import os
import torch
import json
from src.lib.transformer.multi_memory_transformer import MultiMemoryTransformer
from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper

def main():
    # Make sure we import datasets first to avoid Windows CUDA DLL load collision
    try:
        import datasets
    except ImportError:
        pass

    # Load scaled configuration
    config_path = 'configs/pretrain_scaled.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Re-create model and tokenizer structure for seq_len = 256
    tokenizer = HFTokenizerWrapper(config['TOKENIZER_PATH'])
    device = torch.device('cpu')
    model = MultiMemoryTransformer(config, tokenizer).to(device)

    # Load the best pre-trained checkpoint weights
    pretrain_best_path = 'research/models/gidionv_pretrain_demo/gidionv_pretrain_demo_best.pth'
    if not os.path.exists(pretrain_best_path):
        raise FileNotFoundError(f"Could not find pre-trained weights at {pretrain_best_path}")
    
    print(f"Loading weights from {pretrain_best_path}...")
    checkpoint_state = torch.load(pretrain_best_path, map_location='cpu')

    # Extract model state dict
    if 'model_state_dict' in checkpoint_state:
        state_dict = checkpoint_state['model_state_dict']
    else:
        state_dict = checkpoint_state

    # Remove the pre-computed PE buffers to avoid shape mismatch (max_len 64 vs 256)
    # The new PositionalEncoding layer will dynamically initialize a buffer of length 256.
    keys_to_pop = [
        'positional_encoding.pe',
        'positional_encoding.div_term',
        'memory_encoder.positional_encoding.pe',
        'memory_encoder.positional_encoding.div_term'
    ]
    for key in keys_to_pop:
        if key in state_dict:
            state_dict.pop(key)
            print(f"Popped positional encoding buffer: {key}")

    # Load the weights into the scaled model
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("State dict loaded successfully.")
    print(f"Missing keys (should only be PE buffers): {missing}")
    print(f"Unexpected keys: {unexpected}")

    # Create the optimizer with scaled-up learning rate parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['PEAK_LEARNING_RATE'],
        betas=(config['ADAM_BETA1'], config['ADAM_BETA2']),
        weight_decay=config['WEIGHT_DECAY']
    )

    # Prepare checkpoint dictionary for resuming training
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'total_steps': 0,
        'best_val_loss': float('inf'),
        'last_trained_epoch': 0
    }

    # Save checkpoint to the scaled pretraining directory
    model_dir = config['MODEL_DIR']
    model_name = config['MODEL_NAME']
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"{model_name}_latest.pth")
    torch.save(checkpoint, save_path)
    print(f"Scaled pre-training latest checkpoint successfully created at: {save_path}")

if __name__ == '__main__':
    main()
