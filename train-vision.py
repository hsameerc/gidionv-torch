import argparse
import json
from functools import partial

import torch
from torch.utils.data import DataLoader

from src.config.config import get_config
from src.lib.core.hf_tokenizer_wrapper import HFTokenizerWrapper
from src.loaders.vision_loader import VisionLanguageDataset, vision_language_collate_fn


def test(config, tokenizer, device):
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    train_dataset = VisionLanguageDataset(annotations_path=config['TRAIN_FILE_PATH'],
        image_dir=config['IMAGE_DIR'], tokenizer=tokenizer, image_size=config['vision_encoder']['image_size'])
    collate_fn = partial(vision_language_collate_fn, pad_id=pad_id, bos_id=bos_id, config=config)

    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    # 4. Use it in your training loop
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        print(batch)
        exit()

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = HFTokenizerWrapper(config['TOKENIZER_PATH'])
    test(config, tokenizer, device)
    print(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the PyTorch Multi Memory Transformer.")
    parser.add_argument('--config', default='configs/test-vision.json', type=str)
    args = parser.parse_args()

    cfg = get_config()
    if args.config:
        with open(args.config, 'r') as f:
            cfg.update(json.load(f))

    train(cfg)
