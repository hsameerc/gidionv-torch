import argparse
import json

from src.config.config import get_config
from src.trainer.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the PyTorch Multi Memory Transformer.")
    parser.add_argument('--config', default='configs/test-config.json', type=str)
    parser.add_argument('--type', default='finetune', type=str)
    parser.add_argument('--source', default='online', type=str)
    args = parser.parse_args()

    cfg = get_config()
    if args.config:
        with open(args.config, 'r') as f:
            cfg.update(json.load(f))
    cfg.update({"TRAINING_TYPE": args.type})
    cfg.update({"TRAINING_SOURCE": args.source})
    trainer = Trainer(config=cfg)
    trainer.train()
