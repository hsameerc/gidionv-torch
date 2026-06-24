import datasets  # Import datasets first to avoid Windows CUDA DLL collision with pyarrow

# Apply Python 3.14 compatibility monkeypatch for datasets + dill
try:
    import datasets.utils._dill
    import dill
    def patched_batch_setitems(self, items, obj=None):
        if self._legacy_no_dict_keys_sorting:
            try:
                return super(datasets.utils._dill.Pickler, self)._batch_setitems(items, obj)
            except TypeError:
                return super(datasets.utils._dill.Pickler, self)._batch_setitems(items)
        try:
            items = sorted(items)
        except Exception:
            from datasets.fingerprint import Hasher
            items = sorted(items, key=lambda x: Hasher.hash(x[0]))
        try:
            dill.Pickler._batch_setitems(self, items, obj)
        except TypeError:
            dill.Pickler._batch_setitems(self, items)
    datasets.utils._dill.Pickler._batch_setitems = patched_batch_setitems
except Exception as e:
    print(f"Warning: Failed to apply Python 3.14 datasets patch: {e}")

import argparse
import json

from src.config.config import get_config
from src.trainer.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the PyTorch Multi Memory Transformer.")
    parser.add_argument('--config', default='configs/gidionv_multi_memory.json', type=str)
    parser.add_argument('--type', default='pretrain', type=str)
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
