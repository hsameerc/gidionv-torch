from typing import Dict, Any, Optional


def get_config_v5(base_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Defines the complete configuration for training the augmented model."""
    config_default = {
        "TRAINING_TYPE": "pretraining",
        "MODEL_NAME": "gideon_v4_model",
        "MODEL_DIR": "checkpoints/gideon_v4",
        "LOG_FILE_PATH": "logs/gideon_v4_train.csv",
        "TOKENIZER_PATH": "tokenizers/bpe_50K",
        "TRAIN_FILE_PATH": "data/train.txt",
        "VAL_FILE_PATH": "data/val.txt",
        "CHUNK_SIZE_CHARS": 50000000,
        "OVERLAP_LEN_TOKENS": 64,
        "text_memory_encoder": {
            "num_layers": 4,
            "num_heads": 12,
            "ff_hidden_config": [
                2048
            ],
            "num_memory_slots": 2
        },
        "AUDIO_DIR": "data/audio_files/",
        "audio_encoder": {
            "sample_rate": 16000,
            "num_freq_bins": 80,
            "max_audio_len": 3000,
            "num_layers": 8,
            "num_heads": 8,
            "ff_hidden_config": [
                2048
            ]
        },
        "IMAGE_DIR": "data/images/",
        "vision_encoder": {
            "image_size": 224,
            "patch_size": 16,
            "in_channels": 3,
            "num_layers": 12,
            "num_heads": 12,
            "ff_hidden_config": [
                3072
            ],
        },
        "RANDOM_SEED": 42,
        "resume_training": False,
        "EPOCHS": 5,
        "BATCH_SIZE": 8,
        "GRADIENT_ACCUMULATION_STEPS": 8,
        "CLIP_THRESHOLD": 1.0,
        "PEAK_LEARNING_RATE": 1e-4,
        "MIN_LEARNING_RATE": 1e-5,
        "WARMUP_STEPS": 2000,
        "TOTAL_DECAY_STEPS": 100000,
        "ADAM_BETA1": 0.9,
        "ADAM_BETA2": 0.95,
        "EVAL_EVERY_N_STEPS": 1000,
        "SAVE_EVERY_N_STEPS": 1000,
        "LOG_EVERY_N_STEPS": 100,
        "model_dtype": "float32",
        "d_model": 768,
        "max_seq_len": 512,
        "dropout_rate": 0.1,
        "model": {
            "num_memory_streams": 3
        },
        "decoder": {
            "num_layers": 12,
            "num_heads": 12,
            "ff_hidden_config": [
                3072
            ]
        },
        'ENABLE_GRL': True,
        'GRL_EVERY_N_OPTIMIZER_STEPS': 2000,
        'GRL_PARAMS': {
            'alpha': 0.001,
            'reinforce_baseline_alpha': 0.95,
            'gen_batch_size': 4,
            'gen_max_len': 64,
            'prompts': ["In a startling finding,", "The purpose of this project is to"],
            'sampling_temp': 0.9,
            'stop_words': None,
            'top_k': 50,
            'top_p': 0,
        },
    }
    config_default.update(base_config)
    return config_default


def get_config() -> Dict[str, Any]:
    """Defines the complete configuration for training the augmented model."""
    return {
        'MODEL_NAME': 'gidion_aug_v1.0',
        'MODEL_DIR': 'models/gidion_augmented_v1.0',
        'TOKENIZER_PATH': 'tokenizers/bpe_50k.json',
        'TRAIN_FILE_PATH': 'data/train.jsonl',
        'VAL_FILE_PATH': 'data/val.jsonl',
        'MEMORY_PATH': 'memory',
        'LOG_FILE_PATH': 'logs/training_log_v1.0.csv',
        'GRADIENT_NORM_LOG_PATH': 'logs/gradient_norms_v1.0.csv',
        'NUM_WORKERS': 1,
        'd_model': 512,
        'vocab_size': 50257,
        'max_seq_len': 512,
        'pad_id': 0,
        'dropout_rate': 0.1,
        'memory_encoder':
            {
                'num_layers': 4,
                'num_heads': 8,
                'ff_hidden_config': [2048],
                'num_memory_slots': 8
            },
        'decoder': {
            'num_layers': 8,
            'num_heads': 8,
            'ff_hidden_config': [2048]
        },
        'EPOCHS': 5,
        'BATCH_SIZE': 4,
        'CHUNK_SIZE_CHARS': 50000000,
        'OVERLAP_LEN_TOKENS': 64,
        'GRADIENT_ACCUMULATION_STEPS': 16,
        'PEAK_LEARNING_RATE': 3e-4,
        'MIN_LEARNING_RATE': 1e-5,
        'WARMUP_STEPS': 4000,
        'TOTAL_DECAY_STEPS': 250000,
        'CLIP_THRESHOLD': 1.0,
        'ADAM_BETA1': 0.9,
        'ADAM_BETA2': 0.95,
        'EVAL_EVERY_N_STEPS': 1000,
        'SAVE_EVERY_N_STEPS': 200,
        'LOG_EVERY_N_STEPS': 100,
        'ENABLE_GRL': True,
        'GRL_EVERY_N_OPTIMIZER_STEPS': 2000,
        'GRL_PARAMS': {
            'alpha': 0.001,
            'reinforce_baseline_alpha': 0.95,
            'gen_batch_size': 4,
            'gen_max_len': 64,
            'prompts': ["In a startling finding,", "The purpose of this project is to"],
            'sampling_temp': 0.9,
            'stop_words': None,
            'top_k': 50,
            'top_p': 0,
        },
        'RANDOM_SEED': 42,
        'model_dtype': 'float32',
        'resume_training': True,
        'total_steps': 0,
        'last_trained_epoch': 0,
        'special_tokens': {"USER": "<USER>", "ASSISTANT": "<ASSISTANT>", "INST": "<INST>", "END_INST": "</INST>"}
    }
