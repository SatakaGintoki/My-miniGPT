
MODEL_CONFIG = {
    "vocab_size": 10000,
    "context_length": 256,
    "num_layers": 4,
    "d_model": 512,
    "num_heads": 16,
    "d_ff": 1344,
    "rope_theta": 10000.0,
}

TOKENIZER_CONFIG = {
    "vocab_size": 10000,
    "special_tokens": ["<|endoftext|>"],
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "max_steps": 170000,
    "lr": 3e-4,
    "max_lr": 3e-4,
    "min_lr": 3e-5,
    "warmup_steps": 2000,
    "grad_clip_val": 1.0,

    "betas": (0.9, 0.95),
    "weight_decay": 0.1,
    "eps": 1e-8,
}

PATHS = {
    "train_tokens": "data/train_tokens.npy",
    "vocab": "data/vocab.pkl",
    "merges": "data/merges.pkl",
    "checkpoint": "checkpoint.pt",
    "corpus": "data/TinyStoriesV2-GPT4-train.txt",
}
