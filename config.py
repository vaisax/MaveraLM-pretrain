import torch

NORDIC_1B_CONFIG = {
    "vocab_size": 50257,
    "context_length": 8192,
    "emb_dim": 2048,
    "n_heads": 16,
    "n_layers": 24,
    "hidden_dim": 5461,
    "head_dim": 128,
    "n_kv_groups": 8,
    "qk_norm": True,
    "rope_base": 1000000.0,
    "sliding_window": 2048,
    "dtype": torch.bfloat16,
}

NORDIC_1B_LARGE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 8192,
    "emb_dim": 2176,
    "n_heads": 17,
    "n_layers": 24,
    "hidden_dim": 5803,
    "head_dim": 128,
    "n_kv_groups": 8,
    "qk_norm": True,
    "rope_base": 1000000.0,
    "sliding_window": 2176,
    "dtype": torch.bfloat16,
}