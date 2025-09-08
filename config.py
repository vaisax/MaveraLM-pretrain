import torch

NORDIC_4B_CONFIG = {
    "vocab_size": 50257,
    "context_length": 12000,
    "emb_dim": 2816,
    "n_heads": 22,
    "n_layers": 32,
    "hidden_dim": 11264,
    "head_dim": 128,
    "n_kv_groups": 11,
    "qk_norm": True,
    "rope_base": 1000000.0,
    "sliding_window": 2816,
    "dtype": torch.bfloat16,
}