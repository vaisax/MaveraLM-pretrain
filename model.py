import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_rope_params(head_dim, theta_base=10_000, context_length=16384, dtype=torch.float32):
    assert head_dim % 2 == 0, "Head dimension must be even"
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[:head_dim // 2].float() / head_dim))
    angles = torch.arange(context_length, dtype=dtype)[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    x1, x2 = x[..., :head_dim // 2], x[..., head_dim // 2:]
    cos, sin = cos[:seq_len].unsqueeze(0).unsqueeze(0), sin[:seq_len].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos + rotated * sin).to(x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        out = x_f * torch.rsqrt(var + self.eps) * (1.0 + self.scale.float())
        return (out + self.shift.float()).to(x.dtype) if self.shift is not None else out.to(x.dtype)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.head_dim = head_dim or d_in // num_heads
        self.d_out = num_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.W_key(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        queries = queries * self.scaling
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))

class TransformerBlock(nn.Module):
    def __init__(self, cfg, use_sliding_attention=True):
        super().__init__()
        self.att = GroupedQueryAttention(
            cfg["emb_dim"], 
            cfg["n_heads"], 
            cfg["n_kv_groups"], 
            cfg["head_dim"], 
            cfg["qk_norm"], 
            cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm3 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm4 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.use_sliding_attention = use_sliding_attention

    def forward(self, x, mask_global, mask_local, cos, sin):
        mask = mask_local if self.use_sliding_attention else mask_global
        x = x + self.norm2(self.att(self.norm1(x), mask, cos, sin))
        x = x + self.norm4(self.ff(self.norm3(x)))
        return x

# Original 4B Model (keeping for backward compatibility)
class Nordic4BModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.blocks = nn.ModuleList()
        for i in range(cfg["n_layers"]):
            use_sliding = i < (cfg["n_layers"] - 4)
            self.blocks.append(TransformerBlock(cfg, use_sliding_attention=use_sliding))
        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        self.out_head.weight = self.tok_emb.weight
        cos, sin = compute_rope_params(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _create_masks(self, seq_len, device):
        ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        mask_global = torch.triu(ones, diagonal=1)
        mask_local = mask_global.clone()
        sliding_window = self.cfg.get("sliding_window", 4096)
        for i in range(seq_len):
            start_pos = max(0, i - sliding_window)
            if start_pos > 0:
                mask_local[i, :start_pos] = True
        return mask_global, mask_local
    
    def forward(self, input_ids, targets=None):
        b, seq_len = input_ids.shape
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)
        mask_global, mask_local = self._create_masks(seq_len, x.device)
        for block in self.blocks:
            x = block(x, mask_global, mask_local, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["context_length"]:] if idx.size(1) > self.cfg["context_length"] else idx
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx


# New 1B Model optimized for efficiency
class Nordic1BModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        
        # Create transformer blocks with optimized sliding attention pattern
        self.blocks = nn.ModuleList()
        for i in range(cfg["n_layers"]):
            # Use sliding attention for first 2/3 of layers, global for last 1/3
            use_sliding = i < (cfg["n_layers"] * 2 // 3)
            self.blocks.append(TransformerBlock(cfg, use_sliding_attention=use_sliding))
        
        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        
        # Weight tying for efficiency
        self.out_head.weight = self.tok_emb.weight
        
        # Precompute RoPE parameters
        cos, sin = compute_rope_params(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use smaller std for 1B model for better stability
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _create_masks(self, seq_len, device):
        """Create causal and sliding window masks"""
        ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        
        # Global causal mask (upper triangular)
        mask_global = torch.triu(ones, diagonal=1)
        
        # Local sliding window mask
        mask_local = mask_global.clone()
        sliding_window = self.cfg.get("sliding_window", seq_len // 2)
        
        for i in range(seq_len):
            start_pos = max(0, i - sliding_window)
            if start_pos > 0:
                mask_local[i, :start_pos] = True
        
        return mask_global, mask_local
    
    def forward(self, input_ids, targets=None):
        b, seq_len = input_ids.shape
        
        # Embedding with scaling
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)
        
        # Create attention masks
        mask_global, mask_local = self._create_masks(seq_len, x.device)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask_global, mask_local, self.cos, self.sin)
        
        # Final layer norm and output projection
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Generate text using the model"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop context if too long
            idx_cond = idx[:, -self.cfg["context_length"]:] if idx.size(1) > self.cfg["context_length"] else idx
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")
            
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        self.train()
        return idx