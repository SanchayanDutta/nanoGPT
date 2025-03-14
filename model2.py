import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

#
# Example aggregator that uses a spectral DPP approach
#

def dpp_sample_spectral(L: torch.Tensor):
    """
    Performs a spectral DPP sampling from a kernel matrix L,
    but first casts L to float32 so we can call torch.linalg.eigh without error.
    Returns a subset (list of indices).
    """
    # Upcast to float32 to avoid the "linalg_eigh_cuda not implemented for bfloat16" error
    L = L.float()

    # Compute eigenvalues/eigenvectors of L
    w, V = torch.linalg.eigh(L)  # shape [N], [N,N]
    # Convert w to probabilities. In a typical spectral DPP we sample based on these eigenvalues.
    # This is just an example, adjust to your actual logic.
    w = w.clamp(min=0)  # ensure nonnegative
    # We'll pick an eigenvector if we sample from Bernoulli(w_i / (1 + w_i)).
    selected_eigs = []
    for i in range(len(w)):
        if torch.rand(1).item() < w[i] / (1 + w[i]):
            selected_eigs.append(i)

    # Then from the selected eigenvectors, we pick a random subset of items (exact method can vary)
    # For demonstration, let's just pick the argmax of sum of squares for each row
    if len(selected_eigs) == 0:
        return []
    V_sel = V[:, selected_eigs]  # shape [N, k]
    # Then typically a Gram-Schmidt or something to pick actual item subset from these vectors
    # We'll do a trivial approach: pick the max row in norm each time
    row_norms = V_sel.pow(2).sum(dim=1)
    chosen_idx = torch.argsort(row_norms, descending=True)
    # Suppose we only pick the top 3 for demonstration
    result = chosen_idx[:3].tolist()
    return result

def aggregator_for_token_i_unfixed(q_i: torch.Tensor, k: torch.Tensor, v: torch.Tensor, i: int) -> torch.Tensor:
    """
    Demonstrates how we might compute a spectral DPP subset for the current token i
    and then attend to that subset only.
    q_i: shape [d]
    k, v: shape [T, d]
    i: index of the current token
    """
    # Example: build an NxN kernel matrix L among all T keys (toy example).
    # In practice, you'd want a real kernel function of the keys, etc.
    # We'll do a naive dot-product matrix just to illustrate.
    T, d = k.shape
    L = k @ k.T  # shape [T, T], might want to scale or shift

    # Now we do a spectral DPP sample
    subset = dpp_sample_spectral(L)

    # We always want to ensure i is in the subset for "causal" or direct reference
    if i not in subset:
        subset.append(i)

    # Actually attend to only that subset with a standard dot-product attention
    subset_k = k[subset]  # shape [subset_size, d]
    subset_v = v[subset]
    # Compute attention weights from q_i
    att_scores = (q_i.unsqueeze(0) * subset_k).sum(dim=1) / math.sqrt(d)
    att_weights = F.softmax(att_scores, dim=0)  # shape [subset_size]
    out = (att_weights.unsqueeze(1) * subset_v).sum(dim=0)
    return out

#
# Minimal GPT components, with a custom attention block that calls aggregator_for_token_i_unfixed
#

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, 1e-5)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    # Toggle using aggregator_for_token_i_unfixed or standard attention
    use_dpp_aggregator: bool = False

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1,1,config.block_size,config.block_size)
            )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # [B, nh, T, hs]
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        if not self.config.use_dpp_aggregator:
            # Standard attention
            if self.flash:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True
                )
            else:
                att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v
        else:
            # We do a per-token aggregator approach (toy version).
            y = torch.zeros_like(q)
            for b_idx in range(B):
                for head_idx in range(self.n_head):
                    q_bh = q[b_idx, head_idx]  # shape [T, hs]
                    k_bh = k[b_idx, head_idx]
                    v_bh = v[b_idx, head_idx]
                    y_bh = []
                    for i in range(T):
                        out_i = aggregator_for_token_i_unfixed(q_bh[i], k_bh, v_bh, i)
                        y_bh.append(out_i.unsqueeze(0))
                    y_bh = torch.cat(y_bh, dim=0)  # shape [T, hs]
                    y[b_idx, head_idx] = y_bh

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": LayerNorm(config.n_embd, bias=config.bias)
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2*config.n_layer))

        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.shape
        assert t <= self.config.block_size, "Cannot forward, sequence too long"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer["wte"](idx)
        pos_emb = self.transformer["wpe"](pos)
        x = self.transformer["drop"](tok_emb + pos_emb)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)

        if targets is not None:
            logits = self.lm_head(x)  # [b, t, vocab_size]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        import inspect
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and (device_type == "cuda")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # just a placeholder for the standard nanoGPT method
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu
