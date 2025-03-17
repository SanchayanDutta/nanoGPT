import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

#
# Helper functions for the DPP-based approach
#

def dpp_det_score(vecs):
    """
    Computes determinant of (vecs @ vecs.T). Cast to float32 to avoid precision errors.
    """
    vecs_32 = vecs.float()
    gram = vecs_32 @ vecs_32.T
    return torch.linalg.det(gram)

def attention_for_subset(q_i, k_subset, v_subset):
    """
    Standard dot-product attention over a subset of keys. Returns weighted sum of v_subset.
    """
    head_size = q_i.shape[0]
    logits = (q_i.unsqueeze(0) * k_subset).sum(dim=1) / (head_size**0.5)  # [subset_size]
    w = F.softmax(logits, dim=0)  # [subset_size]
    out = (w.unsqueeze(1) * v_subset).sum(dim=0)  # [head_size]
    return out

def dpp_objective_score(E, minimize_det, penalty_alpha):
    """
    DPP objective:
      if minimize_det = True:  score = - log(det(E E^T) + epsilon) / (|S|^penalty_alpha)
      otherwise:               score =   log(det(E E^T) + epsilon) / (|S|^penalty_alpha)
    """
    epsilon = 1e-6
    det_val = dpp_det_score(E)
    log_det = torch.log(det_val + epsilon)
    subset_size = E.size(0)
    if minimize_det:
        return -log_det / (subset_size ** penalty_alpha)
    else:
        return  log_det / (subset_size ** penalty_alpha)

def dpp_greedy_attention(q, k, v, causal, min_size, max_size, top_m,
                         temperature, minimize_det, penalty_alpha):
    """
    Single-head, single-example DPP-based attention using greedy selection.
    q, k, v: shape [T, head_size]
    causal: if True, token i attends only to j <= i
    min_size, max_size: subset sizes must be in [min_size..max_size], always including i
    top_m: only consider top-M neighbors by dot product
    temperature: not used in this greedy approach, but left for code compatibility
    minimize_det: True => negative log-det
    penalty_alpha: exponent that penalizes large subsets
    Returns: [T, head_size] attention output
    """
    T, head_size = q.shape
    out_all = torch.zeros_like(q)

    for i in range(T):
        # 1) Mask for causal
        if causal:
            valid_j = torch.arange(0, i+1, device=q.device)
        else:
            valid_j = torch.arange(0, T, device=q.device)

        # 2) Pick top-M by dot product similarity
        q_i = q[i]
        sim = (q_i.unsqueeze(0) * k[valid_j]).sum(dim=1)  # shape [valid_j_size]
        n_candidate = min(top_m, len(valid_j))
        _, topidx_local = torch.topk(sim, k=n_candidate)
        topidx = valid_j[topidx_local]

        # Ensure i itself is included in the candidate set
        if i not in topidx:
            topidx = torch.cat([topidx, i.unsqueeze(0)])

        # 3) Greedy selection: always start with i
        S = [i]
        current_score = dpp_objective_score(k[S], minimize_det, penalty_alpha)

        # Convert to set for iteration
        candidates = set(topidx.tolist())
        if i in candidates:
            candidates.remove(i)

        while len(S) < max_size:
            best_candidate = None
            best_candidate_score = None

            # Evaluate each candidate
            for c in candidates:
                newS = S + [c]
                new_score = dpp_objective_score(k[newS], minimize_det, penalty_alpha)

                if best_candidate_score is None:
                    best_candidate = c
                    best_candidate_score = new_score
                else:
                    if not minimize_det and new_score > best_candidate_score:
                        best_candidate = c
                        best_candidate_score = new_score
                    elif minimize_det and new_score < best_candidate_score:
                        best_candidate = c
                        best_candidate_score = new_score

            if best_candidate is None:
                break

            # Check if the best candidate improves the score
            if minimize_det:
                improvement_found = (best_candidate_score < current_score)
            else:
                improvement_found = (best_candidate_score > current_score)

            # Accept if it improves or we haven't reached min_size
            if improvement_found or (len(S) < min_size):
                S.append(best_candidate)
                candidates.remove(best_candidate)
                current_score = best_candidate_score
            else:
                break

        # 4) Standard attention over chosen subset
        out_s = attention_for_subset(q_i, k[S], v[S])
        out_all[i] = out_s

    return out_all

#
# GPT with gating between standard attention and DPP-based subset selection
#

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    # DPP options
    use_dpp_attention: bool = False
    dpp_min_size: int = 2
    dpp_max_size: int = 8
    dpp_top_m: int = 8
    dpp_temperature: float = 0.1
    dpp_minimize_det: bool = True         # <-- Default: True (capturing similarity)
    dpp_penalty_alpha: float = 1.0        # Tune as needed

    # Gating
    dpp_gate_init: float = 0.5           # initial value for gate
    learn_dpp_gate: bool = True          # make the gate learnable

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # If PyTorch >= 2.0, we can use scaled_dot_product_attention
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            # For fallback manual masking
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

        # Gate to blend standard attention with DPP-based attention
        self.gate_param = nn.Parameter(torch.tensor(config.dpp_gate_init),
                                       requires_grad=config.learn_dpp_gate)

    def forward(self, x):
        B, T, C = x.size()

        # Project q, k, v
        qkv = self.c_attn(x)  # shape [B, T, 3*C]
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # [B, n_head, T, head_size]
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        if not self.config.use_dpp_attention:
            # Standard attention only
            if self.flash:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True
                )
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v
        else:
            # 1) Standard attention
            if self.flash:
                y_std = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True
                )
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y_std = att @ v

            # 2) DPP-based subset attention (greedy)
            y_dpp = torch.zeros_like(q)
            for b_idx in range(B):
                for h_idx in range(self.n_head):
                    qbh = q[b_idx, h_idx]  # [T, head_size]
                    kbh = k[b_idx, h_idx]
                    vbh = v[b_idx, h_idx]
                    out_bh = dpp_greedy_attention(
                        qbh, kbh, vbh,
                        causal=True,  # typically True for causal LM
                        min_size=self.config.dpp_min_size,
                        max_size=self.config.dpp_max_size,
                        top_m=self.config.dpp_top_m,
                        temperature=self.config.dpp_temperature,
                        minimize_det=self.config.dpp_minimize_det,
                        penalty_alpha=self.config.dpp_penalty_alpha
                    )
                    y_dpp[b_idx, h_idx] = out_bh

            # 3) Combine via learnable gate
            gate = torch.sigmoid(self.gate_param)  # scalar in [0, 1]
            y = gate * y_dpp + (1 - gate) * y_std

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
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
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie word embedding and LM head
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        # GPT-2 residual projection scaling
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, "Sequence length exceeds block_size."
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # For generation, only the last token's logits
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # Example utility method to load GPT-2 from transformers
        from transformers import GPT2LMHeadModel
        override_args = override_args or {}
        config_map = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }
        if model_type not in config_map:
            raise ValueError("Unknown model_type. Must be one of: gpt2, gpt2-medium, gpt2-large, gpt2-xl.")
        config_args = config_map[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True

        # Allow override of certain args
        for k, v in (override_args or {}).items():
            config_args[k] = v

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = [
            'attn.c_attn.weight', 'attn.c_proj.weight',
            'mlp.c_fc.weight',    'mlp.c_proj.weight'
        ]

        if len(sd_keys_hf) != len(sd_keys):
            raise ValueError(f"Mismatched state dict keys: {len(sd_keys_hf)} != {len(sd_keys)}")

        # Load weights, transpose where necessary
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

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

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        flops_promised = 312e12  # A100 GPU bfloat16 peak FLOPS
        return flops_achieved / flops_promised
