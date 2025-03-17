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
    Computes det(vecs @ vecs.T). Cast to float32 to avoid errors when running in bf16/half.
    """
    vecs_32 = vecs.float()
    gram = vecs_32 @ vecs_32.T
    return torch.linalg.det(gram)

def attention_for_subset(q_i, k_subset, v_subset):
    """
    Standard dot-product attention from a single query q_i over only the keys in k_subset.
    Returns a weighted sum of v_subset.
    """
    head_size = q_i.shape[0]
    # Dot with the query
    dot = (q_i.unsqueeze(0) * k_subset).sum(dim=1) / (head_size**0.5)  # shape [subset_size]
    w = F.softmax(dot, dim=0)  # shape [subset_size]
    out = (w.unsqueeze(1) * v_subset).sum(dim=0)  # shape [head_size]
    return out

def dpp_objective_score(E, minimize_det, penalty_alpha):
    """
    Computes the DPP log-determinant based objective (or negative of it),
    with a size penalty: score = (+/-) log(det(E E^T)) / (|S|^penalty_alpha).
    """
    det_val = dpp_det_score(E)
    log_det = torch.log(det_val + 1e-6)
    subset_size = E.size(0)
    if minimize_det:
        # If we want to 'minimize' raw det, the objective is negative log(det).
        # Divided by subset_size^penalty_alpha
        score = -log_det / (subset_size ** penalty_alpha)
    else:
        # If we want to 'maximize' raw det, the objective is log(det).
        # Divided by subset_size^penalty_alpha
        score = log_det / (subset_size ** penalty_alpha)
    return score

def dpp_straight_through_attention(q, k, v, causal, min_size, max_size, top_m,
                                   temperature, minimize_det, penalty_alpha):
    """
    A single-head, single-example version of the discrete DPP-based attention that uses
    a greedy selection approach instead of brute-force enumeration.

    q, k, v: shape [T, head_size]
    causal: if True, token i can only attend to j <= i
    min_size, max_size: subset sizes must be in [min_size..max_size] (including i)
    top_m: only consider top-M neighbors by dot product
    temperature: (not used in greedy, but kept for signature compatibility)
    minimize_det: if True, we do negative log-det (favoring alignment)
    penalty_alpha: exponent for subset size penalty
    Returns: output of shape [T, head_size]
    """
    T, head_size = q.shape
    out_all = torch.zeros_like(q)

    for i in range(T):
        # 1) Determine valid j indices (causal)
        if causal:
            valid_j = torch.arange(0, i+1, device=q.device)
        else:
            valid_j = torch.arange(0, T, device=q.device)

        # 2) Top-M neighbors by dot product
        q_i = q[i]
        sim = (q_i.unsqueeze(0) * k[valid_j]).sum(dim=1)
        n_candidate = min(top_m, len(valid_j))
        _, topidx_local = torch.topk(sim, k=n_candidate)
        topidx = valid_j[topidx_local]

        # If topidx doesn't even contain i, ensure i is included
        # (the code below always includes i, but we also don't want to remove i from the "candidates")
        if i not in topidx:
            topidx = torch.cat([topidx, i.unsqueeze(0)])

        # 3) Greedy selection of subset S that always includes i
        # We'll define a local function to do that selection for the i-th token
        # Note: we only pick from topidx; that is the candidate pool.
        chosen_out = _greedy_select_and_attention(
            i, q_i, k, v, topidx, min_size, max_size, minimize_det, penalty_alpha
        )
        out_all[i] = chosen_out

    return out_all

def _greedy_select_and_attention(i, q_i, k, v, top_candidates, min_size, max_size,
                                 minimize_det, penalty_alpha):
    """
    Internal helper: picks a subset that always includes 'i' using a greedy approach
    and returns the standard attention result over that subset.
    """
    S = [i]  # Always include current token i
    current_score = dpp_objective_score(k[S], minimize_det, penalty_alpha)

    # Convert top_candidates to list for iteration
    top_candidates_list = set(top_candidates.tolist())
    # Make sure i is not re-picked
    if i in top_candidates_list:
        top_candidates_list.remove(i)

    while len(S) < max_size:
        best_candidate = None
        best_candidate_score = None  # We'll track the objective after adding candidate

        # Test adding each candidate c not already in S
        for c in top_candidates_list:
            newS = S + [c]
            new_score = dpp_objective_score(k[newS], minimize_det, penalty_alpha)

            # For minimize_det == True, 'score' is negative if determinant is large,
            # so the "better" subset has a smaller numeric value of new_score.
            # For minimize_det == False, bigger numeric value is better.
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

        # If we found no candidate, just break
        if best_candidate is None:
            break

        # See if the best candidate actually improves the score
        # If not, we stop (unless we haven't reached min_size yet).
        if minimize_det:
            improvement_found = (best_candidate_score < current_score)
        else:
            improvement_found = (best_candidate_score > current_score)

        if improvement_found or (len(S) < min_size):
            # Accept the new candidate
            S.append(best_candidate)
            top_candidates_list.remove(best_candidate)
            current_score = best_candidate_score
        else:
            # No improvement, and we have at least min_size
            break

    # Now we have subset S. Do standard attention over S.
    out_s = attention_for_subset(q_i, k[S], v[S])

    # Straight-through version: we'll treat out_s as final. If you'd like
    # a "soft" distribution across possible subsets, you'd modify this part.
    return out_s

#
# The rest: original GPT code, with no changes aside from calling our new DPP function
#

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't allow bias=False directly. """
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

    # New DPP options
    use_dpp_attention: bool = False
    dpp_min_size: int = 2
    dpp_max_size: int = 8
    dpp_top_m: int = 8
    dpp_temperature: float = 0.1
    dpp_minimize_det: bool = True
    dpp_penalty_alpha: float = 1.0

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
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
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        B, T, C = x.size()

        # Project q, k, v
        qkv = self.c_attn(x)  # shape [B, T, 3*C]
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # [B, n_head, T, head_size]
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # If not using DPP-based approach, do standard attention
        if not self.config.use_dpp_attention:
            if self.flash:
                # flash
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True
                )
            else:
                # manual
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v
        else:
            # Use our new DPP-based approach with greedy selection
            y = torch.zeros_like(q)
            for b_idx in range(B):
                for h_idx in range(self.n_head):
                    qbh = q[b_idx, h_idx]  # [T, head_size]
                    kbh = k[b_idx, h_idx]
                    vbh = v[b_idx, h_idx]
                    # Run the discrete approach
                    out_bh = dpp_straight_through_attention(
                        qbh, kbh, vbh,
                        causal=True,  # or False if you want non-causal
                        min_size=self.config.dpp_min_size,
                        max_size=self.config.dpp_max_size,
                        top_m=self.config.dpp_top_m,
                        temperature=self.config.dpp_temperature,
                        minimize_det=self.config.dpp_minimize_det,
                        penalty_alpha=self.config.dpp_penalty_alpha
                    )
                    y[b_idx, h_idx] = out_bh

        # [B, n_head, T, head_size]
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

        # Tie the word embedding and the final LM head
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Scaled init for residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

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
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
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
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt:", model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # Create from-scratch GPT
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
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
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, {num_decay_params} params")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, {num_nodecay_params} params")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
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
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        return mfu
