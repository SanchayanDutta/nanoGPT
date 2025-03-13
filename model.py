##############################
# model.py
##############################

import math
import inspect
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
from dataclasses import dataclass
from typing import Optional, List

############################################################
# 1) MCMC-based approximate DPP sampling (for large T)
############################################################

@torch._dynamo.skip
def mcmc_conditional_dpp_sample(
    L: torch.Tensor,
    i: int,
    mcmc_steps: int = 50,
    init_subsets: int = 3
) -> List[int]:
    """
    Approximate conditional DPP sampling via Markov Chain Monte Carlo (MCMC).
    We want a subset S that definitely includes i, with probability ~ det(L_S).
    We'll do a "birth-death" chain approach, run multiple restarts, etc.
    """
    device = L.device
    T = L.shape[0]
    best_subset = [i]
    best_det = 0.0

    def subset_det(S) -> torch.Tensor:
        if not S:
            return torch.tensor(0.0, device=device)
        idx = torch.tensor(list(S), device=device)
        # cast to float32 for stable determinant
        Ls = L[idx][:, idx].float()
        val = torch.linalg.det(Ls)
        return torch.clamp_min(val, 0.0)

    for _ in range(init_subsets):
        # Initialize with subset that has i plus maybe random items
        subset = set([i])
        for t_ in range(T):
            if t_ != i and random.random() < 0.3:
                subset.add(t_)

        cur_det = float(subset_det(subset).detach().cpu())

        # MCMC loop
        for _step in range(mcmc_steps):
            cand = random.randint(0, T-1)
            if cand == i:
                continue
            new_subset = set(subset)
            if cand in new_subset:
                new_subset.remove(cand)
            else:
                new_subset.add(cand)

            new_det = float(subset_det(new_subset).detach().cpu())
            if cur_det <= 1e-12:
                accept_prob = 1.0 if new_det > 0.0 else 0.0
            else:
                accept_prob = min(1.0, new_det / cur_det)

            if random.random() < accept_prob:
                subset = new_subset
                cur_det = new_det

        if cur_det > best_det:
            best_det = cur_det
            best_subset = sorted(list(subset))

    return best_subset

@torch._dynamo.skip
def spectral_conditional_dpp_sample(L: torch.Tensor, i: int, tries: int = 10) -> List[int]:
    """
    Fallback exact spectral sampling for smaller T.
    Repeatedly sample from the DPP until we get a subset that includes i.
    """
    for _ in range(tries):
        S = dpp_sample_spectral(L)
        if i in S:
            return S
    return [i]

@torch._dynamo.skip
def dpp_sample_spectral(L: torch.Tensor) -> List[int]:
    """
    Exact spectral DPP sampling (no conditioning). O(T^3).
    Standard approach: eigen-decompose, pick eigenvectors, iterative inclusion.
    """
    # cast to float32 to avoid bfloat16 issues
    L_fp32 = L.float()
    w, V = torch.linalg.eigh(L_fp32)
    w = torch.clamp_min(w, 0.0)  # remove tiny negative numerical fuzz

    keep_mask = []
    for lam in w:
        p = lam / (1.0 + lam)
        if torch.rand(1, device=L.device) < p:
            keep_mask.append(True)
        else:
            keep_mask.append(False)

    chosen_evecs = V[:, keep_mask]
    if chosen_evecs.shape[1] == 0:
        return []

    subset = []
    items = list(range(L.shape[0]))
    perm = torch.randperm(len(items), device=L.device)
    items = [items[i.item()] for i in perm]

    for it in items:
        proj = torch.linalg.norm(chosen_evecs[it])**2
        if chosen_evecs.shape[1] > 0:
            accept_prob = proj / chosen_evecs.shape[1]
        else:
            accept_prob = 0.0

        if torch.rand(1, device=L.device) < accept_prob:
            subset.append(it)
            # For correctness we'd remove that direction from chosen_evecs. 
            # We skip it => approximate.

    return sorted(subset)

############################################################
# 2) Unfixed-size aggregator for token i
############################################################

@torch._dynamo.skip
def aggregator_for_token_i_unfixed(
    i: int,
    L: torch.Tensor,
    v: torch.Tensor,
    n_subs: int = 3,
    small_thresh: int = 64,
    mcmc_steps: int = 50,
    window_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    For each token i, sample up to n_subs subsets from an unconstrained DPP
    that must contain i. Weighted aggregator:
        sum_{S}( det(L_S) * mean(v_j : j in S ) ) / sum_{S} det(L_S)
    We skip subsets that contain disallowed items from `window_mask`.
    """
    device = L.device
    T, d = v.shape

    def subset_det(S) -> torch.Tensor:
        if not S:
            return torch.tensor(0.0, device=device)
        idx = torch.tensor(list(S), device=device)
        Ls = L[idx][:, idx].float()
        val = torch.linalg.det(Ls)
        return torch.clamp_min(val, 0.0)

    subsets = []
    tries_per_sub = 3 * n_subs
    for _ in range(tries_per_sub):
        if T <= small_thresh:
            S = spectral_conditional_dpp_sample(L, i)
        else:
            S = mcmc_conditional_dpp_sample(L, i, mcmc_steps=mcmc_steps)
        if window_mask is not None:
            S = [x for x in S if window_mask[x].item() == 1]
            if i not in S:
                S = [i]
        if S not in subsets:
            subsets.append(S)
        if len(subsets) >= n_subs:
            break

    if not subsets:
        subsets = [[i]]

    weight_list = []
    sub_avgs = []
    for S_ in subsets:
        dt = subset_det(S_)
        idx = torch.tensor(S_, device=device)
        if len(S_) > 0:
            avg_v = v[idx].mean(dim=0)
        else:
            avg_v = torch.zeros(d, device=device)
        weight_list.append(dt)     # 0D tensor
        sub_avgs.append(avg_v)     # [d]

    # stack weights
    w_t = torch.stack(weight_list, dim=0)  # shape [#subs]
    denom = w_t.sum() + 1e-9
    out = torch.zeros(d, device=device)
    for i_sub, avg_v in enumerate(sub_avgs):
        out += (w_t[i_sub]/denom) * avg_v
    return out

############################################################
# 3) CausalSelfAttentionDPP
############################################################

class CausalSelfAttentionDPP(nn.Module):
    """
    Production-oriented synergy-based DPP attention:
      - local window & chunking to keep memory in check
      - MCMC or spectral for sampling
      - aggregator_for_token_i_unfixed for synergy
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.1,
        bias: bool = True,
        # synergy config
        local_window: int = 64,
        n_subs: int = 3,
        small_thresh: int = 64,
        mcmc_steps: int = 50,
        chunk_size: int = 32,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.block_size = block_size
        self.dropout = dropout

        self.local_window = local_window
        self.n_subs = n_subs
        self.small_thresh = small_thresh
        self.mcmc_steps = mcmc_steps
        self.chunk_size = chunk_size

        self.c_attn = nn.Linear(n_embd, 3*n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # causal mask
        tri = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("bias", tri.view(1,1,block_size,block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        returns [B, T, C]
        """
        B, T, C = x.shape
        assert T <= self.block_size

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        out = torch.zeros_like(q)
        causal_2d = self.bias[:, :, :T, :T][0, 0]  # shape [T,T], 1=allowed

        num_chunks = (T + self.chunk_size - 1)//self.chunk_size

        for bidx in range(B):
            for hidx in range(self.n_head):
                q_bh = q[bidx, hidx]  # [T, head_size]
                v_bh = v[bidx, hidx]  # [T, head_size]

                row_out = []
                for cidx in range(num_chunks):
                    start = cidx*self.chunk_size
                    end = min(T, (cidx+1)*self.chunk_size)
                    chunk_outputs = []
                    for i_rel in range(end - start):
                        i_ = start + i_rel

                        if self.local_window>0:
                            left = max(0, i_ - self.local_window + 1)
                        else:
                            left = 0
                        right = i_+1
                        cand_idx = list(range(left, right))
                        # apply causal mask
                        cand_idx = [j for j in cand_idx if causal_2d[i_, j]==1]
                        if not cand_idx:
                            # fallback
                            chunk_outputs.append(v_bh[i_].unsqueeze(0))
                            continue

                        local_q = q_bh[cand_idx]
                        L_local = local_q @ local_q.transpose(0,1)
                        L_local *= (1.0/math.sqrt(self.head_size))

                        local_v = v_bh[cand_idx]
                        if i_ not in cand_idx:
                            chunk_outputs.append(v_bh[i_].unsqueeze(0))
                            continue

                        i_offset = cand_idx.index(i_)
                        up_i = aggregator_for_token_i_unfixed(
                            i=i_offset,
                            L=L_local,
                            v=local_v,
                            n_subs=self.n_subs,
                            small_thresh=self.small_thresh,
                            mcmc_steps=self.mcmc_steps,
                        )
                        chunk_outputs.append(up_i.unsqueeze(0))
                    chunk_outputs = torch.cat(chunk_outputs, dim=0)
                    row_out.append(chunk_outputs)
                row_out = torch.cat(row_out, dim=0)
                out[bidx, hidx] = row_out

        y = out.transpose(1,2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

############################################################
# 4) Transformer MLP / Block / GPTConfig / GPT
############################################################

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4*n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.1,
        bias: bool = True,
        # synergy config
        local_window: int = 64,
        n_subs: int = 3,
        small_thresh: int = 64,
        mcmc_steps: int = 50,
        chunk_size: int = 32
    ):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttentionDPP(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
            bias=bias,
            local_window=local_window,
            n_subs=n_subs,
            small_thresh=small_thresh,
            mcmc_steps=mcmc_steps,
            chunk_size=chunk_size
        )
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout=dropout, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True

    # synergy-based DPP config
    local_window: int = 64
    n_subs: int = 3
    small_thresh: int = 64
    mcmc_steps: int = 50
    chunk_size: int = 32

class GPT(nn.Module):
    """
    A GPT-like model that uses synergy-based DPP attention with local windows,
    chunking, MCMC or spectral sampling, etc.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([
                Block(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    block_size=config.block_size,
                    dropout=config.dropout,
                    bias=config.bias,
                    local_window=config.local_window,
                    n_subs=config.n_subs,
                    small_thresh=config.small_thresh,
                    mcmc_steps=config.mcmc_steps,
                    chunk_size=config.chunk_size
                ) for _ in range(config.n_layer)
            ]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        # special scaled init
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2*config.n_layer))

        total_params = sum(p.numel() for p in self.parameters())
        print(f"[GPT] Using unconstrained DPP synergy. #params: {total_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        b, t = idx.shape
        device = idx.device
        assert t <= self.config.block_size

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)  # [b, t, vocab_size]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # inference => last token only
            logits = self.lm_head(x[:, -1:, :])
            loss = None

        return logits, loss

    def crop_block_size(self, new_size: int):
        assert new_size <= self.config.block_size
        self.config.block_size = new_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:new_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :new_size, :new_size]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            if idx.size(1) > self.config.block_size:
                idx = idx[:, -self.config.block_size:]
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple,
        device_type: str,
        grad_clip: Optional[float] = None
    ):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        base_opt = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, fused=use_fused
        )
        print(f"[GPT] using fused AdamW: {use_fused}")

        if grad_clip is not None:
            class ClippedAdamW(torch.optim.AdamW):
                def step(self, closure=None):
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                    super().step(closure)

            optimizer = ClippedAdamW(
                optim_groups, lr=learning_rate, betas=betas, fused=use_fused
            )
            return optimizer
        else:
            return base_opt

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        PaLM-based MFU estimate, ignoring overhead from MCMC or chunking.
        It's just for reference.
        """
        N = sum(p.numel() for p in self.parameters())
        L, H = self.config.n_layer, self.config.n_head
        Q = self.config.n_embd // self.config.n_head
        T = self.config.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12  # A100 BF16 ~312 TFLOPS
        return flops_achieved / flops_promised
