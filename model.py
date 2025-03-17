import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

#
# Soft DPP helper
#

def soft_dpp_regularization(att_weights, key_vectors, reg_lambda=0.01):
    """
    att_weights: shape [T], a probability distribution over T tokens.
    key_vectors: shape [T, head_size].
    reg_lambda: scalar weighting for how strongly to apply the DPP penalty.

    Returns a scalar DPP penalty that encourages attention to be spread out
    (or concentrated, depending on sign). Typically, you add this to the loss.
    """
    # We can treat att_weights as a diagonal inside a "weighted" Gram matrix
    # Weighted Gram = diag(att_weights) * (K K^T) * diag(att_weights).
    # That is effectively (sqrt of diag(att_weights)) * K * K^T * (sqrt of diag(att_weights)).
    # We then take log-det to get a measure of diversity. We add a negative sign if we
    # want to *increase* that diversity (similar to "maximize det"), or keep it as is
    # if we want to encourage the distribution to concentrate. It depends on your design.
    #
    # Usually we do a negative sign, so that a larger det leads to more negative penalty,
    # which in turn the optimization tries to minimize, thus *increasing* the determinant.
    # But you can also flip the sign if you want the opposite effect.

    # Weighted key vectors by att_weights^0.5
    # shape: [T, 1] * [T, head_size] => broadcast => [T, head_size]
    sqrt_w = torch.sqrt(att_weights + 1e-8).unsqueeze(1)  # [T, 1]
    weighted_k = sqrt_w * key_vectors  # [T, head_size]

    # Now build the Gram matrix: G = weighted_k @ weighted_k.T
    # Then log_det(G)
    G = weighted_k @ weighted_k.T  # shape [T, T]
    # For numerical stability, add tiny epsilon to diagonal
    G = G + 1e-6 * torch.eye(G.size(0), device=G.device)
    det_val = torch.linalg.det(G.float())
    log_det = torch.log(det_val + 1e-8)

    # The final penalty. Typically we do a negative sign to *encourage* a larger determinant:
    penalty = -1.0 * log_det  # negative means we want the model to maximize the det

    # Scale by reg_lambda
    return reg_lambda * penalty


#
# Modified attention that returns both the output and a DPP penalty term
#

class SoftDPPCausalSelfAttention(nn.Module):
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

        # For older PyTorch versions, we do manual masking
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

        # Weight for the soft DPP penalty
        self.dpp_reg_lambda = getattr(config, "dpp_reg_lambda", 0.01)

    def forward(self, x, return_dpp_penalty=False):
        """
        x: [B, T, C]
        return_dpp_penalty: if True, we'll also return the total DPP penalty (summed across heads).
        """
        B, T, C = x.size()
        qkv = self.c_attn(x)  # [B, T, 3*C]
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # [B, n_head, T, head_size]
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # Manual (slow) causal attention
        # shape of att = [B, n_head, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)  # [B, n_head, T, T]

        # Compute the DPP penalty if asked
        dpp_penalty = 0.0
        if return_dpp_penalty and self.dpp_reg_lambda > 0:
            # We sum penalty across all heads and all batch elements
            # att[b, h, i, :] are the attention weights for token i of head h in batch b
            # k[b, h, :, :] are the key vectors for that head
            for b_idx in range(B):
                for h_idx in range(self.n_head):
                    # We can either average penalty across T or sum it
                    # Typically we sum so that each query contributes
                    # you may prefer an average if you want to scale differently
                    batch_head_penalty = 0.0
                    kbh = k[b_idx, h_idx]  # shape [T, head_size]
                    for i_idx in range(T):
                        w_i = att[b_idx, h_idx, i_idx]  # shape [T], distribution for that query
                        # Soft DPP penalty for this distribution
                        batch_head_penalty = batch_head_penalty + soft_dpp_regularization(
                            w_i, kbh, reg_lambda=self.dpp_reg_lambda
                        )
                    dpp_penalty = dpp_penalty + batch_head_penalty

        # standard attention output
        y = att @ v  # [B, n_head, T, head_size]

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        if return_dpp_penalty:
            return y, dpp_penalty
        else:
            return y

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    # Soft DPP config
    dpp_reg_lambda: float = 0.01  # how strongly to apply the DPP penalty

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

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
        self.attn = SoftDPPCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, return_dpp_penalty=False):
        # If we're collecting DPP penalties, pass the flag down
        if not return_dpp_penalty:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x
        else:
            att_out, dpp_penalty = self.attn(self.ln_1(x), return_dpp_penalty=True)
            x = x + att_out
            x = x + self.mlp(self.ln_2(x))
            return x, dpp_penalty

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie embeddings
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

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

    def forward(self, idx, targets=None, return_dpp_penalty=False):
        """
        return_dpp_penalty: if True, we also collect the sum of all DPP penalties
                            across layers and return it as a separate value.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        total_dpp_penalty = 0.0
        if not return_dpp_penalty:
            for block in self.transformer.h:
                x = block(x)
        else:
            for block in self.transformer.h:
                x, block_penalty = block(x, return_dpp_penalty=True)
                total_dpp_penalty = total_dpp_penalty + block_penalty

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if return_dpp_penalty:
                # Combine the standard cross-entropy loss with the DPP penalty
                # Typically we just add them, but you might want a weighting factor
                # if dpp_reg_lambda is not enough.
                loss = loss + total_dpp_penalty

        if return_dpp_penalty:
            return logits, loss, total_dpp_penalty
        else:
            return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond, targets=None, return_dpp_penalty=False)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
