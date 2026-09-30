"""
Recirculation from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - rms_norm
def rms_norm(x, gain, eps=1e-6):
    """Apply RMSNorm over the last dimension with a learnable gain vector and eps 1e-6."""
    # TODO: Implement rms_norm so that it normalizes x over the last dimension and then scales by gain.
    ms = torch.mean(x ** 2, dim = -1, keepdim=True)

    rms = torch.sqrt(ms + eps)

    return x / rms * gain

# Step 2 - causal_self_attention
def causal_self_attention(x, w_q, w_k, w_v, w_o):
    """Compute single-head causal scaled-dot-product attention."""
    # TODO: Compute single-head causal scaled-dot-product attention...
    T = x.shape[-2]
    D = x.shape[-1]

    Q = (x @ w_q) 
    K = (x @ w_k) 
    V = (x @ w_v)

    scores = Q @ K.transpose(-2,-1) / (D ** .5)

    future_mask = torch.triu(torch.ones(T, T, dtype = torch.bool, device=x.device), diagonal=1)

    scores = scores.masked_fill(future_mask, float("-inf"))

    attention_weights = torch.softmax(scores, dim=-1)

    context = attention_weights @ V 

    output = context @ w_o

    return output

# Step 3 - gelu_ffn
def gelu_ffn(x, w_ff1, w_ff2):
    """Apply a two-layer position-wise GELU feed-forward that expands to 4D then projects back to D."""
    # TODO: Apply a two-layer position-wise GELU FFN that expands to 4D then back to D...
    proj = x @ w_ff1

    gelu_func = torch.nn.functional.gelu(proj)

    return gelu_func @ w_ff2

# Step 4 - pre_norm_block
def pre_norm_block(x, block):
    """Wrap attention and feed-forward as a pre-norm residual transformer block."""
    

    norm_sa = rms_norm(x, block["attn_gain"])

    c_s_a = causal_self_attention(norm_sa, block["w_q"], block["w_k"], block["w_v"], block["w_o"])

    x_aftr_attn = x + c_s_a

    norm_ffn = rms_norm(x_aftr_attn, block["ffn_gain"])

    gelu = gelu_ffn(norm_ffn, block["w_ff1"], block["w_ff2"])

    x_after_gelu = x_aftr_attn + gelu

    return x_after_gelu

# Step 5 - embed_tokens
def embed_tokens(tokens, embedding_weight):
    """Embed token ids with a (V, D) table."""
    # TODO: Implement embed_tokens to produce a residual-stream vector for every token id.
    
    return embedding_weight[tokens]

# Step 6 - run_layers
def run_layers(x, blocks):
    """Return residual streams after every layer including the embedding as index 0."""
    
    streams = [x]

    for block in blocks:
       x = pre_norm_block(x, block)
       streams.append(x)

    return streams

# Step 7 - last_axis_l2
def last_axis_l2(x):
    """Return last-axis L2 norms of x with a kept singleton dimension."""
    # TODO: Compute the L2 norm of a tensor over its last axis keeping that axis as a singleton.
    
    return torch.linalg.norm(x, dim=-1, keepdim=True)

# Step 8 - match_source_norm (not yet solved)
# TODO: implement

# Step 9 - convex_mix (not yet solved)
# TODO: implement

# Step 10 - nonconvex_mix (not yet solved)
# TODO: implement

# Step 11 - no_normalization_mix (not yet solved)
# TODO: implement

# Step 12 - recirculate_one_position (not yet solved)
# TODO: implement

# Step 13 - ramped_alpha (not yet solved)
# TODO: implement

# Step 14 - sequential_prefill (not yet solved)
# TODO: implement

# Step 15 - insert_loop (not yet solved)
# TODO: implement

# Step 16 - run_looped (not yet solved)
# TODO: implement

# Step 17 - tied_lm_head (not yet solved)
# TODO: implement

# Step 18 - ntp_loss (not yet solved)
# TODO: implement

# Step 19 - perplexity (not yet solved)
# TODO: implement

# Step 20 - concat_residuals (not yet solved)
# TODO: implement

# Step 21 - scalar_mix_mlp (not yet solved)
# TODO: implement

# Step 22 - vector_mix_mlp (not yet solved)
# TODO: implement

# Step 23 - hadamard_mix (not yet solved)
# TODO: implement

# Step 24 - adaptive_recirculate (not yet solved)
# TODO: implement

# Step 25 - blockwise_recirculate (not yet solved)
# TODO: implement

# Step 26 - lag_diagnostic (not yet solved)
# TODO: implement

# Step 27 - frozen_stack_adaptive_demo (not yet solved)
# TODO: implement

