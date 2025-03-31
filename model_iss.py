# For the first step we have to creat our ModelArgs. This class will hold all configuration parameters (hyperparameters) for our transformer model.

import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass
import math
from typing import Optional
import torch


@dataclass
class ModelArgs:
    # region dim
    # This parameter is our dimension. We are going to use this parameter throughout of all our model. We are going to use this parameter for our weights. We are going to expand our tokens to have a better representation.  Think of it as the "width" of the data as it flows through the Transformer. In simpler terms, each token (word or sub-word) and each position will be represented by a vector of size dim.
    # endregion
    dim: int = 512

    # region n_layers
    # This parameter is our layers. Layers are decoder blocks stacked on top of eact other in the Transformer. Each block processes the input and refines the representation. More layers allow the model to learn hierarchical features and increasingly abstract representations of the input. Early layers might learn basic features (like word meanings), while deeper layers can learn more complex relationships and context.
    # endregion
    n_layers: int = 8

    # region n_heads
    # This parameter is our attention heads. Specifaclly, this is the number of heads for the queries. Generally it equals to the 'n_layers'. But it dosen't have to. Multi-Head Attention allows the model to attend to different parts of the input sequence in parallel. Each head can learn different attention patterns, capturing diverse relationships in the data. With more attention heads we will gain more accurate token prediction.
    # endregion
    n_heads: int = 8

    # region n_kv_heads
    # n_kv_heads is the number of heads specifically for the keys and values in the attention mechanism. It's optional and defaults to None.
    # Grouped-Query Attention (GQA) / Multi-Query Attention (MQA): When n_kv_heads is less than n_heads, it indicates the use of Grouped-Query Attention or Multi-Query Attention. In these techniques, multiple query heads share the same key and value heads.
    # Optimization: GQA/MQA is a technique to reduce the computational cost and memory bandwidth requirements of attention, especially during inference. By sharing key/value heads, you reduce the number of key and value projections and the size of the KV cache.
    # It is also important to determine the Context size (string length); a larger kv cache can get more attention, thus increasing the context window. In the output scenario, it depends on your EOS token configuration and your dataset. If your dataset contains very very long responses, your modal will probably produce longer outputs. You still need a large context window for these outputs to be meaningful.
    # endregion
    n_kv_heads: Optional[int] = None

    # region vocab_size
    # Defines the size of the vocabulary, i.e the numebr of unique tokens in that the model can understand and generate. "-1" means: keep the default as -1 for now. We'll handle setting the actual vocab_size when we build the Transformer class.
    # endregion
    vocab_size: int = -1

    multiple_of: int = 256
    # region ffn_dim_multiplier
    # Hyperparameter for the hidden dimension of the Feed-Forward Network. Scales the FFN hidden dimension. If None, a default scaling is used. FFN hidden dimension is typically set to be a multiple of 'multiple_of' for hardware efficiency.
    # Scaling FFN Dimension: If you provide a float value (e.g., 4.0 or a smaller value), the hidden dimension of the FFN will be multiplied by this factor. Commonly, FFN hidden dimension is set to be 2 to 4 times larger than the model dimension (original Transformer paper used a factor of 4). However, some models might use different scaling factors.
    # endregion
    ffn_dim_multiplier: Optional[float] = None

    # region norm_eps
    # norm_eps is a small value (epsilon) added to the denominator during normalization (specifically in RMSNorm) to prevent division by zero. Numerical Stability: Normalization involves dividing by the standard deviation or a similar measure. If the variance is very close to zero, this can lead to numerical instability (division by a very small number, potentially resulting in very large or infinite values).
    # Preventing NaN/Inf: Adding a small eps (like 1e-5 or 1e-6) ensures that the denominator is always slightly larger than zero, even if the variance is extremely small, thus preventing division by zero and maintaining numerical stability.
    # Standard Practice: Using a small epsilon in normalization layers is a standard practice in deep learning to improve numerical stability and prevent issues during training.
    # endregion
    norm_eps: float = 1e-5

    # region max_seq_len
    # Context window size. Determines the maximum sequence length the model is designed to handle and the size of the KV cache.
    # endregion
    max_seq_len: int = 2048

    # region max_batch_size
    # Max batch size= available GPU memory bytes / 4 / (size of tensors + trainable parameters)
    # You can obtain them with 'from torchsummary import summary summary(model)'
    # Let's assume we have a Tesla P100 at hand with 16 GB memory.
    # (16000 - model_size) / (forward_back_ward_size)
    # (16000 - 4.3) / 13.93 = 1148.29
    # rounded to powers of 2 results in batch size 1024
    # endregion
    max_batch_size = 2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------------------------------------------------

# For this section we are going to code RMSNorm:

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # Normalization hyperparameter.
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable parameter!

    # region What is RMSNorm and how does it compare to LayerNorm?
        # ------------------------------------------------------
        # RMSNorm (Root Mean Square Normalization) is a normalization technique, similar to LayerNorm,
        # used to stabilize training in deep neural networks, particularly in Transformers.
        # Both RMSNorm and LayerNorm normalize activations *within* each layer of a neural network.
        # However, they differ in how they calculate the normalization statistic and what they normalize by.
        #
        # Key Differences from LayerNorm:
        # 1. Normalization Statistic:
        #    - LayerNorm uses *both* the mean and the variance (standard deviation) for normalization. It centers the data (subtracts the mean) and scales it (divides by standard deviation).
        #    - LayerNorm(x) = gamma * (x - mean(x)) / sqrt(variance(x) + epsilon) + beta
        #    - RMSNorm *only* uses the Root Mean Square (RMS) for normalization. It scales the data by dividing by the RMS, but it does *not* center the data (no mean subtraction).
        #    - RMSNorm(x) = gamma * x / RMS(x)
        #
        # 2. Computational Simplicity and Efficiency:
        #    - RMSNorm is computationally slightly simpler and more efficient than LayerNorm because it avoids calculating and subtracting the mean.
        #    - This can be beneficial in large models where even small efficiency gains can be significant.
        #
        # 3. Learnable Parameters:
        #    - Both LayerNorm and RMSNorm typically include a learnable *scale* parameter (gamma - 'self.weight' in our RMSNorm). This allows the network to learn the optimal scale for the normalized features.
        #    - LayerNorm *additionally* includes a learnable *bias* parameter (beta), which RMSNorm usually omits in its standard form.
        #
        # 4. Output Distribution:
        #    - LayerNorm tends to produce outputs that are closer to having *zero mean and unit variance* (approximately, after normalization and learnable scaling/bias).
        #    - RMSNorm focuses on ensuring that the output vectors have a *unit Root Mean Square (RMS) magnitude*. The mean of RMSNorm's output is not necessarily zero.
        #
        # Why RMSNorm can be preferred over LayerNorm in some cases (e.g., large language models):
        # - Empirical Performance: In many large language models, RMSNorm has been shown to perform comparably to or even slightly better than LayerNorm, while being more computationally efficient.
        # - Focus on Magnitude: RMSNorm's focus on normalizing magnitude (RMS) can be sufficient for stabilizing training in deep Transformers, and the centering (mean subtraction) of LayerNorm might be less critical in certain architectures or at certain depths in the network.
        #
        # In this implementation of RMSNorm:
        # - We calculate the RMS of the input tensor `x` along the last dimension.
        # - We normalize `x` by dividing it by its RMS value.
        # - We then apply a learnable element-wise scaling using `self.weight`.
        #
        # Key steps in the forward method:
        # 1. Calculate RMS (approximately variance, mean of squares): variance = x.pow(2).mean(-1, keepdim=True)
        # 2. Normalize by RMS (scaling to unit magnitude): hidden_states = x * torch.rsqrt(variance + self.eps)
        # 3. Apply learnable scaling (adjust magnitude and distribution): return (self.weight * hidden_states).to(input_dtype)
    # endregion
    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype
        x_float = x.float()  # Convert to float at the start
        variance = x_float.pow(2).mean(-1, keepdim=True)
        # RMS normalization formula is = x / square root of(x^2 + eps)
        normalized_x = x_float * torch.rsqrt(variance + self.eps)
        # Convert back after normalization but before weight
        return (self.weight.to(x.device) * normalized_x).to(input_dtype)


# -------------------------------------------------------------------------------------------------------------

# region For the RoPE Implementation: Rotary Position Embeddings (RoPE)
    # ------------------------------------------------------
    # Rotary Position Embedding (RoPE) is a positional encoding technique used in Transformer models,
    # especially in large language models, as an alternative to traditional sinusoidal positional embeddings.
    # RoPE encodes positional information by applying rotations to the query and key vectors
    # based on their positions in the sequence.
    #
    # Key Advantages of RoPE:
    # 1. Encodes Relative Position: RoPE is designed to naturally encode relative positional information,
    #    making the attention score dependent on the relative distance between tokens.
    # 2. No Learnable Parameters: RoPE is parameter-free, reducing the number of learnable parameters in the model.
    # 3. Improved Extrapolation: RoPE has shown better generalization to sequence lengths longer than those seen during training.
    # 4. Computationally Efficient: RoPE is computationally efficient to apply.
    #
    # This section implements two functions for RoPE:
    # 1. precompute_theta_pos_frequencies: Pre-calculates the rotation frequencies (theta values)
    #    that are used to define the rotation angles for different dimensions and positions.
    # 2. apply_rotary_embeddings: Applies the rotary embeddings to input tensors (queries and keys)
    #    using the pre-computed frequencies.
# endregion
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    """
    Precompute rotary positional embedding theta (freqs),
    freqs = m * theta, theta_i = 10000^(-2i/d)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs), 
    theta_i = 10000^(-2i/d)
    freqs[pos, i] = pos * theta_i
    freqs_complex[pos, i] = cos(freqs[pos, i]) + j sin(freqs[pos, i]) = r_theta[pos, i]
    Args:
        head_dim (int): dimension of the attention head (must be even)
        seq_len (int): length of the sequence/context window
        device (str): device to place tensors on (cuda or cpu)
        theta (float, optional): base value for frequency computation. Defaults to 10000.0.

    Returns:
        freqs_complex (torch.Tensor): precomputed complex frequencies, shape [seq_len, head_dim/2]
    """
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"

    # [0,2,4,...,head_dim-2]
    # -> shape: [head_dim/2] # 'i' indices for theta calculation
    theta_numerator = torch.arange(0, head_dim, 2).float()

    # theta_i = 10000^(-2i/d)
    # -> shape: [head_dim/2] # frequency bases
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # [0,1,2,..., seq_len -1]
    # -> shape: [seq_len] # position indices
    m = torch.arange(seq_len, device=device)

    # freqs[pos,i] = pos * theta_i
    # -> shape: [seq_len, head_dim/2] # frequency values for each position and dimension
    freqs = torch.outer(m, theta).float()

    # freqs_complex[pos, i] = cos(freqs[pos, i]) + j sin(freqs[pos, i])
    # -> shape: [seq_len, head_dim/2] # complex exponentials for rotation
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    # shape: [seq_len, head_dim/2]
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    """
    Apply rotary embeddings to input tensor x using precomputed frequencies

    Args:
        x (torch.Tensor): Input tensor, shape [batch_size, seq_len, head_dim]
        freqs_complex (torch.Tensor): Precomputed complex frequencies, shape [seq_len, head_dim/2]
        device (str): device to place output tensor on (cuda or cpu)

    Returns:
        x_out (torch.Tensor): Output tensor with rotary embeddings applied, shape [batch_size, seq_len, head_dim]
    """
    # -> shape: [batch_size, seq_len, head_dim/2] # reshape to complex numbers, pairs from last dimension
    # Here, while reshaping, *x.shape[:-1] expression will preserve the batch and sequence length dimensions. Then we will make the automatic dimension inference with -1 and the final dimension with 2. So, if x.shape is (B, seq_len, head_dim), after reshape it will be shape(B, seq_len, head_dim/2, 2). Where did the head_dim/2 dimension come from? Because we divided the final dimension by 2 and grouped it in pairs.
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # -> shape: [1, 1, seq_len, head_dim/2] # prepare freqs_complex for broadcasting
    # Make freqs_complex tensor suitable for element-wise multiplication with x_complex (rotation application). Set freqs_complex dimensions to be broadcastable to x_complex.unsqueeze(0).unsqueeze(2): Add new dimensions to freqs_complex tensor with dimension length 1 in dimensions 0 and 2. This will change the shape of freqs_complex from (seq_len, head_dim/2) to (1, seq_len, 1, head_dim/2). This will allow broadcasting over batch and head dimensions.
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # apply rotation via complex multiplication (element-wise)
    # -> shape: [batch_size, seq_len, head_dim/2]
    x_rotated = x_complex * freqs_complex

    # Convert the real tensor
    # -> shape: [batch_size, seq_len, head_dim/2, 2] # convert back to real tensor (interleaved real and imag parts in last dim)
    x_out = torch.view_as_real(x_rotated)

    # Get back to the original shape
    # -> shape: [batch_size, seq_len, head_dim] # reshape back to original shape, combine last two dims
    x_out = x_out.reshape(*x.shape)

    # Secure the data type and set to the device.
    return x_out.type_as(x).to(device)


# -------------------------------------------------------------------------------------------------------------

# region KV_Repeat Implementation:
    # What is KV Cache and Why Do We Need It?

    # Problem: Autoregressive Generation Inefficiency: Decoder-only Transformer models (and the decoder part of encoder-decoder models) work autoregressively (recursively) in tasks such as text generation. In other words, they generate word by word (token by token). To generate the next word, all previous words must be processed repeatedly.

    # Example: When generating the sentence "The cat sat on the...":

    # "The" is generated.

    # To generate "cat", "The" is processed again, "cat" is generated.

    # To generate "sat", "The cat" is processed again, "sat" is generated.

    # To generate "on", "The cat sat" is processed again, "on" is generated.

    # To generate "the", "The cat sat on" is processed again, "the" is generated.

    # To generate "...", "The cat sat on the" is processed again, "..." is generated.

    # In each new word generation step, all past words are processed from scratch! This means a lot of unnecessary computation, especially when generating long texts. Because the attention mechanism has to calculate the key and value vectors for all past words repeatedly at each step.

    # Solution: KV Cache (Caching): KV Cache is a technique developed to prevent these unnecessary computations. The idea is simple: store (cache) the key and value vectors calculated once and reuse them in the next steps.

    # How it works?

    # When Processing the First Token: When processing the first token (e.g. "The"), the key and value vectors for the attention mechanism are calculated. These keys and values ​​are stored in the KV Cache.

    # When Processing Subsequent Tokens: When processing subsequent tokens (e.g. "cat", "sat", ...):

    # Only the key and value for the new token are calculated.

    # The cached keys and values ​​in the KV Cache are combined with the key and values ​​of the new token.

    # The attention mechanism works on the combined keys and values.

    # In this way, the key and value calculations for past tokens are done only once and can be reused in the next steps.
# endregion
def repeat_kv(x: torch.Tensor, n_repeats: int) -> torch.Tensor:
    """
    Repeats key or value heads to match the number of query heads in Grouped-Query Attention (GQA) or Multi-Query Attention (MQA).

    Args:
        x (torch.Tensor): Input tensor (keys or values), shape [batch_size, seq_len, n_kv_heads, head_dim]
        n_repeats (int): Number of times to repeat the heads

    Returns:
        torch.Tensor: Output tensor with repeated heads, shape [batch_size, seq_len, n_kv_heads * n_repeats, head_dim]
    """
    # Get input tensor shape: [batch_size, seq_len, n_kv_heads, head_dim]
    # batch_size, seq_len, n_kv_heads, head_dim = x.shape
    batch_size, seq_len, n_kv_heads, head_dim = x.shape

    if n_repeats == 1:
        return x

    # Add a new dimension for repeating heads: [batch_size, seq_len, n_kv_heads, 1, head_dim]
    x_expanded = x[:, :, :, None, :]

    # -> shape: [batch_size, seq_len, n_kv_heads, n_repeats, head_dim]
    x_expanded = x_expanded.expand(
        batch_size, seq_len, n_kv_heads, n_repeats, head_dim)

    # -> shape: [batch_size, seq_len, n_kv_heads * n_repeats, head_dim]
    x_repeated = x_expanded.reshape(
        batch_size, seq_len, n_kv_heads * n_repeats, head_dim)
    return x_repeated


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args

        # Indicate the number of keys and values heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        # Indicate the number of heads for the queries
        self.n_q_heads = args.n_heads

        # Indicate how many times the keys and values should be repeated
        self.n_reps = self.n_q_heads // self.n_kv_heads

        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads

        # nn.Linear(in,out) its written as the Transpose from.
        self.wq = nn.Linear(args.dim, self.n_q_heads *
                            self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads *
                            self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads *
                            self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_q_heads * self.head_dim,
                            args.dim, bias=False)

        self.cache_keys = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
            device=args.device
        )
        self.cache_values = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
            device=args.device
        )

    @staticmethod
    def attention(query, key, value, head_dim: int, mask=None):
        # Scaled dot product between the query and the key^T
        attention_scores = (query @ key.transpose(-2, -1)
                            ) / math.sqrt(head_dim)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        # Normalize the last dimension of the attention score (all attentions) with softmax
        attention_scores = F.softmax(attention_scores, dim=-1).type_as(query)
        # Matmul with the value matrix to have a weighted sum of the attention values
        output = attention_scores @ value
        return output

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_complex: torch.Tensor
    ):
        batch_size, seq_len, _ = x.shape

        # x.shape: [batch_size, seq_len, args.dim(self.head_dim * args.n_heads)]
        # self.wq.shape [args.dim,args.dim(self.head_dim * args.n_heads)]
        # inners cancel out and just the out dims left so
        # self.wq(x)  = [batch_size, seq_len, self.head_dim * args.n_heads]
        # -> shape: [batch_size, seq_len, n_heads * head_dim]
        query = self.wq(x)

        # -> shape: [batch_size, seq_len, n_kv_heads * head_dim]
        key = self.wk(x)

        # -> shape: [batch_size, seq_len, n_kv_heads * head_dim]
        value = self.wv(x)

        # -> shape: [batch_size, seq_len, n_q_heads, head_dim]
        query = query.view(batch_size, seq_len, self.n_q_heads, self.head_dim)

        # -> shape: [batch_size, seq_len, n_kv_heads, head_dim]
        key = key.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # -> shape: [batch_size, seq_len, n_kv_heads, head_dim]
        value = value.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # -> shape: [batch_size, seq_len, n_q_heads, head_dim]
        query = apply_rotary_embeddings(query, freqs_complex, device=x.device)

        # -> shape: [batch_size, seq_len, n_kv_heads, head_dim]
        key = apply_rotary_embeddings(key, freqs_complex, device=x.device)

        # During training you usually don’t need to cache key/value tensors across batches (caching is more common during inference to avoid recomputation). If you do need caching, you should detach the cached tensors from the current graph so that they don’t participate in gradient computations.
        # SAVE TO KV CACHE
        # -> shape: [batch_size, seq_len, n_kv_heads, head_dim]
        self.cache_keys[:batch_size,
                        start_pos:start_pos+seq_len] = key.detach()
        # -> shape: [batch_size, seq_len, n_kv_heads, head_dim]
        self.cache_values[:batch_size,
                          start_pos:start_pos+seq_len] = value.detach()

        # RETRIEVE FROM KV CACHE
        # -> shape: [batch_size, start_pos + seq_len, self.n_kv_heads, head_dim]
        keys = self.cache_keys[:batch_size, : start_pos + seq_len]
        # -> shape: [batch_size, start_pos + seq_len, n_kv_heads, head_dim]
        values = self.cache_values[:batch_size, : start_pos + seq_len]

        # -> shape: [batch_size, start_pos + seq_len, n_q_heads (n_kv_heads * self.n_reps), head_dim]
        keys = repeat_kv(keys, self.n_reps)
        values = repeat_kv(values, self.n_reps)

        # For operating our attentions in batches in parallel:
        # -> shape: [batch_size, n_q_heads, seq_len, head_dim]
        query = query.transpose(1, 2)
        # -> shape: [batch_size, n_q_heads, start_pos + seq_len, head_dim]
        key = keys.transpose(1, 2)
        # -> shape: [batch_size, n_q_heads, start_pos + seq_len, head_dim]
        value = values.transpose(1, 2)

        # --- Causal Mask ---
        # query shape: [B, n_heads, current_seq_len, head_dim]
        # key shape: [B, n_heads, total_seq_len, head_dim] where total_seq_len = start_pos + current_seq_len
        current_seq_len = query.shape[-2]
        total_seq_len = key.shape[-2]
        # For each query token at index i (in current forward pass),
        # allow keys with indices <= (start_pos + i)
        mask = (torch.arange(total_seq_len, device=query.device)
                .unsqueeze(0) <= (torch.arange(current_seq_len, device=query.device).unsqueeze(1) + start_pos))
        # Shape: [1, 1, current_seq_len, total_seq_len]
        mask = mask.unsqueeze(0).unsqueeze(0)

        # -> shape: [batch_size, n_q_heads, seq_len, head_dim]
        output = MultiHeadAttentionBlock.attention(
            query, key, value, self.head_dim, mask)

        # From the pytorch documentation: contiguous() → Tensor
        # Returns a contiguous tensor containing the same data as self tensor. If self tensor is contiguous, this function returns the self tensor.
        # Where contiguous here means not only contiguous in memory, but also in the same order in memory as the indices order: for example doing a transposition doesn't change the data in memory, it simply changes the map from indices to memory pointers, if you then apply contiguous() it will change the data in memory so that the map from indices to memory location is the canonical one.
        # -> shape: [batch_size, seq_len, n_q_heads * head_dim]
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1))

        # -> shape: [batch_size, seq_len, args.dim]
        return self.wo(output)


# -------------------------------------------------------------------------------------------------------------

# MLP (FFN) Implementation:
class FeedForward(nn.Module):
    def __init__(
            self,
            args: ModelArgs
    ):
        super().__init__()

        # The value of 4 is a traditional multiplier to increase capacity by expanding the FFN.
        hidden_dim = 4 * args.dim

        # Purpose: Rounding the hidden_dim value up to a multiple of args.multiple_of.
        # args.multiple_of (e.g. 256): This parameter is related to hardware optimizations. In particular, accelerators such as GPUs and TPUs perform more efficiently when some dimensions are aligned to specific numbers (usually multiples of 2, such as 64, 128, 256). Operations such as memory access and matrix multiplication can be optimized.

        # ((hidden_dim + args.multiple_of - 1) // args.multiple_of): This expression performs ceiling division. It divides hidden_dim by args.multiple_of to find the number of multiples of args.multiple_of that hidden_dim is (or the nearest larger multiple).

        # + args.multiple_of - 1: This part is added to allow rounding up.

        # // args.multiple_of: Integer division is performed.

        # args.multiple_of * ...: The result is multiplied by args.multiple_of to obtain the hidden_dim value rounded to the nearest larger multiple of args.multiple_of.
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for FeedForward network.

        Args:
            x (torch.Tensor): Input tensor, shape [batch_size, seq_len, dim]

        Returns:
            torch.Tensor: Output tensor after FFN, shape [batch_size, seq_len, dim]
        """

        # region 3. Swish (SiLU) and Why First w3 Then w2 Implemented? (FFN Architecture)

        # SiLU (Swish-1) Activation: F.silu(self.w1(x))
        # SiLU (Sigmoid Linear Unit): 𝑆𝑖𝐿𝑈(𝑥)=𝑥⋅𝜎(𝑥)
        # SiLU(x)=x⋅σ(x), where 𝜎(𝑥)=1/(1+𝑒−𝑥)σ(x)=1/(1+e−x)sigmoid function.

        # Why SiLU? It has some advantages over other activation functions like ReLU:

        # Smooth: SiLU is a smooth function without sharp corners like ReLU. This can improve the gradient flow and provide more stable training.

        # Better Performance: Some studies have shown that SiLU can outperform ReLU in Transformer models. It is a popular choice especially in large language models.

        # Dynamic Non-linearity: SiLU changes its non-linear behavior depending on the input value. For negative inputs, it approaches zero like ReLU, while for positive inputs it behaves linearly.

        # w1, w2, w3 Linear Layers and "SwiGLU-like" Architecture:

        # Traditional FFN (Original Transformer): It usually consists of 2 linear layers:

        # Linear(dim, hidden_dim) -> ReLU -> Linear(hidden_dim, dim) (ReLU activation is usually used in the middle).

        # FFN in this Implementation (3 Linear Layers and "Gating"): This implementation uses a slightly more complex FFN architecture (similar to SwiGLU):

        # w1: Linear(dim, hidden_dim): Linear projection from input to hidden dimension (for SiLU activation).

        # w3: Linear(dim, hidden_dim): A separate linear projection from the input to the hidden dimension (for the door mechanism).

        # swish = F.silu(self.w1(x))
        # endregion

        # Apply linear layer w1 and SiLU activation: swish = silu(w1(x))
        # -> shape: [batch_size, seq_len, hidden_dim]
        # Expected: [2, 1, 4096]
        swish = F.silu(self.w1(x))

        # Apply linear layer w3: x_V = w3(x) (separate linear path for gating)
        # -> shape: [batch_size, seq_len, hidden_dim]
        # Expected: [2, 1, 16384]
        x_V = self.w3(x)

        # Gating mechanism: x = swish * x_V (element-wise multiplication)
        # -> shape: [batch_size, seq_len, hidden_dim]
        # Expected: [2, 1, 4096]
        x = swish * x_V

        # Apply final linear layer w2: x = w2(x)
        # Should still be [2, 1, 16384]
        x = self.w2(x)

        # shape: [batch_size, seq_len, dim]
        # Expected: [2, 1, 4096]
        return x


# -------------------------------------------------------------------------------------------------------------
# Decoder Block Implementation
class DecoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = MultiHeadAttentionBlock(args)
        self.feed_forward = FeedForward(args)

        # Normalization before attention block
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization before feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )

        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


# -------------------------------------------------------------------------------------------------------------

# Transformer Implementation
class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(
            self.vocab_size, args.dim).to(args.device)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(DecoderBlock(args).to(args.device))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps).to(args.device)
        self.output = nn.Linear(args.dim, self.vocab_size,
                                bias=False).to(args.device)

    def forward(self, tokens: torch.Tensor, start_position: int):

        # (B, seq_len)
        batch_size, seq_len = tokens.size()
        # I dont want to use this assert because I want to use the model for multiple tokens at a time so i'll implement a casualmask group query attention
        # assert seq_len == 1, "Only one token at a time"

        # Converting to embeddings
        # (B, seq_len) -> (B, seq_len, dim)
        tokens = tokens.to(self.args.device)
        h = self.tok_embeddings(tokens)

        # Retrive the Pairs (m, theta) correposing to the position [start_position, start_position+seq_len] because this is rotary embedding
        # Compute freqs_complex dynamically based on the actual sequence length
        freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads, seq_len, device=self.args.device)

        # Apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_position, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output


if __name__ == "__main__":
    # Test Transformer (Uçtan uca test!)
    print("cuda" if torch.cuda.is_available() else "cpu")
    print("\nTransformer Test:")
    # vocab_size ve max_seq_len ayarla
    args = ModelArgs(vocab_size=1000, max_seq_len=10)
    transformer = Transformer(args)  # Transformer örneği oluştur
    tokens = torch.randint(0, args.vocab_size, (2, 1)).to(
        args.device)  # Rastgele tokenler (batch=2, seq_len=1)
    start_position = 0
    output = transformer.forward(
        tokens, start_position)  # Transformer'ı çalıştır

    print("Input tokens shape:", tokens.shape)  # Beklenen: [2, 1]
    print("Output logits shape:", output.shape)  # Beklenen: [2, 1, vocab_size]
    print("\nOutput Logits (first example, first token):\n",
          output[0, 0, :10])  # İsteğe bağlı: İlk 10 logiti yazdır

    # args = ModelArgs()
    # print(args)  # Print the default arguments
    # print(f"Model Dimension: {args.dim}")  # Access a specific argument

    # # Test RMSNorm
    # rms_norm = RMSNorm(dim=10)
    # input_tensor = torch.randn(2, 5, 10)  # Example input
    # output_tensor = rms_norm(input_tensor)
    # print("\nRMSNorm Test:")
    # print("Input shape:", input_tensor.shape)
    # print("Output shape:", output_tensor.shape)
    # # Optional: print some values
    # print("\nInput Tensor (first row):\n", input_tensor[0])
    # # Optional: print some values
    # print("\nOutput Tensor (first row):\n", output_tensor[0])

    # # RoPE Testi (apply_rotary_embeddings)
    # print("\nRoPE Uygulama Testi:")
    # head_dim = 16  # Örnek head_dim
    # seq_len = 32  # Örnek seq_len
    # device = "cpu"  # veya "cuda"
    # freqs_complex = precompute_theta_pos_frequencies(
    #     head_dim, seq_len, device)  # frekansları hesapla
    # x_rope_input = torch.randn(2, seq_len, head_dim)  # örnek girdi
    # x_rope_output = apply_rotary_embeddings(
    #     x_rope_input, freqs_complex, device)  # RoPE uygula

    # # Shape kontrolü: (seq_len, head_dim/2)
    # print("freqs_complex shape:", freqs_complex.shape)
    # # Shape kontrolü: (2, seq_len, head_dim)
    # print("Input x_rope_input shape:", x_rope_input.shape)
    # # Shape kontrolü: (2, seq_len, head_dim)
    # print("Output x_rope_output shape:", x_rope_output.shape)
    # # print("\nInput Tensor (ilk row, ilk 5 eleman):\n", x_rope_input[0, 0, :5]) # İsteğe bağlı: Değer kontrolü
    # # print("\nOutput Tensor (ilk row, ilk 5 eleman sonrası RoPE):\n", x_rope_output[0, 0, :5]) # İsteğe bağlı: Değer kontrolü

    # # Test repeat_kv
    # print("\nrepeat_kv Test:")
    # batch_size = 2
    # seq_len = 4
    # n_kv_heads = 2
    # head_dim = 3
    # n_repeats = 3  # Örnek tekrar sayısı (n_q_heads // n_kv_heads gibi)

    # x_kv_input = torch.randn(
    #     batch_size, seq_len, n_kv_heads, head_dim)  # örnek girdi
    # x_kv_repeated = repeat_kv(x_kv_input, n_repeats)  # repeat_kv uygula

    # # Beklenen: [2, 4, 2, 3]
    # print("Input x_kv_input shape:", x_kv_input.shape)
    # # Beklenen: [2, 4, 6, 3] (n_kv_heads * n_repeats = 2 * 3 = 6)
    # print("Output x_kv_repeated shape:", x_kv_repeated.shape)
    # # print("\nInput Tensor:\n", x_kv_input) # İsteğe bağlı: Değer kontrolü
    # # print("\nOutput Tensor (repeated heads):\n", x_kv_repeated) # İsteğe bağlı: Değer kontrolü

    # # Test FeedForward
    # print("\nFeedForward Test:")
    # args = ModelArgs(dim=512)  # Örnek ModelArgs
    # ffn = FeedForward(args)  # FeedForward örneği oluştur
    # input_ffn = torch.randn(2, 32, args.dim)  # Örnek girdi
    # output_ffn = ffn(input_ffn)  # FeedForward'dan geçir

    # # Beklenen: [2, 32, 512] (örnek args.dim'e göre)
    # print("Input ffn_input shape:", input_ffn.shape)
    # print("Output ffn_output shape:", output_ffn.shape)
