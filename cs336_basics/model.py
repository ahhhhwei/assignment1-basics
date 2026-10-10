import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_out, d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = Q.shape[-1] # shape[-1] 就是取 Tensor 最后一个维度
    # 多头 Attention：
    # Q.shape = (batch, heads, queries, d_k)
    # K.shape = (batch, heads, key,     d_k)
    # 二维可以用 .T，多维用transpose，只交换最后两个维度
    scores = Q @ K.transpose(-2, -1)
    scores = scores / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attention_weights = torch.softmax(scores, dim=-1)
    return attention_weights @ V

# 早期 transformer 是直接告诉位置的，ROPE 改成旋转角度，主要是因为 Attention 真正关心的往往不是绝对位置，而是相对距离
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 1. 计算每一对维度的旋转频率
        freq = self.theta ** (-torch.arange(0, self.d_k, 2, device=x.device, dtype=x.dtype) / self.d_k)

        # 2. 每个 token 的位置 × 频率 = 旋转角
        angles = token_positions[..., None] * freq

        # 3. 计算旋转需要的 sin / cos
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # 4. 把维度两两拆开
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # 5. 二维旋转
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # 6. 再交叉放回原来的维度
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd

        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 把 d_model 拆成 num_heads × d_k
        Q = Q.unflatten(-1, (self.num_heads, self.d_k))
        K = K.unflatten(-1, (self.num_heads, self.d_k))
        V = V.unflatten(-1, (self.num_heads, self.d_k))

        # (..., seq_len, heads, d_k)
        # →
        # (..., heads, seq_len, d_k)
        Q = Q.transpose(-3, -2)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)

        seq_len = x.shape[-2]

        # tril 取下三角矩阵，语言模型不能偷看未来
        # token0 只能看 token0
        # token1 可以看 token0 token1
        # token2 可以看 token0 token1 token2
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        output = scaled_dot_product_attention(Q, K, V, mask)
        # (..., heads, seq_len, d_k)
        # →
        # (..., seq_len, heads, d_k)
        output = output.transpose(-3, -2)

        # heads × d_k 合并回 d_model
        output = output.flatten(-2)
        return self.output_proj(output)

class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        # 每个 head 中 Q/K 的维度
        self.d_k = d_model // num_heads

        # 生成 Q、K、V 的线性层
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)

        # 多头 Attention 做完之后的输出投影
        self.output_proj = Linear(d_model, d_model)

        # RoPE
        self.rope = RoPE(
            theta=theta,
            d_k=self.d_k,
            max_seq_len=max_seq_len,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # ① 生成 Q、K、V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # ② 把 d_model 拆成 num_heads × d_k
        Q = Q.unflatten(-1, (self.num_heads, self.d_k))
        K = K.unflatten(-1, (self.num_heads, self.d_k))
        V = V.unflatten(-1, (self.num_heads, self.d_k))

        # (..., seq_len, heads, d_k)
        # →
        # (..., heads, seq_len, d_k)
        Q = Q.transpose(-3, -2)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)

        seq_len = x.shape[-2]

        # ③ 如果没有提供位置，就默认 0,1,2,...,seq_len-1
        if token_positions is None:
            token_positions = torch.arange(
                seq_len,
                device=x.device,
            )

        # 为 head 这个维度补一个维度，方便广播
        while token_positions.ndim < Q.ndim - 1:
            token_positions = token_positions.unsqueeze(-2)

        # ④ 只给 Q、K 做 RoPE
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        # ⑤ causal mask：不能看未来 token
        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device,
            )
        )

        # ⑥ 每个 head 做 Attention
        output = scaled_dot_product_attention(
            Q,
            K,
            V,
            mask,
        )

        # ⑦ 把 head 移回去
        output = output.transpose(-3, -2)

        # ⑧ num_heads × d_k → d_model
        output = output.flatten(-2)

        # ⑨ 输出投影
        return self.output_proj(output)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model)

        self.attn = MultiHeadSelfAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
        )

        self.ln2 = RMSNorm(d_model)

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 第一部分：Attention
        x = x + self.attn(self.ln1(x))

        # 第二部分：FFN
        x = x + self.ffn(self.ln2(x))

        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()

        # 1. token id -> d_model 维向量
        self.token_embeddings = Embedding(
            vocab_size=vocab_size,
            d_model=d_model,
        )

        # 2. 堆叠 num_layers 个 Transformer Block
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
            )
            for _ in range(num_layers)
        ])

        # 3. 最后再做一次 RMSNorm
        self.ln_final = RMSNorm(d_model)

        # 4. d_model -> vocab_size
        self.lm_head = Linear(
            d_in=d_model,
            d_out=vocab_size,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        # token id -> embedding
        x = self.token_embeddings(token_ids)

        # 依次经过所有 Transformer Block
        for layer in self.layers:
            x = layer(x)

        # 最终归一化
        x = self.ln_final(x)

        # 映射到整个词表，得到 logits
        logits = self.lm_head(x)

        return logits