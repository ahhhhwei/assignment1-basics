import torch
import math
from torch import nn

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # torch.empty() 分配一块内存，但暂时不做初始化
        # nn.Parameter(...) 告诉 nn.model 这是模型需要管理和训练的参数
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        std = math.sqrt(2 / (d_in + d_out))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y=xW^T
        return x @ self.weight.T

# 把 tokenID 变成一个向量
class Embedding(nn.Module):
    # vocab_size 表示词表大小（tokenizer 总计定义了多少个 token，d_model 表示每个 token 的向量维度
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

# 把一个 token 的隐藏向量做归一化，让数值尺度更稳定
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # torch.ones()：创建一个元素全为1的 Tensor
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dim=-1：沿最后一个维度计算 keepdim=True：保持 shape 更规整，方便后续广播运算
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff) # 升维
        self.w2 = Linear(d_ff, d_model) # 降维
        self.w3 = Linear(d_model, d_ff) # 升维
        self.silu = SiLU() # 创建一个 SiLU 激活函数 SiLU(x)=xσ(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x)) # * 就是逐元素相乘 ⊙
        # nn.Module 的对象可以像函数一样调用，这是因为 Pytorch 的 nn.Module 实现了 __call__()，比较类似于 C++ 中的 operator()

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = Q.shape[-1] # shape[-1] 就是取 Tensor 最后一个维度
    # 多头 Attention：
    # Q.shape = (batch, heads, queries, d_k)
    # K.shape = (batch, heads, key,     d_k)
    # 二维可以用 .T，多维用transpose，只交换最后两个维度
    scores = Q @ K.transpose(-2, -1)
    scores = scores / (d_k ** 0.5)
    if mask is not None:
        # mask 为 False 的地方，把 score 改成负无穷（e^(-∞)=0）
        scores = scores.masked_fill(~mask, float("-inf")) # ~mask 表示布尔取反
    attention_weights = torch.softmax(scores, dim=-1)
    return attention_weights @ V

# 早期 transformer 是直接告诉位置的，ROPE 改成旋转角度，主要是因为 Attention 真正关心的往往不是绝对位置，而是相对距离
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        self.theta = theta # 控制不同维度组 ⌈转得多快⌋ 的一个超参数
        self.d_k = d_k # d_k 是每个 head 的维度：d_k=d_model/head_num
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 1. 计算每一对维度的旋转频率 w_i=Θ^(-2i/d_k)
        freq = self.theta ** (-torch.arange(0, self.d_k, 2, device=x.device, dtype=x.dtype) / self.d_k)
        # torch.arange(start, end, step) 是 PyTorch 里生成等差序列张量的函数，这里是在取每一组二维维度的第一个下标

        # 2. 旋转角 = 每个 token 的位置 * 频率
        angles = token_positions[..., None] * freq # ... 表示前面的所有维度都保留    None 表示插入一个长度为1的新维度
        # 这里 token_positions[..., None] 是通过增加一个维度，把 token_positions 从横着变成竖着
        # 假设此时 token_positions 维度 [4, 1]，freq 维度是 [3]，最终两者都会被广播成 [4, 3]
        # PyTorch 的广播规则：
        # - 从最后一个维度开始比较。
        # - 两个维度满足以下任意一个，就能广播：
        #    - 两者相等
        #    - 其中一个是 1
        #    - 某一边没有这个维度，默认当成 1

        # 3. 计算旋转需要的 sin / cos
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        # 4. 把维度两两拆开
        # ... 前面的所有维度都保留，只对最后一维进行切片
        x_even = x[..., 0::2] # 偶数切片 从下标 0 开始，一直到最后，每隔 2 个取一个
        x_odd = x[..., 1::2]  # 奇数切片 从下标 1 开始，一直到最后，每隔 2 个取一个
        # 5. 二维旋转
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        # 6. 再交叉放回原来的维度
        out = torch.empty_like(x) # 先创建一个跟 x：shape 一样，dtype 一样，device 一样的 Tensor
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
        Q = Q.unflatten(-1, (self.num_heads, self.d_k)) # 把最后一个维度按照 num_heads × d_k 的方式拆开
        K = K.unflatten(-1, (self.num_heads, self.d_k))
        V = V.unflatten(-1, (self.num_heads, self.d_k))

        # (..., seq_len, heads, d_k)
        # → 倒数第二个和倒数第三个维度交换 对应 scaled_dot_product_attention 的输入
        # (..., heads, seq_len, d_k)
        Q = Q.transpose(-3, -2)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)

        seq_len = x.shape[-2]

        # tril 取下三角矩阵，语言模型不能偷看未来
        # token0 只能看 token0
        # token1 可以看 token0 token1
        # token2 可以看 token0 token1 token2
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)) # device=x.device 让新创建的 Tensor 放到和 x 一样的设备上
        output = scaled_dot_product_attention(Q, K, V, mask)
        # (..., heads, seq_len, d_k)
        # →
        # (..., seq_len, heads, d_k)
        output = output.transpose(-3, -2)

        # heads × d_k 合并回 d_model
        output = output.flatten(-2)
        return self.output_proj(output)

class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
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
        self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
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
            token_positions = torch.arange(seq_len, device=x.device)
        # 为 head 这个维度补一个维度，方便广播
        while token_positions.ndim < Q.ndim - 1:
            token_positions = token_positions.unsqueeze(-2)
        # ④ 只给 Q、K 做 RoPE
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        # ⑤ causal mask：不能看未来 token
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        # ⑥ 每个 head 做 Attention
        output = scaled_dot_product_attention(Q, K, V, mask)
        # ⑦ 把 head 移回去
        output = output.transpose(-3, -2)
        # ⑧ num_heads × d_k → d_model
        output = output.flatten(-2)
        # ⑨ 输出投影
        return self.output_proj(output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一部分：Attention
        x = x + self.attn(self.ln1(x))
        # 第二部分：FFN
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        # 1. token id -> d_model 维向量
        self.token_embeddings = Embedding(vocab_size=vocab_size, d_model=d_model)
        # 2. 堆叠 num_layers 个 Transformer Block
        self.layers = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta)
            for _ in range(num_layers) # nn.ModuleList：PyTorch Module 的列表
        ])
        # 3. 最后再做一次 RMSNorm
        self.ln_final = RMSNorm(d_model)
        # 4. d_model -> vocab_size：把语义向量和词表匹配，映射到分数
        self.lm_head = Linear(d_in=d_model, d_out=vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token id -> embedding
        x = self.token_embeddings(token_ids)
        # 依次经过所有 Transformer Block
        for layer in self.layers:
            x = layer(x)
        # 最终归一化
        x = self.ln_final(x)
        # 映射到整个词表，得到 logits：只是模型打的原始分数，不是概率
        logits = self.lm_head(x)
        return logits