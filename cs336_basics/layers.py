import math

import torch
from torch import nn


class Linear(nn.Module):
    """无偏置的线性层，将输入向量转换成指定维度的输出向量。"""

    def __init__(
        self,
        in_features,
        out_features,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # 每一行权重负责计算一个输出数字，训练时会调整这些权重。
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        # 按讲义要求设置初始权重：均值为 0，截断范围为 ±3 个标准差。
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x):
        # 只改变最后一个维度，支持同时处理一批 token 的向量。
        return x @ self.weight.T


class Embedding(nn.Module):
    """根据 token ID，从可训练的向量表中取出对应行。"""

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # 每个 token 占一行，每一行有 embedding_dim 个数字。
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )

        # 按讲义要求，给表里填入初始随机数。
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids):
        # 根据 token ID，取出对应行的向量。
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """按每个 token 向量的均方根缩放，再乘上可训练的 gain。"""

    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps

        # 每个向量维度都有一个可训练的缩放参数 g，初始值全部为 1。
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        # 先用 float32 计算，避免低精度输入在平方时溢出。
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # 只对每个 token 自己的向量求平均，保留最后一维方便逐个相除。
        mean_square = x.square().mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)

        # 每个数字除以所属向量的 RMS，再乘对应维度的缩放参数 g。
        normalized = x / rms
        result = normalized * self.weight.to(torch.float32)

        # 输出恢复为输入原来的数据类型，形状保持不变。
        return result.to(in_dtype)


def silu(x):
    """每个数字乘上它自己经过 sigmoid 后得到的系数。"""
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """沿指定维度，将分数转换成总和为 1 的比例。"""
    in_dtype = x.dtype
    # 低精度输入先用 float32 计算，避免指数求和时溢出或损失过多精度。
    # float64 输入保留原来的精度。
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.to(torch.float32)

    # 每组都减去自己的最大值，让最大的指数输入为 0。
    shifted = x - x.amax(dim=dim, keepdim=True)
    exp_values = torch.exp(shifted)

    # 每个数除以本组的总和，保留维度以便对应相除。
    probabilities = exp_values / exp_values.sum(dim=dim, keepdim=True)
    return probabilities.to(in_dtype)


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """根据原始分数和正确 token 编号，计算所有预测位置的平均损失。

    logits 的形状为 (..., vocab_size)，targets 为对应的 (...) 整数编号。
    """
    # 低精度输入先用 float32 计算，float64 输入保留原来的精度。
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()

    # 每组减去自己的最大值，保持概率不变，同时避免指数溢出。
    shifted = logits - logits.amax(dim=-1, keepdim=True)

    # 将 Softmax 和负对数合并计算，避免对接近 0 的概率直接取对数。
    log_total = torch.log(torch.exp(shifted).sum(dim=-1))
    correct_scores = shifted.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # 每个位置的损失 = 指数总和的对数 - 正确答案的分数。
    losses = log_total - correct_scores
    return losses.mean()


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """比较 Q、K 得到关注比例，再用这些比例汇总 V 中的信息。

    Q、K、V 的形状分别为 (..., queries, d_k)、(..., keys, d_k)、
    (..., keys, d_v)。布尔 mask 可广播到 (..., queries, keys)，True 表示允许关注。
    """
    in_dtype = Q.dtype
    # 低精度输入先提升精度，避免点积在进入 Softmax 之前就溢出。
    if Q.dtype in (torch.float16, torch.bfloat16):
        Q, K, V = Q.float(), K.float(), V.float()

    # 只交换 K 的最后两个维度，一次比较整批位置，并按向量维度缩放。
    scores = (Q @ K.transpose(-1, -2)) / math.sqrt(Q.shape[-1])

    # 不允许关注的位置在 Softmax 之前设为负无穷，分配到的比例就为 0。
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # 沿每个 query 对应的所有 key 分配比例，再加权汇总它们的 V。
    attention_weights = softmax(scores, dim=-1)
    return (attention_weights @ V).to(in_dtype)


class SwiGLU(nn.Module):
    """两路 Linear 的结果经过 SiLU 和逐元素相乘，再由输出 Linear 组合。"""

    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()

        # 未指定中间维度时，按讲义的 8/3 倍向上取到 64 的倍数。
        # 显式指定 d_ff 时保留传入值，便于加载老师给定的权重。
        if d_ff is None:
            d_ff = math.ceil((8 / 3) * d_model / 64) * 64

        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        # 同一份输入走两条分支，各使用自己的一套 Linear 权重。
        gate = silu(self.w1(x))
        values = self.w3(x)

        # 对应位置相乘；gate 是 SiLU 输出，不局限在 0～1 之间。
        combined = gate * values

        # 最后重新组合，输出恢复到原来的向量维度。
        return self.w2(combined)


class RotaryPositionalEmbedding(nn.Module):
    """按 token 的位置，旋转向量中相邻的每一对数字。"""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k must be a positive even number")
        if theta <= 0 or not math.isfinite(theta):
            raise ValueError("theta must be positive and finite")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.d_k = d_k

        # 每对数字使用不同的旋转速度；越后面的数字对，旋转越慢。
        pair_dimensions = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inverse_frequencies = theta ** (-pair_dimensions / d_k)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = positions[:, None] * inverse_frequencies[None, :]

        # 固定的查找表会随模块移动设备，但不参与训练，也不写入权重文件。
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x 的形状为 (..., 序列长度, d_k)。位置形状需能广播到 x.shape[:-1]。
        # 所有样本共用位置时可传 [序列长度]；按样本指定时可传 [批量, 序列长度]。
        # 多头输入 [批量, 头数, 序列长度, d_k] 可传 [批量, 1, 序列长度]。
        if x.shape[-1] != self.d_k:
            raise ValueError(f"Expected last dimension {self.d_k}, got {x.shape[-1]}")

        # 用实际位置查表；例如位置从 5 开始，就应从表的第 5 行取值。
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # 把 (第 0、1 个数)、(第 2、3 个数)……分别看成一对。
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # 将两路结果交错放回原顺序，保持输入的形状和数据类型。
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2).to(x.dtype)


class MultiHeadSelfAttention(nn.Module):
    """生成 Q、K、V，按多个头进行因果注意力计算，再合并各头的信息。"""

    def __init__(
        self,
        d_model,
        num_heads,
        max_seq_len=None,
        theta=None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # 1. 检查能否平均分头，并记住各项设置。
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model 必须为正数，并且能够被 num_heads 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 2. 每次创建 Linear，都会得到一套独立初始化、可训练的权重。
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 3. 传入 theta 时启用 RoPE，旋转维度是每个头的向量长度。
        self.rope = None
        if theta is not None:
            if max_seq_len is None:
                raise ValueError("使用 RoPE 时需要提供 max_seq_len")
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.head_dim,
                max_seq_len=max_seq_len,
                device=device,
            )

    def forward(self, x, token_positions=None):
        # 4. 输入形状为 (..., token 数, 完整向量长度)。
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 5. 先拆开向量，再交换轴，得到 (..., 头数, token 数, 每头维度)。
        # 例如 [2, 3, 8] -> [2, 3, 2, 4] -> [2, 2, 3, 4]。
        head_shape = (*batch_shape, seq_len, self.num_heads, self.head_dim)
        q = q.reshape(head_shape).transpose(-3, -2)
        k = k.reshape(head_shape).transpose(-3, -2)
        v = v.reshape(head_shape).transpose(-3, -2)

        # 6. 位置在各个头之间共享；只旋转 Q、K，不旋转 V。
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            # [批量, token 数] -> [批量, 1, token 数]，避免批量与头数错位。
            rope_positions = token_positions.unsqueeze(-2)
            q = self.rope(q, rope_positions)
            k = self.rope(k, rope_positions)

        # 7. 下三角为 True：每个位置只允许关注自己和前面的位置。
        mask = torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=x.device
        ).tril()

        # 8. 用已有的注意力函数，一次处理全部批量和全部头。
        head_outputs = scaled_dot_product_attention(q, k, v, mask=mask)

        # 9. 先交换回 token、头两个轴，再把各头的结果拼回完整向量。
        merged = head_outputs.transpose(-3, -2).reshape(
            *batch_shape, seq_len, self.d_model
        )
        return self.output_proj(merged)


class TransformerBlock(nn.Module):
    """先归一化再计算，并在注意力和前馈网络之后分别加回输入。"""

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        max_seq_len,
        theta,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # 两段各有一套独立的 RMSNorm，缩放参数互不共享。
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        # 第一段：调整尺度 -> 汇总上下文 -> 加回这一段最初的输入。
        attention_output = self.attn(self.ln1(x), token_positions=token_positions)
        x = x + attention_output

        # 第二段以更新后的 x 为输入：调整尺度 -> SwiGLU -> 再加回输入。
        feedforward_output = self.ffn(self.ln2(x))
        return x + feedforward_output


class TransformerLM(nn.Module):
    """将 token ID 转成向量，经多个 Block 后输出整个词表的预测分数。"""

    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if context_length <= 0 or num_layers <= 0:
            raise ValueError("context_length 和 num_layers 必须为正数")
        self.context_length = context_length

        # 1. 每个 token ID 查表，得到 d_model 维向量。
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )

        # 2. 每次循环都新建一个 Block，各层拥有独立权重。
        # ModuleList 让 PyTorch 正确登记、移动和保存这些层的参数。
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # 3. 最后再调整一次尺度，并给整个词表中的每个 token 计算分数。
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids, token_positions=None):
        seq_len = token_ids.shape[-1]
        if not 1 <= seq_len <= self.context_length:
            raise ValueError(f"输入 token 数必须在 1 到 {self.context_length} 之间")
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=token_ids.device)

        # (..., token 数) -> (..., token 数, 向量长度)。
        x = self.token_embeddings(token_ids)
        for block in self.layers:
            x = block(x, token_positions=token_positions)

        x = self.ln_final(x)
        # 返回 logits 原始分数；训练损失或生成采样在后续步骤处理。
        return self.lm_head(x)
