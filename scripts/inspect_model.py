"""按讲义的 TinyStories 基准配置统计模型参数，不进行训练。"""

import torch

from cs336_basics.layers import TransformerLM


def main():
    # 只在 CPU 上创建模型，所有权重使用 float32，每个数字占 4 字节。
    model = TransformerLM(
        vocab_size=10000,       # 词表有 10000 个 token
        context_length=256,    # 每段最多处理 256 个 token
        d_model=512,           # 每个 token 的向量有 512 个数字
        num_layers=4,          # 串联 4 个独立的 Block
        num_heads=16,          # 每个 Block 使用 16 个注意力头
        d_ff=1344,             # 按讲义显式指定 SwiGLU 的中间维度
        rope_theta=10000,
        device="cpu",
        dtype=torch.float32,
    )

    # named_children() 依次列出模型的四个大组件。
    # parameters() 取出权重表，numel() 数出表里有多少个数字。
    print("各组件参数量：")
    for name, part in model.named_children():
        count = sum(parameter.numel() for parameter in part.parameters())
        print(f"{name}: {count:,}")

    total_params = sum(parameter.numel() for parameter in model.parameters())
    weight_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    print(f"\n参数总数：{total_params:,}")
    print(f"权重占用：{weight_bytes / 1024**2:.2f} MiB")
    print("这里只统计权重，不包括梯度、优化器状态、RoPE 缓存和中间计算结果。")


if __name__ == "__main__":
    main()
