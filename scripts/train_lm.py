from pathlib import Path

import numpy as np
import torch

import time

from cs336_basics.data import get_batch
from cs336_basics.layers import TransformerLM, cross_entropy
from cs336_basics.optimizers import AdamW, gradient_clipping

def evaluate_loss(model, batches):
    # 记住原来的模式，然后切换到评估模式。
    previous_mode = model.training
    model.eval()

    total_loss = 0.0

    # 验证时不建立求梯度所需的计算记录，节省内存。
    with torch.no_grad():
        for inputs, targets in batches:
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            total_loss += loss.item()

   

    # 检查完，恢复原来的模式。
    model.train(previous_mode)

    return total_loss / len(batches)

def main():
    # 1. 设置运行方式
    # 先使用 4 个 CPU 计算线程，实际速度运行后再观察。
    torch.set_num_threads(4)

    # 固定随机种子，让重复运行时的初始权重、抽样更容易对照。
    torch.manual_seed(336)
    np.random.seed(336)

    device = "cpu"
    batch_size = 2
    context_length = 64
    total_steps = 30

    # 2. 找到并打开已经编码好的数据
    project_root = Path(__file__).resolve().parents[1]
    data_dir = (
        project_root
        / "data"
        / "tinystories_tokens_10000"
        / "20261024_173852_046637"
    )

    train_data = np.memmap(
        data_dir / "train.bin",
        dtype="<u2",
        mode="r",
    )
    valid_data = np.memmap(
        data_dir / "valid.bin",
        dtype="<u2",
        mode="r",
    )

    # 3. 创建一个新的、随机初始化的语言模型
    model = TransformerLM(
        vocab_size=10000,
        context_length=context_length,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=384,
        rope_theta=10000,
        device=device,
        dtype=torch.float32,
    )

    # 4. 创建优化器，让它负责更新上面这个模型的权重
    optimizer = AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    # 5. 确认准备结果
    parameter_count = sum(p.numel() for p in model.parameters())

    print(f"训练集 token 数：{len(train_data):,}")
    print(f"验证集 token 数：{len(valid_data):,}")
    print(f"模型参数量：{parameter_count:,}")
    print(f"运行设备：{device}")
    print(f"每步：{batch_size} 段，每段 {context_length} 个 token")
    print(f"计划训练：{total_steps} 步")
    print("准备完成，目前还没有更新模型权重。")
    
     # 固定抽取 5 个验证批次，训练前后重复使用。
    validation_batches = []

    # 保存抽样状态，避免验证抽样改变原来的训练抽样顺序。
    random_state = np.random.get_state()

    for _ in range(5):
        batch = get_batch(
            valid_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        validation_batches.append(batch)

    np.random.set_state(random_state)

    # 在模型还没有开始学习之前，先测一次。
    initial_val_loss = evaluate_loss(model, validation_batches)

    print(
        f"训练前 | val_loss={initial_val_loss:.4f}",
        flush=True,
    )

        # 切换到训练模式；真正的训练由下面的计算和更新完成。
    model.train()

    print("开始训练……", flush=True)

    for step in range(1, total_steps + 1):
        step_start = time.perf_counter()

        # ① 随机抽两段，每段构造 64 个输入和对应答案。
        inputs, targets = get_batch(
            train_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # ② 清空上一步的梯度，准备计算本次梯度。
        optimizer.zero_grad(set_to_none=True)

        # ③ 模型根据输入，为每个位置输出整个词表的预测分数。
        logits = model(inputs)

        # ④ 对照真实的下一个 token，计算平均损失。
        loss = cross_entropy(logits, targets)

        if not torch.isfinite(loss).item():
            raise RuntimeError("loss 出现 NaN 或无穷大，停止训练。")

        # ⑤ 根据 loss，计算各个权重的梯度。
        loss.backward()

        # ⑥ 限制整体梯度过大，然后真正更新权重。
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        optimizer.step()

        elapsed = time.perf_counter() - step_start

        print(
            f"第 {step:02d}/{total_steps} 步 | "
            f"train_loss={loss.item():.4f} | "
            f"耗时={elapsed:.2f} 秒",
            flush=True,
        )

    print("这轮 30 步训练结束。", flush=True)
    # 用更新后的模型，重新回答刚才那批验证题。
    final_val_loss = evaluate_loss(model, validation_batches)

    print(
        f"训练后 | val_loss={final_val_loss:.4f}",
        flush=True,
    )
    print(
        f"验证 loss 变化："
        f"{initial_val_loss:.4f} → {final_val_loss:.4f}",
        flush=True,
    )

if __name__ == "__main__":
    main()