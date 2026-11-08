"""讲义第 30—31 页的 SGD 学习率小实验，由你手动运行观察。

这个实验用“权重平方的平均值”作为 loss，因此目标是让权重接近 0。
语言模型训练时会换成预测下一个 token 的交叉熵损失。
"""

import math
from collections.abc import Callable
from typing import Optional

import torch


# 先运行 10；观察后，分别改成 100、1000，再各运行一次。
LEARNING_RATE = 1000
ITERATIONS = 10


class SGD(torch.optim.Optimizer):
    """讲义给出的 SGD：第 t 次更新的步长为 lr / sqrt(t + 1)。"""

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


def main():
    # 固定随机种子：三次运行从相同的初始权重出发，便于比较学习率。
    torch.manual_seed(336)

    # 创建 10 × 10 个可训练的数字，也就是 100 个权重。
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=LEARNING_RATE)

    print(f"初始学习率：{LEARNING_RATE}；更新次数：{ITERATIONS}")
    print("以下每行是本轮更新之前的 loss 和其中一个权重。")

    for t in range(ITERATIONS):
        # 1. 清掉上一轮的梯度。
        opt.zero_grad()

        # 2. 前向计算 loss：100 个权重分别平方，再取平均。
        loss = (weights**2).mean()
        print(
            f"第 {t + 1:02d} 轮：loss={loss.item():.8g}，"
            f"第一个权重={weights[0, 0].item():.8g}"
        )

        # 3. 反向传播：算出 loss 相对于这些权重的梯度。
        loss.backward()

        # 4. 优化器根据梯度更新权重，下一轮使用新的权重。
        opt.step()


if __name__ == "__main__":
    main()
