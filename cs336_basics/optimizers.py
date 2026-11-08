"""训练用优化器，按照讲义的更新公式实现。"""

import math
from collections.abc import Iterable

import torch


class AdamW(torch.optim.Optimizer):
    """AdamW 优化器，适用于当前使用实数、float32 权重的课程模型。"""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        beta1, beta2 = betas

        if lr < 0:
            raise ValueError("学习率不能为负数")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("beta1 和 beta2 必须在 [0, 1) 范围内")
        if eps <= 0:
            raise ValueError("eps 必须大于 0")
        if weight_decay < 0:
            raise ValueError("weight_decay 不能为负数")

        # 父类负责保存参数分组，并提供 zero_grad 和 state_dict 等接口。
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        # 兼容 PyTorch 可选的“重新计算 loss”接口。
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # p 是一整块权重张量；下面的运算同时处理其中所有数字。
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("这个 AdamW 实现不支持稀疏梯度")

                # 每块权重分别保存自己的更新次数，以及两份历史记录。
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]

                # 1. 独立于梯度更新，使用原始学习率做权重衰减。
                p.mul_(1 - lr * weight_decay)

                # 2. m 记录梯度的历史平均，v 记录梯度平方的历史平均。
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 3. 修正历史记录从 0 开始造成的初期偏小。
                adjusted_lr = (
                    lr
                    * math.sqrt(1 - beta2**t)
                    / (1 - beta1**t)
                )

                # 4. 按讲义公式，将 eps 加在未校正的 sqrt(v) 后。
                # sqrt() 创建新张量，因此这里不会改掉历史记录 v。
                denominator = v.sqrt().add_(eps)
                p.addcdiv_(m, denominator, value=-adjusted_lr)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """根据从 0 开始的训练步数，返回预热与余弦下降后的学习率。

    warmup_iters 是预热结束的步数；cosine_cycle_iters 是余弦下降
    结束的总步数，不是预热之后再执行的步数。允许预热步数为 0。
    """
    if it < 0:
        raise ValueError("当前训练步数不能为负数")
    if not 0 <= min_learning_rate <= max_learning_rate:
        raise ValueError("学习率必须满足 0 <= 最小值 <= 最大值")
    if warmup_iters < 0 or cosine_cycle_iters <= warmup_iters:
        raise ValueError("预热步数不能为负数，余弦结束步数必须大于预热步数")

    # 阶段 1：预热，按照完成比例从 0 升到最大学习率。
    # 预热步数为 0 时直接跳过，因此不会出现除以 0。
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    # 阶段 3：下降结束后，学习率保持在最小值。
    if it > cosine_cycle_iters:
        return min_learning_rate

    # 阶段 2：进度从 0 变到 1，余弦系数从 1 平滑降到 0。
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)


@torch.no_grad()
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """按所有参数梯度的全局 L2 范数，原地缩放梯度，保持权重不变。"""
    if max_l2_norm < 0:
        raise ValueError("梯度范数上限不能为负数")

    # model.parameters() 可能是只能遍历一次的生成器，先收集有效梯度。
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    # 每块矩阵先求范数，再汇总成全局范数，避免拼接所有大梯度。
    norms = []
    for grad in grads:
        if grad.is_sparse:
            raise RuntimeError("这个梯度裁剪实现不支持稀疏梯度")
        stable_grad = (
            grad.float()
            if grad.dtype in (torch.float16, torch.bfloat16)
            else grad
        )
        norms.append(torch.linalg.vector_norm(stable_grad, ord=2))

    norm_device = norms[0].device
    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(device=norm_device) for norm in norms]),
        ord=2,
    )

    # 低于上限时保持不变；否则所有梯度都乘同一个比例。
    scale = torch.where(
        total_norm < max_l2_norm,
        torch.ones_like(total_norm),
        max_l2_norm / (total_norm + 1e-6),
    )
    for grad in grads:
        grad.mul_(scale.to(device=grad.device))
