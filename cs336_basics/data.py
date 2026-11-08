"""从已编码的 token 序列中，抽取语言模型训练用的输入和目标。"""

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回形状均为 (batch_size, context_length) 的输入及下一 token 目标。

    dataset 可以是普通一维整数数组，也可以是只读的 np.memmap。
    每次只读取抽中的窗口，不把整份数据转成内存数组。
    """
    if dataset.ndim != 1:
        raise ValueError("token 数据必须是一维数组")
    if dataset.dtype.kind not in "iu":
        raise TypeError("token 数据必须使用整数类型")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size 和 context_length 必须大于 0")
    if len(dataset) <= context_length:
        raise ValueError("数据至少需要包含 context_length + 1 个 token")

    # 每个窗口比输入多读一个 token，才能给最后一个位置配上答案。
    # randint 的上限不包含在结果中；最后一个合法起点也能被取到。
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    positions = starts[:, None] + np.arange(context_length + 1)[None, :]

    # 只将这个小批次转换成 int64，兼容磁盘文件里的 uint16 token。
    windows = np.asarray(dataset[positions], dtype=np.int64)

    # 比如窗口 [10, 20, 30, 40]：输入 [10, 20, 30]，目标 [20, 30, 40]。
    # 分别复制成连续数组，方便后面展开维度，同时避免修改原始数据。
    inputs = torch.from_numpy(windows[:, :-1].copy()).to(device=device)
    targets = torch.from_numpy(windows[:, 1:].copy()).to(device=device)
    return inputs, targets
