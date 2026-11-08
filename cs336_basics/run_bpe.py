"""把本文件放在 cs336_basics 目录，用 --dataset 选择训练数据。"""

import argparse
import cProfile
from datetime import datetime
from pathlib import Path
import pickle
import pstats
import time

import psutil

from cs336_basics.bpe_train import train_bpe


# 两套任务配置都保留，共用同一个 train_bpe 函数。
DATASETS = {
    "tinystories": ("TinyStoriesV2-GPT4-train.txt", 10000),
    "owt": ("owt_train.txt", 32000),
}


def main():
    # 从启动命令中读取数据集名称。
    parser = argparse.ArgumentParser(description="选择数据集，训练并保存 BPE 词表。")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--profile", action="store_true", help="额外记录各函数耗时")
    args = parser.parse_args()

    filename, vocab_size = DATASETS[args.dataset]
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / filename
    if not input_path.is_file():
        parser.error(f"找不到数据文件：{input_path}")

    # 每次运行单独保存，保留以前的训练结果。
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        project_root / "data" / f"{args.dataset}_bpe_{vocab_size}" / run_name
    )
    output_dir.mkdir(parents=True)

    print("训练文件：", input_path, flush=True)
    print("目标词表大小：", vocab_size, flush=True)
    print("结果目录：", output_dir, flush=True)

    # 调用已经写好的训练函数。
    profiler = cProfile.Profile() if args.profile else None
    start_time = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    try:
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
        )
    finally:
        if profiler is not None:
            profiler.disable()
    elapsed = time.perf_counter() - start_time
    peak_memory = psutil.Process().memory_info().peak_wset / (1024 ** 3)

    # 两套数据的结果分别保存在各自目录下。
    with (output_dir / "vocab.pkl").open("wb") as file:
        pickle.dump(vocab, file)
    with (output_dir / "merges.pkl").open("wb") as file:
        pickle.dump(merges, file)

    longest_token = max(vocab.values(), key=len)
    report = (
        f"训练文件：{input_path}\n"
        f"词表大小：{len(vocab)}\n"
        f"合并次数：{len(merges)}\n"
        f"性能分析：{'开启，耗时包含分析开销' if args.profile else '关闭'}\n"
        f"训练耗时：{elapsed:.2f} 秒\n"
        f"进程物理内存峰值：{peak_memory:.2f} GiB\n"
        f"最长 token 的字节数：{len(longest_token)}\n"
        f"最长 token 的原始字节：{longest_token!r}\n"
        f"尝试显示为文字：{longest_token.decode('utf-8', errors='replace')!r}\n"
    )
    (output_dir / "results.txt").write_text(report, encoding="utf-8")

    if profiler is not None:
        profiler.dump_stats(str(output_dir / "profile.prof"))
        with (output_dir / "profile.txt").open("w", encoding="utf-8") as file:
            stats = pstats.Stats(profiler, stream=file).strip_dirs()
            stats.sort_stats("cumulative").print_stats(20)
            stats.sort_stats("tottime").print_stats(20)

    print("\n实验结果：\n" + report)
    print("结果已保存到：", output_dir)


if __name__ == "__main__":
    main()
