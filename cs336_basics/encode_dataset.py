"""将训练/验证文本流式编码为原始 uint16 文件；不会自动启动其他实验。"""

import argparse
from datetime import datetime
from functools import lru_cache
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys
import time

import numpy as np


DATASETS = {
    "tinystories": {
        "vocab_size": 10000,
        "train": "TinyStoriesV2-GPT4-train.txt",
        "valid": "TinyStoriesV2-GPT4-valid.txt",
    },
    "owt": {
        "vocab_size": 32000,
        "train": "owt_train.txt",
        "valid": "owt_valid.txt",
    },
}
SPECIAL_TOKENS = ["<|endoftext|>"]
BATCH_SIZE = 100_000
PROGRESS_SECONDS = 10


def encode_file(tokenizer, source, destination):
    """分块读取、分批写 ID，核对全文还原校验值后才去掉 .part 后缀。"""
    partial = destination.with_suffix(destination.suffix + ".part")
    started = time.perf_counter()
    last_report = started
    token_count = 0
    source_size = source.stat().st_size
    bytes_read = 0
    source_hash = hashlib.sha256()
    restored_hash = hashlib.sha256()
    progress_path = destination.with_suffix(".progress.json")

    def save_progress(status):
        elapsed = time.perf_counter() - started
        progress = {
            "status": status,
            "source": str(source),
            "output": str(destination),
            "source_bytes": source_size,
            "bytes_read": bytes_read,
            "percent_read": round(100 * bytes_read / source_size, 3) if source_size else 100,
            "token_count": token_count,
            "backend": getattr(tokenizer, "backend", "python"),
            "elapsed_seconds": elapsed,
            "updated_at": datetime.now().astimezone().isoformat(),
        }
        temporary = progress_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(progress_path)

    # 固定大小分块，避免网页中的特别长行一次占用太多内存。
    with source.open("r", encoding="utf-8", newline="") as input_file:
        def chunks():
            nonlocal bytes_read
            while True:
                chunk = input_file.read(1024 * 1024 if hasattr(tokenizer, "iter_encoded_batches") else 64 * 1024)
                if not chunk:
                    return
                raw = chunk.encode("utf-8")
                bytes_read += len(raw)
                source_hash.update(raw)
                yield chunk

        save_progress("in_progress")
        if hasattr(tokenizer, "iter_encoded_batches"):
            batches = tokenizer.iter_encoded_batches(chunks())
        else:
            token_iterator = iter(tokenizer.encode_iterable(chunks()))

            def python_batches():
                while True:
                    batch = list(islice(token_iterator, BATCH_SIZE))
                    if not batch:
                        return
                    yield batch

            batches = python_batches()
        with partial.open("xb") as output_file:
            for batch in batches:
                values = np.asarray(batch)
                if not values.size:
                    continue
                # 转换前检查范围，避免 uint16 溢出导致 ID 损坏。
                if values.min() < 0 or values.max() > 65535:
                    raise ValueError("发现 uint16 无法表示的 token ID。")
                np.asarray(values, dtype="<u2").tofile(output_file)
                if hasattr(tokenizer, "decode_bytes"):
                    restored_hash.update(tokenizer.decode_bytes(values))
                else:
                    restored_hash.update(b"".join(tokenizer.vocab[token_id] for token_id in batch))
                token_count += len(batch)

                now = time.perf_counter()
                if now - last_report >= PROGRESS_SECONDS:
                    print(
                        f"{source.name}：已读取约 {100 * bytes_read / source_size:.1f}%，{token_count:,} tokens，"
                        f"已用 {now - started:.1f} 秒",
                        flush=True,
                    )
                    last_report = now
                    save_progress("in_progress")

    if bytes_read != source_size or source_hash.digest() != restored_hash.digest():
        save_progress("verification_failed")
        raise RuntimeError(f"全文还原校验失败，暂不生成正式结果：{source}")
    if partial.stat().st_size != token_count * 2:
        raise RuntimeError(f"编号文件大小与 uint16 数量不一致：{partial}")
    partial.rename(destination)
    elapsed = time.perf_counter() - started
    save_progress("complete")
    print(f"完成：{destination}", flush=True)
    print(f"token_count={token_count:,}，耗时 {elapsed:.1f} 秒", flush=True)
    return {
        "path": destination.name,
        "token_count": token_count,
        "source_bytes": source_size,
        "elapsed_seconds": elapsed,
        "bytes_per_second": source_size / elapsed if elapsed else None,
        "source_sha256": source_hash.hexdigest(),
        "decoded_sha256": restored_hash.hexdigest(),
        "roundtrip_verified": True,
    }


def main():
    parser = argparse.ArgumentParser(description="流式编码 TinyStories 或 OpenWebText")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--split", choices=["train", "valid", "all"], default="all")
    parser.add_argument("--backend", choices=["auto", "python", "native"], default="auto")
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument(
        "--vocab-dir",
        type=Path,
        help="指定含 vocab.pkl、merges.pkl 的目录；相对路径基于项目根目录",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="项目根目录；默认是本脚本所在目录的上一级",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        parser.error(f"项目根目录不存在：{project_root}")

    settings = DATASETS[args.dataset]
    vocab_size = settings["vocab_size"]
    data_dir = project_root / "data"
    # all 先完成验证集；开始前仍检查本次需要的全部输入。
    splits = ["valid", "train"] if args.split == "all" else [args.split]
    sources = {split: data_dir / settings[split] for split in splits}
    missing_inputs = [str(path) for path in sources.values() if not path.is_file()]
    if missing_inputs:
        parser.error("找不到输入文件：\n" + "\n".join(missing_inputs))

    if args.vocab_dir is not None:
        vocab_dir = args.vocab_dir
        if not vocab_dir.is_absolute():
            vocab_dir = project_root / vocab_dir
    elif args.dataset == "tinystories":
        vocab_dir = data_dir / "tinystories_bpe_10000"
    else:
        base = data_dir / "owt_bpe_32000"
        candidates = [
            path
            for path in base.iterdir()
            if path.is_dir()
            and (path / "vocab.pkl").is_file()
            and (path / "merges.pkl").is_file()
        ] if base.is_dir() else []
        if not candidates:
            parser.error(f"未找到完整的 OpenWebText 词表目录：{base}")
        # 已有训练脚本以时间戳命名子目录，名称排序即运行时间顺序。
        vocab_dir = max(candidates, key=lambda path: path.name)

    vocab_dir = vocab_dir.resolve()
    vocab_path = vocab_dir / "vocab.pkl"
    merges_path = vocab_dir / "merges.pkl"
    if not vocab_path.is_file() or not merges_path.is_file():
        parser.error(f"词表目录必须同时包含 vocab.pkl 和 merges.pkl：{vocab_dir}")

    # 指定 project-root 后，本示例也可从仓库外运行。
    sys.path.insert(0, str(project_root))
    from cs336_basics.tokenizer import Tokenizer

    tokenizer = Tokenizer.from_files(
        str(vocab_path), str(merges_path), special_tokens=SPECIAL_TOKENS
    )
    # 全量任务使用更大的有界缓存，重复片段不必反复合并。
    tokenizer.cached_piece = lru_cache(maxsize=50000)(tokenizer._encode_ordinary_piece)
    if args.backend != "python":
        try:
            from cs336_basics.fast_tokenizer import FastTokenizer
            tokenizer = FastTokenizer(tokenizer, workers=args.workers)
        except (ImportError, ValueError) as error:
            if args.backend == "native":
                raise
            print(f"当前词表保留 Python 后端：{error}", flush=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = data_dir / f"{args.dataset}_tokens_{vocab_size}" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = output_dir / "metadata.json"

    print(f"数据集：{args.dataset}；处理顺序：{', '.join(splits)}", flush=True)
    for split, source in sources.items():
        print(f"{split} 输入：{source}", flush=True)
    print(f"词表来源：{vocab_dir}", flush=True)
    print(f"编码后端：{getattr(tokenizer, 'backend', 'python')}", flush=True)
    print(f"编码线程：{getattr(tokenizer, 'workers', 1)}", flush=True)
    print(f"输出目录：{output_dir}", flush=True)

    metadata = {
        "dataset": args.dataset,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "in_progress",
        "dtype": "<u2",
        "format": "raw binary, no header",
        "text_encoding": "utf-8",
        "newline": "",
        "vocab_dir": str(vocab_dir),
        "special_tokens": SPECIAL_TOKENS,
        "backend": getattr(tokenizer, "backend", "python"),
        "backend_version": getattr(tokenizer, "backend_version", sys.version.split()[0]),
        "encoding_workers": getattr(tokenizer, "workers", 1),
        "vocab_sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
        "merges_sha256": hashlib.sha256(merges_path.read_bytes()).hexdigest(),
        "sources": {split: str(path) for split, path in sources.items()},
        "token_count": 0,
        "files": {},
    }

    def save_metadata():
        temporary = metadata_path.with_suffix(".json.part")
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(metadata_path)

    save_metadata()
    for split, source in sources.items():
        destination = output_dir / f"{split}.bin"
        result = encode_file(tokenizer, source, destination)
        metadata["files"][split] = result
        metadata["token_count"] += result["token_count"]
        save_metadata()

    metadata["status"] = "complete"
    save_metadata()
    print(f"全部完成：{metadata['token_count']:,} tokens", flush=True)
    print(f"元数据：{metadata_path}", flush=True)


if __name__ == "__main__":
    main()
