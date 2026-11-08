import pickle
import time
from collections import Counter
from pathlib import Path
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def iter_documents(
    input_path,
    special_tokens,
    chunk_size=1024 * 1024,
):
    with open(input_path, "r", encoding="utf-8") as file:
        if not special_tokens:
            yield file.read()
            return

        ordered_tokens = sorted(
            set(special_tokens),
            key=lambda token: (-len(token), token),
        )

        separator = re.compile(
            "|".join(
                re.escape(token)
                for token in ordered_tokens
            )
        )

        longest = max(len(token) for token in ordered_tokens)
        buffer = ""

        while True:
            # 每次只读取一部分文字。
            chunk = file.read(chunk_size)
            finished = chunk == ""
            buffer += chunk

            # 末尾可能有不完整的特殊标记，暂时留着。
            if finished:
                safe_start = len(buffer)
            else:
                safe_start = max(
                    0,
                    len(buffer) - longest + 1,
                )

            consumed = 0

            for match in separator.finditer(buffer):
                if not finished and match.start() >= safe_start:
                    break

                # 交出一篇完整文档，不包含分隔标记。
                yield buffer[consumed:match.start()]
                consumed = match.end()

            # 剩余文字接到下一块前面。
            buffer = buffer[consumed:]

            if finished:
                yield buffer
                return


def train_bpe(input_path, vocab_size, special_tokens):
    counts = Counter()
    documents_seen = 0

    print("开始分块读取并统计……", flush=True)

    for document in iter_documents(input_path, special_tokens):
        for match in re.finditer(PAT, document):
            piece = match.group()
            counts[piece] += 1

        documents_seen += 1

        if documents_seen % 10000 == 0:
            print(
                f"已处理 {documents_seen:,} 个文档片段",
                flush=True,
            )

    print("不同片段的数量：", len(counts))
    print("出现次数最多的 10 种片段：")

    for piece, count in counts.most_common(10):
        print(repr(piece), count)


    byte_counts = Counter()

    for piece, count in counts.items():
        encoded = piece.encode("utf-8")

        tokens = []

        for value in encoded:
            token = bytes([value])
            tokens.append(token)

        byte_counts[tuple(tokens)] += count


    words = list(byte_counts.keys())
    frequencies = list(byte_counts.values())

    # 相邻对的总次数。
    pair_counts = Counter()

    # 每个相邻对出现在哪些片段中。
    pair_words = {}

    for word_id, tokens in enumerate(words):
        # 统计当前片段内部的相邻对。
        local_counts = Counter(zip(tokens, tokens[1:]))

        for pair, occurrences in local_counts.items():
            # 片段内部的次数 × 片段在原文中的次数。
            pair_counts[pair] += (
                occurrences * frequencies[word_id]
            )

            # 记下这个相邻对出现在哪个片段中。
            pair_words.setdefault(pair, set()).add(word_id)

    print("出现次数最多的 10 种相邻对：")

    for pair, count in pair_counts.most_common(10):
        print(pair, count)


    vocab = {}

    # 放入全部 256 种单字节 token。
    for value in range(256):
        vocab[value] = bytes([value])

    # 加入特殊标记。
    for special_token in special_tokens:
        token_bytes = special_token.encode("utf-8")

        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes

    # 按先后顺序记录合并规则。
    merges = []

    def merge_word(tokens, pair):
        result = []
        position = 0

        while position < len(tokens):
            if (
                position + 1 < len(tokens)
                and tokens[position] == pair[0]
                and tokens[position + 1] == pair[1]
            ):
                result.append(pair[0] + pair[1])
                position += 2
            else:
                result.append(tokens[position])
                position += 1

        return tuple(result)

    while len(vocab) < vocab_size and pair_counts:
        # 1. 选择当前出现次数最多的相邻对。
        # 次数相同，就选择字典序更大的那一对。
        best_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], pair),
        )

        # 2. 保存合并规则，并把新 token 加入词表。
        merges.append(best_pair)

        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        # 3. 找出包含这一对的片段。
        affected_words = list(pair_words[best_pair])

        for word_id in affected_words:
            old_tokens = words[word_id]
            new_tokens = merge_word(old_tokens, best_pair)

            # 当前片段合并前后的相邻对统计。
            old_pairs = Counter(
                zip(old_tokens, old_tokens[1:])
            )
            new_pairs = Counter(
                zip(new_tokens, new_tokens[1:])
            )

            involved_pairs = old_pairs.keys() | new_pairs.keys()

            for pair in involved_pairs:
                change = new_pairs[pair] - old_pairs[pair]

                # 4. 更新整个数据集里的相邻对次数。
                if change:
                    pair_counts[pair] += (
                        change * frequencies[word_id]
                    )

                    if pair_counts[pair] == 0:
                        del pair_counts[pair]

                # 5. 更新相邻对出现在哪些片段中的记录。
                if old_pairs[pair] and not new_pairs[pair]:
                    pair_words[pair].discard(word_id)

                    if not pair_words[pair]:
                        del pair_words[pair]

                elif new_pairs[pair] and not old_pairs[pair]:
                    pair_words.setdefault(pair, set()).add(word_id)

            # 6. 保存当前片段合并后的新表示。
            words[word_id] = new_tokens

        if len(merges) % 100 == 0:
            print(
                f"已合并 {len(merges)} 次，"
                f"当前词表大小：{len(vocab)}"
            )

    print("BPE 训练结束")
    print("最终词表大小：", len(vocab))
    print("合并次数：", len(merges))
    print("最先学到的 10 条规则：", merges[:10])

    return vocab, merges

if __name__ == "__main__":
    import cProfile
    import pstats
    import psutil

    # 一、确定训练文件和结果保存位置。
    project_root = Path(__file__).resolve().parent.parent

    input_path = (
        project_root / "data/TinyStoriesV2-GPT4-train.txt"
    )

    output_dir = project_root / "data/tinystories_bpe_10000"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 二、开始记录时间，并开启性能分析。
    profiler = cProfile.Profile()
    start_time = time.perf_counter()

    vocab, merges = profiler.runcall(
        train_bpe,
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    elapsed = time.perf_counter() - start_time

    # Windows 下，读取当前进程用过的最高物理内存。
    peak_memory = (
        psutil.Process().memory_info().peak_wset
        / (1024 ** 3)
    )

    # 三、保存词表和合并规则，后面可以重新加载使用。
    with open(output_dir / "vocab.pkl", "wb") as file:
        pickle.dump(vocab, file)

    with open(output_dir / "merges.pkl", "wb") as file:
        pickle.dump(merges, file)

    # 四、找出词表中占用字节数最多的 token。
    longest_token = max(vocab.values(), key=len)

    report = (
        f"词表大小：{len(vocab)}\n"
        f"合并次数：{len(merges)}\n"
        f"训练耗时（含性能分析开销）：{elapsed:.2f} 秒\n"
        f"进程物理内存峰值：{peak_memory:.2f} GiB\n"
        f"最长 token 的字节数：{len(longest_token)}\n"
        f"最长 token 的原始字节：{longest_token!r}\n"
        f"尝试显示为文字："
        f"{longest_token.decode('utf-8', errors='replace')!r}\n"
    )

    (output_dir / "results.txt").write_text(
        report,
        encoding="utf-8",
    )

    # 五、保存性能分析结果，用来找最耗时间的地方。
    profiler.dump_stats(str(output_dir / "profile.prof"))

    with open(
        output_dir / "profile.txt",
        "w",
        encoding="utf-8",
    ) as file:
        stats = pstats.Stats(profiler, stream=file)
        stats.strip_dirs()

        file.write("按函数及其调用的累计耗时排序：\n")
        stats.sort_stats("cumulative").print_stats(20)

        file.write("\n按函数自身耗时排序：\n")
        stats.sort_stats("tottime").print_stats(20)

    print("\n实验结果：")
    print(report)
    print("结果已保存到：", output_dir)
