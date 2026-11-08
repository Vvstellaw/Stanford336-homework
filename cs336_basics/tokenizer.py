import pickle
from collections import deque
from functools import lru_cache
from pathlib import Path

import regex


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        # 1. 保存训练结果，保留原来的编号和合并顺序。
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(
            dict.fromkeys(special_tokens or [])
        )

        if any(token == "" for token in self.special_tokens):
            raise ValueError("特殊标记不能是空字符串")

        # 编号 → 字节：self.vocab
        # 字节 → 编号：self.token_to_id
        self.token_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }

        # 指定的特殊标记如果不在词表中，就补进去。
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")

            if token_bytes not in self.token_to_id:
                token_id = max(self.vocab, default=-1) + 1
                self.vocab[token_id] = token_bytes
                self.token_to_id[token_bytes] = token_id

        self.special_ids = {
            token: self.token_to_id[token.encode("utf-8")]
            for token in self.special_tokens
        }

        # 每条规则的编号，表示它在训练时被学到的先后顺序。
        self.merge_ranks = {}

        for rank, pair in enumerate(self.merges):
            self.merge_ranks.setdefault(pair, rank)

        self.pattern = regex.compile(PAT)

        # 较长的特殊标记优先匹配。
        ordered = sorted(
            self.special_tokens,
            key=lambda token: (-len(token), token),
        )

        self.special_pattern = (
            regex.compile("|".join(regex.escape(token) for token in ordered))
            if ordered
            else None
        )

        self.longest_special = max(
            map(len, self.special_tokens),
            default=1,
        )

        # 重复片段复用编码结果，缓存数量有上限。
        self.cached_piece = lru_cache(maxsize=1024)(
            self._encode_ordinary_piece
        )

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # 2. 读取已经保存的词表和合并规则。
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)

        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)

        return cls(vocab, merges, special_tokens)

    def _pieces(self, text):
        # 先隔开特殊标记，再对普通文字进行预分词。
        position = 0

        if self.special_pattern is not None:
            for special in self.special_pattern.finditer(text):
                for match in self.pattern.finditer(
                    text, position, special.start()
                ):
                    yield match.start(), match.end(), match.group()

                yield special.start(), special.end(), special.group()
                position = special.end()

        for match in self.pattern.finditer(text, position):
            yield match.start(), match.end(), match.group()

    def _encode_ordinary_piece(self, text):
        # 3. 一个普通片段先拆成单字节 token。
        parts = [
            bytes([value])
            for value in text.encode("utf-8")
        ]

        while len(parts) > 1:
            best_pair = None
            best_rank = len(self.merges)

            # 找当前相邻对中，训练时最早学到的合并规则。
            for pair in zip(parts, parts[1:]):
                rank = self.merge_ranks.get(pair)

                if rank is not None and rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            # 从左到右，合并选中的相邻对。
            merged = []
            position = 0

            while position < len(parts):
                if (
                    position + 1 < len(parts)
                    and (parts[position], parts[position + 1]) == best_pair
                ):
                    merged.append(parts[position] + parts[position + 1])
                    position += 2
                else:
                    merged.append(parts[position])
                    position += 1

            parts = merged

        # 最后把每个 token 的字节转换成编号。
        return tuple(
            self.token_to_id[part]
            for part in parts
        )

    def _piece_ids(self, text):
        if text in self.special_ids:
            return (self.special_ids[text],)

        return self.cached_piece(text)

    def encode(self, text):
        # 4. 按原文顺序收集所有 token 编号。
        ids = []

        for _, _, piece in self._pieces(text):
            ids.extend(self._piece_ids(piece))

        return ids

    def decode(self, ids):
        # 5. 先拼接所有字节，再统一还原成文字。
        all_bytes = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )

        return all_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable):
        # 6. 分批编码。
        # 未完成的特殊标记可能改变它前面空白的预分词方式，
        # 因此先留住这种后缀，再保留普通区域末尾的两个片段。
        special_prefixes = {
            token[:length]
            for token in self.special_tokens
            for length in range(1, len(token))
        }
        prefix_lengths = sorted(
            {len(prefix) for prefix in special_prefixes},
            reverse=True,
        )
        buffer = ""

        for chunk in iterable:
            buffer += chunk
            stable_end = len(buffer)
            candidates = [
                len(buffer) - length
                for length in prefix_lengths
                if length <= len(buffer)
                and buffer[-length:] in special_prefixes
            ]

            if candidates:
                complete_spans = [
                    (match.start(), match.end())
                    for match in self.special_pattern.finditer(buffer)
                    if match.end() > candidates[0]
                ]
                for start in candidates:
                    # 已匹配的完整标记会遮住其内部的其他标记起点。
                    # 同起点则仍可能延长为另一个更长的特殊标记。
                    if not any(left < start < right for left, right in complete_spans):
                        stable_end = start
                        break

            pending = deque()
            consumed = 0

            for piece in self._pieces(buffer[:stable_end]):
                pending.append(piece)

                while len(pending) > 2:
                    _, end, text = pending.popleft()
                    yield from self._piece_ids(text)
                    consumed = end

            buffer = buffer[consumed:]

        # 文件读完后，处理剩下的内容。
        yield from self.encode(buffer)


if __name__ == "__main__":
    # 直接运行本文件，只检查一句话的编码和解码。
    project_root = Path(__file__).resolve().parent.parent
    folder = project_root / "data" / "tinystories_bpe_10000"

    tokenizer = Tokenizer.from_files(
        folder / "vocab.pkl",
        folder / "merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    text = "Once upon a time there was a little boy named Ben."
    ids = tokenizer.encode(text)
    restored = tokenizer.decode(ids)

    assert restored == text

    print("原文：", text)
    print("Token 编号：", ids)
    print("还原文字：", restored)
    print("检查通过")
