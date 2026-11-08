"""仅加速兼容的自训词表；编号和规则继续来自本地训练产物。"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version

import tiktoken

from cs336_basics.tokenizer import PAT


class FastTokenizer:
    def __init__(self, reference, workers=4):
        self.reference = reference
        self.vocab = reference.vocab
        self.backend = "tiktoken-custom-vocab"
        self.backend_version = version("tiktoken")
        self.workers = max(1, min(4, workers))
        self.marker = "<|endoftext|>"
        if reference.special_tokens != [self.marker]:
            raise ValueError("加速文件编码目前要求唯一特殊标记为 <|endoftext|>")
        if len(set(self.vocab.values())) != len(self.vocab):
            raise ValueError("词表包含重复字节，不能直接映射原生编码优先级")
        special_id = reference.special_ids[self.marker]
        ranks = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
            if token_id != special_id
        }
        if len(ranks) != 256 + len(reference.merges):
            raise ValueError("词表与合并规则数量不符合自训词表格式")
        for value in range(256):
            if ranks.get(bytes([value])) != value:
                raise ValueError("初始单字节编号不符合当前加速格式")
        previous = 255
        known = {bytes([value]) for value in range(256)}
        for left, right in reference.merges:
            merged = left + right
            rank = ranks.get(merged)
            if left not in known or right not in known or rank is None or rank <= previous:
                raise ValueError("词表编号与合并创建顺序不一致，保留 Python 实现")
            known.add(merged)
            previous = rank
        self.engine = tiktoken.Encoding(
            name="cs336-own-trained-vocab",
            pat_str=PAT,
            mergeable_ranks=ranks,
            special_tokens={self.marker: special_id},
        )

    def encode(self, text):
        return self.engine.encode(text, allowed_special={self.marker})

    def decode_bytes(self, ids):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self.engine.decode_bytes(ids)

    def _batch(self, text):
        return self.engine.encode_to_numpy(text, allowed_special={self.marker})

    def _ordinary_safe_cut(self, text):
        # 极长文档没有结束标记时，选择稳定、且不以空白结束的预分词边界。
        # 这样重新编码前缀不会因为 EOF 改变空白的分组。
        stable_end = len(text)
        for length in range(len(self.marker) - 1, 0, -1):
            if text.endswith(self.marker[:length]):
                stable_end -= length
                break
        pending = deque()
        cut = 0
        for piece in self.reference._pieces(text[:stable_end]):
            pending.append(piece)
            if len(pending) > 2:
                _, end, value = pending.popleft()
                if not value.isspace():
                    cut = end
        return cut

    def _text_batches(self, chunks):
        buffer = ""
        for chunk in chunks:
            buffer += chunk
            last_marker = buffer.rfind(self.marker)
            if last_marker >= 0:
                end = last_marker + len(self.marker)
                yield buffer[:end]
                buffer = buffer[end:]
            # 常规批次在完整文档边界结束；超长文档也保持有界缓冲。
            if len(buffer) > 4 * 1024 * 1024:
                cut = self._ordinary_safe_cut(buffer)
                if cut:
                    yield buffer[:cut]
                    buffer = buffer[cut:]
                elif len(buffer) > 64 * 1024 * 1024:
                    raise RuntimeError("遇到超过64Mi字符的单个未结束片段，请检查输入文本")
        if buffer:
            yield buffer

    def iter_encoded_batches(self, chunks):
        # 只并行完整、安全的文本批次；按原顺序收取结果。
        # 最多保留 workers * 2 个批次，避免把全文排进内存。
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            pending = deque()
            for text in self._text_batches(chunks):
                pending.append(pool.submit(self._batch, text))
                if len(pending) >= self.workers * 2:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()

    def encode_iterable(self, chunks):
        for batch in self.iter_encoded_batches(chunks):
            yield from batch.tolist()
