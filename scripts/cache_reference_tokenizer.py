"""Prepare only the public GPT-2 reference files used by the teacher's tests.

The installed tiktoken_ext/openai_public.py supplies the official URLs and hashes.
This does not replace or modify the student's tokenizer, trained vocabularies, or tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".tools" / "tiktoken-cache"
MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 15
REFERENCE_FILES = (
    {
        "name": "vocab.bpe",
        "url": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
        "sha256": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
        "fixture": "gpt2_merges.txt",
    },
    {
        "name": "encoder.json",
        "url": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
        "sha256": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
        "fixture": "gpt2_vocab.json",
    },
)


def _matches(data: bytes, expected_hash: str) -> bool:
    return hashlib.sha256(data).hexdigest() == expected_hash


def _matching_fixture(entry: dict[str, str]) -> tuple[bytes, str] | None:
    path = PROJECT_ROOT / "tests" / "fixtures" / entry["fixture"]
    if not path.is_file():
        return None
    original = path.read_bytes()
    lf_only = original.replace(b"\r\n", b"\n")
    candidates = [(original, "exact"), (lf_only, "CRLF-to-LF")]
    if entry["name"] == "vocab.bpe":
        # The supplied merges fixture omits the official 14-byte header.
        # Restore it in memory only, accepting the result solely on exact SHA256.
        header = b"#version: 0.2\n"
        candidates.extend([
            (header + original, "official-header"),
            (header + lf_only, "official-header-and-LF"),
        ])
    for data, transformation in candidates:
        if _matches(data, entry["sha256"]):
            return data, f"fixture:{entry['fixture']}:{transformation}"
    return None


def _download(entry: dict[str, str]) -> bytes:
    try:
        with urllib.request.urlopen(
            entry["url"], timeout=DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            data = response.read(MAX_DOWNLOAD_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(
            f"无法下载官方参考文件 {entry['name']}：{error}"
        ) from error
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise RuntimeError(f"参考文件 {entry['name']} 超过允许的下载大小")
    if not _matches(data, entry["sha256"]):
        raise RuntimeError(f"官方参考文件 {entry['name']} 的 SHA256 校验失败，未写入缓存")
    return data


def _write_verified(path: Path, data: bytes, expected_hash: str) -> None:
    # Check before any file write; temporary and final files both stay in CACHE_DIR.
    if not _matches(data, expected_hash):
        raise RuntimeError("参考内容 SHA256 校验失败，未写入缓存")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_DIR / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_cache(offline: bool = True) -> dict[str, object]:
    """Return verified cache details; default calls can never download anything."""
    records = []
    for entry in REFERENCE_FILES:
        # tiktoken names raw download cache files by SHA1 of the original URL.
        filename = hashlib.sha1(entry["url"].encode("utf-8")).hexdigest()
        path = CACHE_DIR / filename
        source = "cache"
        if path.is_file() and _matches(path.read_bytes(), entry["sha256"]):
            size = path.stat().st_size
        else:
            match = _matching_fixture(entry)
            if match is not None:
                data, source = match
            elif offline:
                raise RuntimeError(
                    f"离线模式缺少已校验的 {entry['name']}。"
                    "请恢复原 tests/fixtures，或联网运行 "
                    "python scripts/cache_reference_tokenizer.py 后再测试。"
                )
            else:
                data = _download(entry)
                source = "official-https"
            _write_verified(path, data, entry["sha256"])
            size = len(data)
        records.append({
            "name": entry["name"],
            "source": source,
            "path": str(path),
            "sha256": entry["sha256"],
            "bytes": size,
        })
    os.environ["TIKTOKEN_CACHE_DIR"] = str(CACHE_DIR)
    return {"cache_dir": str(CACHE_DIR), "offline": offline, "entries": records}


def main() -> int:
    parser = argparse.ArgumentParser(description="准备老师 GPT-2 对照测试的公共参考缓存。")
    parser.add_argument("--offline", action="store_true", help="仅使用本地缓存和经校验的 fixtures，不联网")
    args = parser.parse_args()
    try:
        result = ensure_cache(offline=args.offline)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"参考分词器缓存准备失败：{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
