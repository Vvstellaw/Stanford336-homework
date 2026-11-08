"""Check the real project interpreter, dependencies, and one small CPU calculation."""

import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime
from pathlib import Path

from packaging.requirements import Requirement

PROJECT = Path(__file__).resolve().parents[1]
LOCAL_TEMP = PROJECT / ".tools" / "tmp"
LOCAL_TEMP.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(LOCAL_TEMP)
os.environ.update(TEMP=str(LOCAL_TEMP), TMP=str(LOCAL_TEMP),
                  TIKTOKEN_CACHE_DIR=str(PROJECT / ".tools" / "tiktoken-cache"))


def main():
    checks = []

    def record(label, passed, detail):
        checks.append({"name": label, "passed": bool(passed), "detail": str(detail)})
        print(f"[{'OK' if passed else 'FAIL'}] {label}: {detail}", flush=True)

    print("CS336 project environment check", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    print(f"Base runtime: {sys.base_prefix}", flush=True)
    record("Python version", sys.version_info[:3] == (3, 12, 14), platform.python_version())
    record("Project virtual environment", Path(sys.prefix).resolve() == PROJECT / ".venv", sys.prefix)
    record("Project-owned base runtime", Path(sys.base_prefix).resolve().is_relative_to(PROJECT / ".tools"), sys.base_prefix)

    configuration = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    for specification in configuration["project"]["dependencies"]:
        requirement = Requirement(specification)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            version = importlib.metadata.version(requirement.name)
            record(requirement.name, requirement.specifier.contains(version), version)
        except importlib.metadata.PackageNotFoundError:
            record(requirement.name, False, "not installed")

    for name in ("torch", "numpy", "regex", "tiktoken", "einops", "einx", "jaxtyping", "pytest", "psutil", "wandb"):
        try:
            module = importlib.import_module(name)
            record(f"Import {name}", True, getattr(module, "__file__", "imported"))
        except Exception as error:
            record(f"Import {name}", False, repr(error))

    try:
        import cs336_basics
        import torch
        from cs336_basics.layers import MultiHeadSelfAttention

        record("Editable project import", Path(cs336_basics.__file__).resolve().is_relative_to(PROJECT / "cs336_basics"), cs336_basics.__file__)
        layer = MultiHeadSelfAttention(8, 2, max_seq_len=8, theta=10000)
        inputs = torch.randn(2, 3, 8, requires_grad=True)
        outputs = layer(inputs)
        outputs.square().mean().backward()
        passed = outputs.shape == inputs.shape and torch.isfinite(outputs).all().item()
        passed = passed and inputs.grad is not None and torch.isfinite(inputs.grad).all().item()
        record("Small attention forward/backward", passed, f"{tuple(inputs.shape)} -> {tuple(outputs.shape)}")
        print(f"Compute: {'CUDA available' if torch.cuda.is_available() else 'CPU (CUDA is not required for these checks)'}", flush=True)
    except Exception as error:
        record("Project calculation", False, repr(error))

    uv = PROJECT / ".tools" / "uv" / "uv.exe"
    if uv.is_file():
        completed = subprocess.run([str(uv), "--version"], capture_output=True, text=True, check=False)
        record("Project uv", completed.returncode == 0, completed.stdout.strip() or completed.stderr.strip())
    else:
        record("Project uv", False, "run scripts/setup_windows.ps1")

    try:
        from cache_reference_tokenizer import ensure_cache

        ensure_cache(offline=True)
        import tiktoken

        tokenizer = tiktoken.get_encoding("gpt2")
        record("Offline reference tokenizer", tokenizer.n_vocab == 50257, f"gpt2: {tokenizer.n_vocab} tokens")
    except Exception as error:
        record("Offline reference tokenizer", False, repr(error))

    datasets = {name: (PROJECT / "data" / name).is_file() for name in (
        "TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt", "owt_train.txt", "owt_valid.txt"
    )}
    print(f"Local source datasets present: {sum(datasets.values())}/{len(datasets)} (no files downloaded or scanned)", flush=True)
    report = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "python_executable": sys.executable,
        "base_prefix": sys.base_prefix,
        "checks": checks,
        "datasets": datasets,
        "passed": all(item["passed"] for item in checks),
    }
    folder = PROJECT / ".tools" / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "environment_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {folder / 'environment_check.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
