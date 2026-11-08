"""Run the completed homework sections and keep reproducible test records.

Update this explicit list as new sections are implemented. Full homework discovery
remains available separately; future NotImplementedError tests are not skipped.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
LOCAL_TEMP = PROJECT / ".tools" / "tmp"
LOCAL_TEMP.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(LOCAL_TEMP)
os.environ.update(TEMP=str(LOCAL_TEMP), TMP=str(LOCAL_TEMP),
                  TIKTOKEN_CACHE_DIR=str(PROJECT / ".tools" / "tiktoken-cache"))
COMPLETED = [
    "tests/test_train_bpe.py",
    "tests/test_tokenizer.py",
    "tests/test_data.py",
    "tests/test_model.py::test_linear",
    "tests/test_model.py::test_embedding",
    "tests/test_model.py::test_rmsnorm",
    "tests/test_model.py::test_silu_matches_pytorch",
    "tests/test_model.py::test_swiglu",
    "tests/test_model.py::test_rope",
    "tests/test_nn_utils.py::test_softmax_matches_pytorch",
    "tests/test_nn_utils.py::test_cross_entropy",
    "tests/test_nn_utils.py::test_gradient_clipping",
    "tests/test_optimizer.py::test_adamw",
    "tests/test_optimizer.py::test_get_lr_cosine_schedule",
    "tests/test_model.py::test_scaled_dot_product_attention",
    "tests/test_model.py::test_4d_scaled_dot_product_attention",
    "tests/test_model.py::test_multihead_self_attention",
    "tests/test_model.py::test_multihead_self_attention_with_rope",
    "tests/test_model.py::test_transformer_block",
    "tests/test_model.py::test_transformer_lm",
    "tests/test_model.py::test_transformer_lm_truncated_input",
]


def main():
    from cache_reference_tokenizer import ensure_cache

    ensure_cache(offline=True)
    reports = PROJECT / ".tools" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", *COMPLETED, "-q", "-p", "no:cacheprovider",
               f"--junitxml={reports / 'completed-tests.xml'}", *sys.argv[1:]]
    print(f"Project Python: {sys.executable}", flush=True)
    print("Running completed sections only; use project.cmd test-all for all homework tests.", flush=True)
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    with (reports / "completed-tests.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8") as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            result = process.wait()
    record = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "python_executable": sys.executable,
        "base_prefix": sys.base_prefix,
        "command": command,
        "exit_code": result,
        "source_sha256": {
            str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in (PROJECT / "cs336_basics", PROJECT / "tests") for path in sorted(folder.glob("*.py"))
        },
    }
    (reports / "completed-tests.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
