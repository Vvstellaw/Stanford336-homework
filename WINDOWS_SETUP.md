# Windows 本地运行说明

这份说明用于配置、运行和检查作业项目。作业代码当前已完成 Transformer 语言模型的结构与前向计算，损失函数、优化器和训练循环等内容仍按后续学习进度完成；模型参数尚未进行正式训练。

## 1. 打开正确的项目

在 VS Code 里选择“文件 → 从文件打开工作区”，打开外层文件：

```text
C:\Users\stellaw\Desktop\stanford336\assignment1-basics-main\Stanford336.code-workspace
```

也可以直接打开内层项目文件夹：

```text
C:\Users\stellaw\Desktop\stanford336\assignment1-basics-main\assignment1-basics-main
```

资源管理器里应能同时看到 `pyproject.toml`、`uv.lock`、`cs336_basics`、`tests` 和 `data`。

打开工作区后，按提示安装推荐的 Python、Python Debugger 扩展。再打开一个新的 VS Code 终端，让本项目的终端配置生效。

## 2. 这个项目使用哪些程序

| 位置 | 作用 |
| --- | --- |
| `.tools/python-3.12.14/python.exe` | 项目保存的基础 Python 3.12.14 |
| `.venv/Scripts/python.exe` | 运行作业、测试和 VS Code 调试时使用的 Python |
| `.venv/Lib/site-packages` | 本项目的 PyTorch、pytest 等依赖 |
| `.tools/uv/uv.exe` | 项目使用的 uv 环境管理工具 |
| `.tools/tiktoken-cache` | 老师 GPT-2 对照测试使用的公共参考缓存 |
| `.tools/tmp` | 本项目运行时的临时文件 |
| `.env` | 环境准备脚本生成、供 VS Code 测试和调试读取的本地环境设置 |
| `pyproject.toml` | Python 版本范围和依赖要求 |
| `uv.lock` | 已解析的依赖版本记录 |

运行时使用项目自己的解释器和依赖，不需要先进入 Python 的 `>>>` 界面。VS Code 终端已经把本项目的程序目录放到搜索路径前面，无需手动执行虚拟环境激活脚本。

## 3. 先检查环境，再检查已完成的作业

以下命令都在 PowerShell 或 VS Code 下方终端中运行。先进入项目根目录：

```powershell
cd "C:\Users\stellaw\Desktop\stanford336\assignment1-basics-main\assignment1-basics-main"
```

检查 Python、依赖和项目能否正常加载：

```powershell
.\project.cmd check
```

这一步检查运行条件，不会启动模型训练。以终端实际输出为准。

检查当前已经实现的部分：

```powershell
.\project.cmd test
```

它运行项目维护的“已完成部分”测试清单。测试结果应结合清单中的具体模块判断，不能把它理解为整份作业已经完成。

也可以在 VS Code 中按 `Ctrl+Shift+P`，搜索“任务: 运行任务”，选择：

- `CS336：检查运行环境`
- `CS336：测试已完成部分`
- `CS336：运行全部作业测试`
- `CS336：准备参考分词器缓存（离线）`

## 4. 运行代码和调试

运行一个 Python 文件：

```powershell
.\uv.cmd run python 路径\文件名.py
```

运行包里的模块：

```powershell
.\uv.cmd run python -m cs336_basics.模块名
```

例如，只查看 BPE 训练程序接受哪些参数，不开始训练：

```powershell
.\uv.cmd run python -m cs336_basics.run_bpe --help
```

在 VS Code 中调试时，打开要运行的 Python 文件，按 `F5`，选择 `CS336：调试当前 Python 文件`。工作目录使用项目根目录，所以 `data/...`、`tests/...` 这类相对路径从项目根目录开始查找。

只包含类或函数定义的文件，执行后可能没有输出；程序需要实际调用函数，才会进行对应计算。

## 5. 全部测试为什么还会有失败

运行整套老师的测试：

```powershell
.\project.cmd test-all
```

老师的测试覆盖整份作业。当前尚未实现的损失函数、优化器、数据取批次和检查点等，仍会出现 `NotImplementedError`。

这表示对应作业功能尚未完成。环境是否正常，先看 `project.cmd check` 和已完成模块的测试。不要通过删除测试或跳过未完成题目来制造“全通过”。

VS Code 的测试发现仍然包含全部 `tests`，方便继续完成作业时逐项检查。Windows 不支持的 Linux 专用内存限制测试，已有平台跳过条件；`skipped` 与测试通过含义不同。

## 6. 老师的 GPT-2 对照测试缓存

有些测试会调用 tiktoken 的 GPT-2 编码器，将它的结果与我们实现的分词器进行比较。这需要两份总计约 1.5 MB 的公共参考文件：`vocab.bpe` 和 `encoder.json`。

项目会优先从老师已有的 `tests/fixtures` 恢复这两份文件，并核对 tiktoken 指定的官方 SHA256。合并规则的 fixture 缺少官方格式头行，脚本只在内存里补回这一行，并且完整校验通过后才写入 `.tools/tiktoken-cache`；原测试文件保持原样。

这些缓存只供老师的 GPT-2 对照测试使用。我们训练的 TinyStories、OpenWebText 词表和合并规则，仍从原来的实验结果加载。

环境检查和已完成部分测试使用离线缓存准备方式；VS Code 的“全部作业测试”任务也会先运行离线准备任务。单独检查缓存可以执行：

```powershell
.\.venv\Scripts\python.exe scripts/cache_reference_tokenizer.py --offline
```

如果缓存和可校验的 fixtures 都缺失，离线模式会直接报错，不会偷偷联网。恢复老师的原 fixtures 后可以重试；也可以明确允许参考缓存脚本联网：

```powershell
.\.venv\Scripts\python.exe scripts/cache_reference_tokenizer.py
```

脚本仅在本地内容无法恢复时访问预设的官方 HTTPS 地址，限制下载超时，并在 SHA256 校验通过后写入缓存。测试使用项目设置的 `TIKTOKEN_CACHE_DIR`，无需访问系统默认缓存目录。VS Code 通过 `.env` 和任务环境读取本地缓存与临时目录设置；更新环境后应重新打开终端。

## 7. 现有数据和实验结果

`data` 目录里的文本、BPE 词表和编码结果继续保留。修复环境不需要重新下载这些数据，也不需要重新训练分词器或重新编码完整数据集。

检查环境、运行组件测试不会自动启动这些耗时任务。需要重做某项实验时，再明确选择对应的数据集和命令。

## 8. 移动项目、换电脑或从版本库恢复

`.tools` 和 `.venv` 是本机运行环境，不应提交到版本库。`data` 也不应连同源码一起提交。依赖要求和版本记录分别保存在 `pyproject.toml`、`uv.lock`。

移动路径、重新解压或换电脑时，应重建环境，不要直接复制旧 `.venv` 使用，因为其中可能记录旧的解释器路径。

项目提供了 Windows 环境准备脚本。在项目根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
.\project.cmd check
.\project.cmd test
```

`setup_windows.ps1` 会准备项目自己的 uv；优先复用已有的项目基础 Python，否则安装 Python 3.12.14 到 `.tools/managed-python`，再建立或修复虚拟环境，并按 `uv.lock` 同步依赖。命令中的执行策略选项只针对这一次 PowerShell 进程。

本地工具和依赖缓存齐全后，也可以离线重复检查环境准备过程：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Offline
```

离线模式缺少所需文件时会报错，不会自动联网下载。

如果只需要单独管理 Python 或同步依赖，也可以使用包装命令：

```powershell
.\uv.cmd python install 3.12.14
.\uv.cmd sync --locked
```

`uv.cmd` 优先使用 `.tools/uv/uv.exe`，找不到时使用系统搜索路径上的 `uv.exe`。全新克隆如果两者都没有，先运行上面的环境准备脚本。

首次准备 uv、Python 或缺少的依赖时需要下载；已有的依赖可由 uv 缓存复用。下载需要正常的公司网络或已配置的代理，不在项目里固定某一个代理地址。不要删除 `uv.lock` 来解决下载或网络问题。uv 的通用安装说明也可参考 [官方文档](https://docs.astral.sh/uv/getting-started/installation/)。

恢复后重新打开 VS Code 工作区。如果状态栏仍指向旧路径，运行“Python: 选择解释器”，选择本项目的 `.venv/Scripts/python.exe`。
