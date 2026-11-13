#!/usr/bin/env python3
"""Sync docs/ from the workspace document tree.

The authoritative copies of these four documents live in the workspace at
    <WORKSPACE>/docs/60-llama-split/
because that tree is index-checked (tools/gen_index.py) and covers every project.
This repo carries a copy so that code and its design notes travel together.

Run this after editing the workspace docs. It is idempotent: it re-copies, then
applies the same small set of repository-specific rewrites, then regenerates the
extracted ledger. If a rewrite no longer matches, it fails loudly instead of
silently producing a stale document.
"""
import datetime
import os
import re
import sys

WORKSPACE = os.environ.get(
    "SPLIT_DOCS_SRC",
    # Maintainer's local default. Point SPLIT_DOCS_SRC at your own document tree before running.
    r"C:/Users/normin/WorkBuddy/2026-10-27-09-59-46/docs",
)
SRC = os.path.join(WORKSPACE, "60-llama-split")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

COPIES = {
    "architecture.md": "architecture.md",
    "runbook.md": "runbook.md",
    "create-overhead.md": "create-overhead.md",
    "README.md": "README.md",
}

# ---------------------------------------------------------------- rewrites
# Each entry must match exactly once, otherwise the source drifted and we abort.

README_HEAD_OLD = """> 代码：驱动器 `D:/llama-split`（运行在设备 `/data/local/tmp/llama.cpp`）、
> llama.cpp 侧改造 `D:/llama-hex`（`layer-split.patch`）。
> **两者都不在本树**，本目录只有结论、架构与复现指引。"""

README_HEAD_NEW = """> **这份文档的权威版本在工作区 `docs/60-llama-split/`**（那棵树由 `tools/gen_index.py` 校验链接，
> 并且覆盖 PMS / Qwen-Image / gemma4 全部项目）。这里是搬进代码仓库的副本，好让代码和设计说明一起走。
> 两份不一致时**以工作区为准**；同步方式见文末。
>
> 代码：本仓库根目录（部署到设备 `/data/local/tmp/llama.cpp`）。
> llama.cpp 侧的层切分改造以补丁形式放在 [`patches/`](../patches/)（基线 `7d7f028`）——
> **补丁在本仓库，llama.cpp 源码需要自己 clone**。"""

README_BLOCKS_OLD = """| 块 | 关系 |
|---|---|
| `../30-gemma4-e4b/` | gemma4 走的是**硬件异构**（NPU prefill + CPU decode），不是层切分；两者可叠加 |
| `../20-qwen-image-qnn/` | 同一套「分段 + 共享内存」思路的另一个实例（11 段链）；`architecture.md` §5.3 的 ABI 想统一这两者 |
| `../50-platform/` | 运行时/工具链/adb 坑都在那里 |"""

README_BLOCKS_NEW = """这三个块属于原工作区的文档树（`docs/` 下与本目录平级的其他块），**不在本仓库**：

| 块（工作区 `docs/`） | 关系 |
|---|---|
| `30-gemma4-e4b/` | gemma4 走的是**硬件异构**（NPU prefill + CPU decode），不是层切分；两者可叠加 |
| `20-qwen-image-qnn/` | 同一套「分段 + 共享内存」思路的另一个实例（11 段链）；`architecture.md` §5.3 的 ABI 想统一这两者 |
| `50-platform/` | 运行时/工具链/adb 坑都在那里 |"""

LEDGER_REFS = [
    ("create-overhead.md",
     "结论依据见 `docs/LEDGER.md` 的 `L-043`、`L-042`、`L-038`、`L-033`。",
     "结论依据见 [`ledger.md`](ledger.md) 的 `L-043`、`L-042`、`L-038`、`L-033`。"),
    ("runbook.md",
     "结论依据见 `docs/LEDGER.md` 的 `L-028` ~ `L-036`。",
     "结论依据见 [`ledger.md`](ledger.md) 的 `L-028` ~ `L-036`。"),
]

# The upstream driver lives in patches/ in this repo, so the workspace wording ("it lives in
# D:/llama-hex, not in this tree") is both wrong here and useless to a reader. Rewrite those
# two spots to name the real artefact before the path sanitiser runs.
FILE_REWRITES = {
    # "60" is the block number in the workspace tree; meaningless in this repo
    "README.md": [
        ("# 60 · 层切分引擎（多进程 pipeline）", "# 层切分引擎（多进程 pipeline）"),
        ("| 一句话结论 | 台账 `L-028` ~ `L-061` |",
         "| 一句话结论（40 条，含被否决的方案） | [`ledger.md`](ledger.md) |"),
        ("同类坑：本机两个 adb 版本混用会互相杀 daemon，`forward` 静默失效 —— 统一到\n"
         "   `D:/android-tools/sdk/platform-tools/adb.exe`。",
         "同类坑：两个来源不同的 adb 混用会互相杀 daemon，`forward` 静默失效 —— 只保留一个。"),
    ],
    "architecture.md": [(
        "代码位置：驱动器 `D:/llama-split/llama-split.cpp`、服务前端 `D:/llama-split/serve_http.py`、\n"
        "llama.cpp 侧改造 `D:/llama-hex`（`layer-split.patch`）、设备运行时 `/data/local/tmp/llama.cpp`。",
        "代码位置：驱动器 `llama-split.cpp`、服务前端 `serve_http.py`（均在本仓库根目录）、\n"
        "llama.cpp 侧改造见 `patches/`（基线 `7d7f028`）、设备运行时 `/data/local/tmp/llama.cpp`。",
    )],
    "runbook.md": [(
        "| `D:/llama-split/quantize_gguf.py` | 纯 numpy GGUF 量化器（本机没有能用的 x86 `llama-quantize` 时用它） |",
        "| `quantize_gguf.py` | 纯 numpy GGUF 量化器（没有可用的 x86 `llama-quantize` 时用它） |\n"
        "| `serve_http.py` | PC 侧 HTTP 前端：流式、`/cancel`、每请求一条独立设备连接 |\n"
        "| `stage_abi.h` / `stage_pass.c` | 外部 stage 的 ABI 契约 + 参考实现（见「扩展它」） |\n"
        "| `lib/` | 运行时库 12 个 .so，换机器直接推，见 `lib/README.md` |",
    ), (
        "| `D:/llama-hex`（`layer-split.patch`） |",
        "| `patches/`（`llama-layer-split.patch`） |",
    ), (
        old_block if False else '```bash\n# 1) llama.cpp（docker，增量约 8 min）\ncd D:/llama-hex && python scripts/snapdragon/build.py --target adb --hex-arch v73\nexport MSYS_NO_PATHCONV=1\nADB=D:/android-tools/sdk/platform-tools/adb.exe\n$ADB push D:/llama-hex/pkg-adb/llama.cpp/lib/libllama.so /data/local/tmp/llama.cpp/lib/libllama.so\n\n# 2) 驱动器（约 30 s，产物在 stdout 里落地）\ncd D:/llama-split && ./build.sh\n$ADB push llama-split /data/local/tmp/llama.cpp/bin/llama-split\n$ADB shell "chmod 755 /data/local/tmp/llama.cpp/bin/llama-split"\n```',
        '仓库自带 `lib/`（12 个 .so）和编好的 `llama-split`，所以**默认不需要编译任何东西**：\n\n```bash\nexport MSYS_NO_PATHCONV=1        # Git Bash 会改写 /data/... 这类路径\nADB=adb                          # 或你的 adb 绝对路径\n$ADB shell "mkdir -p /data/local/tmp/llama.cpp/bin /data/local/tmp/llama.cpp/lib"\n$ADB push lib/*.so /data/local/tmp/llama.cpp/lib/\n$ADB push llama-split /data/local/tmp/llama.cpp/bin/llama-split\n$ADB shell "chmod 755 /data/local/tmp/llama.cpp/bin/llama-split"\n```\n\n什么时候才需要重新编译：\n\n| 改了什么 | 要重编 | 代价 |\n|---|---|---|\n| `llama-split.cpp` / `stage_pass.c` | `./build.sh` / `./build_stage.sh` | ~30 s（docker + NDK） |\n| `patches/` 里的上游改动 | llama.cpp 全量（`scripts/snapdragon/build.py`） | ~8 min 增量，产物替换 `lib/*.so` |\n| 都没改 | 不用编 | — |\n\n重编 llama.cpp 后**必须同时更新 `lib/*.so`**：host 库和 DSP applet 出自同一次构建，混用会崩。',
    )],
}

README_TAIL = """
## 怎么同步这份文档

这几份是工作区 `docs/60-llama-split/` 的副本，**不要直接在这里改**——改了下次同步就没了。
正确流程：

```sh
# 1. 改工作区那份，并让它自己的检查通过
cd <工作区> && python tools/gen_index.py --check

# 2. 同步到本仓库（复制 + 结构改写 + 路径清洗 + 重新抽取台账）
python sync_docs.py
```

`sync_docs.py` 是幂等的。如果上游文档改到它认不出来的程度，它会**报错而不是静默产出旧内容**。
"""


def rewrite(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            sys.exit("!! %s: pattern matched %d times (expected 1), source drifted:\n%s"
                     % (os.path.basename(path), n, old[:90]))
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="\n").write(s)


# ---------------------------------------------------------------- sanitation
# The workspace copies are written for this machine: they carry absolute paths to the
# source trees, the adb binary, and the gguf files. That is useful there and useless (or
# confusing) to anyone else, including an AI agent reading this repo. The repo copies are
# sanitised so every reference resolves either inside this repository or to "something you
# must obtain yourself".
#
# Deliberately NOT touched: device paths (/data/local/tmp, /vendor/lib64, /system_ext) --
# those are where things actually live on the phone, not local-machine noise.
SANITIZE = [
    # Shell snippets must stay runnable. These run before the generic rules below, and they
    # MUST anchor on the whole line: without (?m) a "cd D:/src" in the middle of a code block
    # silently became "cd llama.cpp 源码树", which is a broken command, not a broken sentence.
    (r"(?m)^cd D:/llama-hex[ \t]*$", "cd <llama.cpp>"),
    (r"(?m)^cd D:/llama-hex && ", "cd <llama.cpp> && "),
    (r"(?m)^cd D:/llama-split[ \t]*$", "cd <repo>"),
    (r"(?m)^cd D:/llama-split && ", "cd <repo> && "),
    (r"(?m)^cd D:/llama-split/", "cd <repo>/"),
    # repository-internal paths become relative to the repo root
    (r"D:/llama-split/", ""),
    (r"D:/llama-split", "本仓库"),
    # the llama.cpp source tree: needed to rebuild lib/, but not shipped here
    (r"D:/llama-hex/pkg-adb/llama\.cpp/", "llama.cpp/"),
    (r"D:/llama-hex/", "llama.cpp/"),
    (r"D:/llama-hex", "llama.cpp 源码树"),
    (r"D:/android-tools/sdk/platform-tools/adb\.exe", "adb"),
    (r"`D:/android-tools/sdk`", "SDK 里的 adb"),
    (r"`D:/ipcbench/ipcbench\.c`", "自写的 fork/eventfd 微基准"),
    (r"`D:/llama-cpu`", "一个早年构建的 x86 llama.cpp"),
    # the workspace document tree
    (r"C:/Users/[^/\s]+/WorkBuddy/[0-9-]+/docs", "工作区 docs/"),
    (r"C:/Users/[^/\s]+/WorkBuddy/[0-9-]+", "工作区"),
    # model files: you have to supply your own; keep the quant suffix, commands rely on it
    (r"D:/Qwen3\.5-[0-9A-Za-z._-]*-f16\.gguf", "<model>-f16.gguf"),
    (r"D:/Qwen3\.5-[0-9A-Za-z._-]*-Q4_0\.gguf", "<model>-Q4_0.gguf"),
    (r"D:/Qwen3\.5-[0-9A-Za-z._-]*-Q4_K_M\.gguf", "<model>-Q4_K_M.gguf"),
    (r"D:/Qwen3\.5-[0-9A-Za-z._-]*-Q8_0\.gguf", "<model>-Q8_0.gguf"),
    (r"D:/Qwen3\.5-[0-9A-Za-z._-]*\.gguf", "<model>.gguf"),
    (r"D:/[Qq]wen3\.5-[0-9A-Za-z._-]*", "<模型目录>"),
    # wording that only makes sense on the author's machine
    (r"本机两个 adb 版本混用", "两个来源不同的 adb 混用"),
    (r"本机没有能用的 x86", "没有可用的 x86"),
    (r"本机 python 3\.13 已装", "python 3.13 可用"),
    (r"本次实测", "实测"),
    (r"我们拿到", "实测拿到"),
    (r"我们还没支持", "尚未支持"),
    (r"与本次无关", "与之无关"),
    (r"本次教训固化", "教训固化"),
    (r"本次所有输出", "这些输出"),
    (r"本次排查中", "排查中"),
    (r"本次调试记录", "调试记录"),
    (r"本机", "本地"),
    # the patch used to be described by its old location; it now lives in patches/
    (r"llama\.cpp/layer-split\.patch", "patches/llama-layer-split.patch"),
]


def sanitize(text):
    for pat, repl in SANITIZE:
        text = re.sub(pat, repl, text)
    return text


def fix_pipes(line):
    """Escape unescaped `|` inside the conclusion cell.

    A few entries quote code that contains `||` or a regex alternation, e.g.
    `if (ls_active && (il < ls_first || il >= ls_last))` — a raw `|` there splits the
    markdown table into extra columns and the row renders as garbage.
    Already-escaped ones (`\\|`) are left alone, so this is idempotent.
    """
    cells = re.split(r"(?<!\\)\|", line)
    if len(cells) <= 8:
        return line
    conclusion = r"\|".join(cells[4:len(cells) - 3])
    return "|".join(cells[:4] + [conclusion] + cells[-3:])


def extract_ledger():
    rows = open(os.path.join(WORKSPACE, "LEDGER.md"), encoding="utf-8").read().split("\n")
    L = [sanitize(fix_pipes(r)) for r in rows
         if (m := re.match(r"^\| L-(\d+)", r)) and int(m.group(1)) >= 24]
    # E-016: kill -9 把 adb 搞死; E-017: 旧补丁过期 + 生成补丁的行尾陷阱;
    # E-018: HTP 并发挂机后只能断电池复位 —— L-067 的直接证据，必须一起带过来
    E_IDS = (16, 17, 18)
    E = [sanitize(fix_pipes(r)) for r in rows
         if (m := re.match(r"^\| E-(\d+)", r)) and int(m.group(1)) in E_IDS]
    if not L:
        sys.exit("!! no L rows found in the workspace ledger")

    l_ids = [int(re.match(r"^\| L-(\d+)", r).group(1)) for r in L]
    l_span = f"L-{min(l_ids):03d} ~ L-{max(l_ids):03d}"
    e_span = " / ".join(f"E-{i:03d}" for i in E_IDS)

    hdr = "| ID | 日期 | 域 | 结论 | 状态 | 证据 |\n|---|---|---|---|---|---|"
    out = f"""# 本项目的结论台账（抽取版）

> **这是快照，不是权威版本。**
>
> 权威版本在原工作区的 `docs/LEDGER.md`，那里是跨项目的全量台账
> （含 PMS / Qwen-Image / gemma4 / 硬件）。
> 这里只抽出跟层切分这条线相关的 **{len(L)} 条 {l_span}** 和 **{len(E)} 条环境条目（{e_span}）**。
>
> 改结论请改工作区那份，然后跑 `python sync_docs.py` 重新抽取；两份不一致时以工作区为准。
> 抽样日期：{datetime.date.today().isoformat()}。

## 为什么从 L-024 开始

L-024~L-027 是「多进程之前」的四种尝试（整份回滚 / 按层串行 / `fastrpc_munmap` 根因 /
`rpcmem_free` 方案），L-028 起才是多进程切分。**被否决的方案都留着**——它们是「别再试这个方向」
的唯一记录，删掉就会有人重新踩。更早的 L-001~L-023 是 gemma4 混推的结论，不在这条线上。

## 状态含义

| 状态 | 含义 |
|---|---|
| `现行` | 当前成立，可以引用 |
| `待验` | 假设，**不要当结论用** |
| `推翻` | 墓碑，别复活（通常写着被哪条推翻） |
| `禁止` | 方向已试到底，别再碰 |

## 证据指针缩写（只列本仓库内的）

| 缩写 | 文件 |
|---|---|
| `split/arch` | [`architecture.md`](architecture.md)（现状架构 + 通用化路线） |
| `split/runbook` | [`runbook.md`](runbook.md) |
| `split/create-overhead` | [`create-overhead.md`](create-overhead.md) |

其余缩写（`plat/*` `qwen/*` `gemma/*` `pms/*`）指向工作区 `docs/` 下的其他块，不在本仓库。

## L：层切分线（{l_span}）

{hdr}
""" + "\n".join(L) + f"""

## E：环境与工具链（{e_span}）

{hdr}
""" + "\n".join(E) + "\n"

    open(os.path.join(DST, "ledger.md"), "w", encoding="utf-8", newline="\n").write(out)
    return len(L), len(E)


def main():
    if not os.path.isdir(SRC):
        sys.exit("!! source not found: %s\n   set SPLIT_DOCS_SRC to the workspace docs dir" % SRC)
    os.makedirs(DST, exist_ok=True)

    for src, dst in COPIES.items():
        data = open(os.path.join(SRC, src), encoding="utf-8").read()
        open(os.path.join(DST, dst), "w", encoding="utf-8", newline="\n").write(data)
        print("copied  %s" % dst)

    rewrite(os.path.join(DST, "README.md"),
            [(README_HEAD_OLD, README_HEAD_NEW), (README_BLOCKS_OLD, README_BLOCKS_NEW)])
    for f, old, new in LEDGER_REFS:
        rewrite(os.path.join(DST, f), [(old, new)])

    for f, pairs in FILE_REWRITES.items():
        rewrite(os.path.join(DST, f), pairs)

    # strip this machine's absolute paths last: the rewrites above still match the raw text
    for dst in COPIES.values():
        p = os.path.join(DST, dst)
        text = sanitize(open(p, encoding="utf-8").read())
        open(p, "w", encoding="utf-8", newline="\n").write(text)
        print("cleaned %s" % dst)

    # the sync note is appended last so it always ends the file
    p = os.path.join(DST, "README.md")
    s = open(p, encoding="utf-8").read().rstrip("\n")
    open(p, "w", encoding="utf-8", newline="\n").write(s + "\n" + README_TAIL)

    nL, nE = extract_ledger()
    print("wrote   ledger.md (%d L rows, %d E rows)" % (nL, nE))
    print("done    docs/ is in sync with %s" % SRC)


if __name__ == "__main__":
    main()
