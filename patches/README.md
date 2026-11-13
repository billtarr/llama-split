# patches/ — llama.cpp 侧的改造

`llama-split` 本身不依赖这些改动就能编译，但**没有它们就跑不出正确数值**：
按层只加载本段权重、按层只分配本段 KV、ubatch 的确定性修正，都在这里。

本仓库不含 llama.cpp 源码（那是 ~60 MB 的上游仓库），只放它上面的改动。

| 项 | 值 |
|---|---|
| 文件 | `llama-layer-split.patch`（69 KB，15 个文件，+758 / −82） |
| 上游基线 | `7d7f028`（2026-11-08） |
| 生成方式 | `git diff`（未提交的工作树改动） |

## 怎么用

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 7d7f028
git apply /path/to/llama-layer-split.patch
```

基线必须对得上。上游 `7d7f028` 之后如果动过同名文件，`git apply` 会拒绝，
那时用 `git apply -3` 做三方合并，再手工核对下面这几处语义改动。

## 改了什么

| 文件 | 职责 |
|---|---|
| `src/llama-model-loader.cpp` `.h` | **只加载本段的层**：按 `LLAMA_LAYER_SPLIT=a:b` 跳过区间外的 `blk.*`，并处理 `output.weight` / `token_embd` 这类共享 tensor |
| `src/llama-model.cpp` `.h` | 层过滤的模型侧入口；`n_layer` 与实际加载层数解耦 |
| `src/llama-kv-cache.cpp` | **KV 按层切分**：`il < ls_first \|\| il >= ls_last` 的层不建 KV（`L-038`） |
| `src/llama-memory-recurrent.cpp` | 同上，循环状态（Mamba 类 / 混合注意力）的按层切分 |
| `src/llama-batch.cpp` `.h` | ubatch 带 token id：纯 embd 的 ubatch 会读未初始化 token，导致切分数值**偶发不一致**（`L-035`） |
| `src/llama-graph.cpp` `.h` | 无 token 的 ubatch 清零 `inp_tokens`；`s_copy` 在无 ROPE buffer 时判空 |
| `src/models/qwen3.cpp` `qwen35.cpp` `gemma4.cpp` | 模型图的层循环按切分区间跳过；MTP 层按需创建 |
| `src/llama-kv-cache-dsv4.cpp` | 层过滤的一行适配 |
| `ggml/src/ggml-hexagon/ggml-hexagon.cpp` | HTP 后端的可调项：单设备限制、buffer 预算、applet 选择（`L-062`） |

`L-xxx` 指向 [`../docs/ledger.md`](../docs/ledger.md)。

## 为什么补丁放在这里而不是当成 git submodule

- 每次调试改的都是这几处，patch 比 submodule 更容易看清「上游 + 我的改动」的边界
- 重新生成只要一条命令：`cd llama.cpp && git diff > ../llama-split/patches/llama-layer-split.patch`
- 缺点：基线漂移后要重新 rebase，而且**补丁和 `lib/*.so` 必须来自同一次构建**
  （库是从改过的 llama.cpp 编出来的，补丁对不上就意味着库的来源不明）

## 验证补丁没写错

在**已经应用**的所有改动的工作树上，反向检查应当通过：

```sh
git apply --reverse --check /path/to/llama-layer-split.patch
```

它证明补丁完整描述了当前状态（多一个 hunk 或少一个都会失败）。
注意生成补丁时**不要**加 `-c core.autocrlf=false`：Windows 上那会把行尾差异当成
「整个文件重写」，`llama-kv-cache.cpp` 会从 47 行变成 6227 行，补丁从 69 KB 涨到 596 KB。
