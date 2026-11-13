# 层切分引擎（多进程 pipeline）

> **这份文档的权威版本在工作区 `docs/60-llama-split/`**（那棵树由 `tools/gen_index.py` 校验链接，
> 并且覆盖 PMS / Qwen-Image / gemma4 全部项目）。这里是搬进代码仓库的副本，好让代码和设计说明一起走。
> 两份不一致时**以工作区为准**；同步方式见文末。
>
> 代码：本仓库根目录（部署到设备 `/data/local/tmp/llama.cpp`）。
> llama.cpp 侧的层切分改造以补丁形式放在 [`patches/`](../patches/)（基线 `7d7f028`）——
> **补丁在本仓库，llama.cpp 源码需要自己 clone**。

## 这个块是什么

一个模型装不进单进程 VA 窗口时，**按层切成 K 段、每段一个进程、隐藏态走共享内存串起来**。

```
段 A[0,11)  ──hidden 16 KiB──▶  段 B[11,24)  ──hidden──▶  段 C[24,32)+lm_head(在 DSP 上)
   │                                                          │
   └──────────────────── next_token ◀─────────────────────────┘
```

它**不是加速手段**：单 CDSP、算力不叠加、同一时刻只有一段在算。
它是「把 5.2 GiB 的模型塞进 2048 MiB 的 IOVA 窗口」的唯一办法（`L-026`、`L-054`）。

但**切法本身能提速**：把 `output.weight`（Q6_K 796 MiB）放到 DSP 上、末段相应缩到 8 层，
decode 从 3.36 → **5.05 t/s**（`L-058`）。这是 `--route` 存在的理由。

从哪读起：

| 想知道 | 读 |
|---|---|
| **它现在是什么形状、要通用化还缺什么** | [`architecture.md`](architecture.md) ★ 主线 |
| 怎么编、怎么跑、怎么验数值、切点怎么选 | [`runbook.md`](runbook.md) |
| 冷启动 2.8 s 花在哪、怎么摊掉 | [`create-overhead.md`](create-overhead.md) |
| **别人家的段怎么插进来** | [`architecture.md`](architecture.md) §5.3.1（stage ABI + `--ext-after`） |
| 一句话结论（40 条，含被否决的方案） | [`ledger.md`](ledger.md) |

**切点不用再人肉凑了**：`python plan_split.py <model.gguf>` 读 GGUF 的真实逐层字节
（9B 是 112.5~133.7 MiB，层与层不一样重）+ DP，连 `lm_head` 放哪一起算，直接给出
`--split` + `--route` 与预估 t/s（9B 上误差 0.6%）。`--va*` 覆盖三段各自的窗口预算，
`--all` 看所有可行段数。详见 `architecture.md` §5.1.1 / §5.1.2。

## 现状（2026-11-10）

| 项 | 数值 | 配置 |
|---|---|---|
| 模型 | Qwen3.5-**9B** Q4_K_M（5.2 GiB / 32 层 / n_embd 4096 / n_vocab 248320） | 三段 `11,24` |
| 段内存 | 1496 / 1713 / 1902 MiB（含末段的 796 MiB lm_head） | ctx 8192 × 2 slots |
| **decode（单请求）** | **5.05 t/s**（198 ms/token） | `--slots 2 --batch 64` |
| **decode（双并发）** | **5.77 t/s**（1.14×） | 同上 |
| prefill | 260 / 244 t/s | `--batch 64~256` 分块 |
| 上下文 | **8192**（KV 只 128 MiB/slot，混合注意力仅 1/4 层有 KV） | `-c 8192` |
| 数值 | 与 CPU 逐位一致 `11751 13 198 760 6511 314 9338 369` | `L-048` 铁律 |
| 冷启动 | ~3.3 s（三段并行，wall = max） | 每段 1.9~2.6 s 建 ctx |
| 服务 | `http://127.0.0.1:7788`，流式 + 停止生成 + 真实模型名；**每次生成一条独立设备连接** | `serve_http.py` |
| **lm_head** | 在 DSP 上 21 ms（原来 CPU 上 75 ms） | `--route '.*output\.weight=htp'` |

每 token 的账（旧配置，lm_head 在 CPU）：A 72.5 + B 66 + C 121 = 259.5 ms 计算，端到端 250 ms ——
**IPC 落在噪声里（≤1%）**，详见 `architecture.md` §3。lm_head 挪到 DSP 后每层还从 6.2 降到 4.6 ms，
整链 267 → 198 ms（`L-058`）。

## 五条容易踩的硬约束

1. **父进程绝不能碰 llama**：加载过模型再 fork，子进程第一次分配 session 就 SIGSEGV（`L-033`）。
   代价是每段各读一份权重。
2. **元数据必须跟槽走，且收到事件时立刻快照**。放全局 `shm_hdr` 会被上游覆写；
   快照晚了会读到下一块的值 —— 表现为 tg 输出乱序、答非所问（`L-047`）。
3. **CANCEL 只能断在 token 边界**：那是三段唯一同时空闲、事件计数平衡的位置（`L-052`）。
4. **三段的天花板不一样，而且 init 起来不代表装得下**：大 buffer 首次 decode 才 mmap，
   超窗要跑一次真实生成才会 `fastrpc_mmap failed`。首段 1496 过 / 1618 崩，末段 1902 过 / 2023 崩（`L-059`）。
5. **`SplitClient(port=DEVICE_PORT)` 这种默认参数会骗人**：默认值在类定义时就绑定了，
   `main()` 里改端口对它无效，表现为「指定 18081 却一直连 18080」（`L-056`）。
   同类坑：两个来源不同的 adb 混用会互相杀 daemon，`forward` 静默失效 —— 只保留一个。

## 已经探底、别再试的（详见 `architecture.md` §7）

- **优化 IPC**：总成本 2.5 ms/token（1%）。futex / 自旋 / f16 传输 / 去 `ev_free`，上限 1.2%。
- **单请求内 batch > 1**：自回归必须串行，批内成本反而翻倍。
- **加线程数**：decode 是带宽受限，4 线程已最优。
- 真正的开销在**每段各跑一次 `llama_decode` 的固定开销**（一个边界 ≈ 15 ms），
  只能靠「减段数」或「一次摊更多 token（MTP）」省。

## 与其他块的关系

这三个块属于原工作区的文档树（`docs/` 下与本目录平级的其他块），**不在本仓库**：

| 块（工作区 `docs/`） | 关系 |
|---|---|
| `30-gemma4-e4b/` | gemma4 走的是**硬件异构**（NPU prefill + CPU decode），不是层切分；两者可叠加 |
| `20-qwen-image-qnn/` | 同一套「分段 + 共享内存」思路的另一个实例（11 段链）；`architecture.md` §5.3 的 ABI 想统一这两者 |
| `50-platform/` | 运行时/工具链/adb 坑都在那里 |

## 扩展它：写一个自己的 stage

```sh
bash build_stage.sh my_stage.c        # 只需要 #include "stage_abi.h"，别链接 llama.cpp
llama-split ... --split 11,24 --ext-after 0:./my_stage.so
```

契约是 `stage_abi.h` 里的三个函数（open/run/close）+ `split_hidden`（带 dtype）+ `split_slot_meta`。
参考实现 `stage_pass.c` 同时是测试件：`STAGE_PASS_BIAS=0` 必须与不插 ext 时逐 token 相同，
`=5000` 必须变样。**先跑这两个再谈你的 stage 接对了没有。**

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
