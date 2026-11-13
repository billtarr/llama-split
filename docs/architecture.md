# 层切分引擎：现状架构与通用化路线

> 最后更新：2026-11-10
>
> 这份文档讲三件事：**它现在是什么形状**、**为什么是这个形状**、**要变成通用引擎还缺哪四层**。
> 操作细节（怎么编、怎么跑、怎么验）见 `runbook.md`；初始化成本见 `create-overhead.md`；
> 一句话结论见台账 `L-028` ~ `L-054`。

---

## 0. 一句话

模型装不进单进程的 VA 窗口时，按层切成 K 段、每段一个进程、隐藏态走共享内存串起来。
**它是「内存扩容」手段，不是加速手段** —— 单进程多 context 装得下的时候，这套就不该用。

## 1. 它真正在解决什么

一个容易被说错的点：**多进程省的是地址空间，不是内存带宽。**

| 说法 | 对不对 | 实情 |
|---|---|---|
| 「多进程解决了单进程运行时内存来回搬运的开销」 | ❌ | 恰恰相反：分进程**新增**了每 token 2×16 KiB 的跨进程搬运 |
| 「单进程 2 GB 的 IOVA 窗口装不下 5.2 GiB 模型」 | ✅ | 这才是分进程唯一的硬理由（`L-054`、`L-026`） |
| 「分进程能让两段并行加速」 | ❌ | 共用一颗 CDSP，算力不叠加（`L-029`）；同一时刻只有一段在算 |

v73 上 DSP 侧可映射的权重总量约 **2048 MiB**（VA 窗口 2304 MB 扣除运行时占用）。9B Q4_K_M 是
5.2 GiB，**必然**要分。8 Gen 3 的 VA 有 7.65 GB，整链能装进一个进程，那种机器上分进程
只剩坏处（每段多付一次 per-call 开销）。

→ 所以通用实现的第一个设计约束：**「单进程多 context / 多线程 / 多进程」三种形态必须可切换，
按 VA 约束自动选**，不能像现在这样把分进程写死。

## 2. 现状架构

### 2.1 进程与数据流

```
                    ┌──────────────────────────────────────────┐
   行协议 (TCP)      │  父进程：只做 fork / waitpid / 端口转发     │
   INFO/GEN/CANCEL  └──────────────────────────────────────────┘
                                  │ fork × K
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
   ┌─────────┐   hidden   ┌─────────┐   hidden   ┌─────────┐
   │ stage A │──────────▶│ stage B │──────────▶│ stage C │
   │ 层[0,11)│  shm slot  │层[11,22)│  shm slot  │层[22,32)│
   │ 含嵌入   │            │         │            │ +lm_head│
   └─────────┘            └─────────┘            └─────────┘
        ▲                                             │
        └──────────────── next_token ◀───────────────┘
```

- **父进程从不碰 llama**。父进程一旦加载过模型再 fork，子进程第一次分配 session 就 SIGSEGV
  （`L-033`），所以每段各读一份权重。
- **段 0 兼任 HTTP 服务端**（`--serve <port>`），其余段是纯 worker。
- **父进程 `waitpid(-1)` 监护任一段**：任何一段退出就 `SIGTERM` 其余并 `exit(2)`，
  避免「一段死了、段 0 硬撑占端口」的半死状态（`L-050`）。

### 2.1.1 一条硬前提：fork 的父进程必须从未碰过 backend

上面那句「父进程从不碰 llama」不是风格偏好，是**唯一被验证过不崩的写法**（`L-067`）。
它的反面在两个项目上都付过代价：

| 做法 | 结果 |
|---|---|
| **fork 发生在任何 backend init 之前**（本仓库） | 天天在用：9B 三段常驻服务、`--slots 2` 并发、反复重启 |
| 父进程加载完模型再 fork（想共享权重省内存） | 子进程第一次分配 session 直接 SIGSEGV（`L-033`、`L-044` ⑤） |
| 父进程 HTP 已在跑，再 fork+exec 出另一个 HTP 进程 | 本项目不触发；`qwen-image-split` 那边挂过，但**真因是 OOM**（见下） |

三个推论，每条都被实测钉过：

1. **「多个进程同时在初始化 HTP」本身不是问题 —— 别为此加 sleep。**
   `v2` 是 `for (i...) fork()` 紧凑循环（`llama-split.cpp:2156` 注释
   *fork workers first; only stage 0 ever starts threads*），父进程中间不等待，
   K 个子进程的 `llama_backend_init()`（`v2_stage0` 内）是**真并发**。
   `L-044` ③ 实测两段初始化 wall = `max(2044, 2766) ≈ 2.8 s`，不是相加。
   → 所谓「错开启动避免撞车」既治不了病，还会把并行初始化退化成串行，白亏时间。
2. **exec 补救不了。** 子进程 `execv` 之后地址空间是干净的，但 **fork 那一刻**已经复制了
   父进程全部 fastrpc / dma_buf 的 VMA 与 fd 表；exec 拆这些 VMA 时仍要过驱动。
   `qwen-image-split` 那次连 `close(3..1024)` + 6 s 错峰都加了，照样挂（`I-028`）。
3. **代价必须认：每段各读一份权重。** 父进程碰过 backend 就没法靠 fork 共享页表，
   于是这里让子进程各自 mmap gguf —— 文件页缓存是系统级共享的，RSS 会重复计，
   物理内存只存一份（`L-044` ⑤ 记的就是这笔账）。

**移植检查项**：想「父进程先加载、再 fork 共享权重」时，先确认要共享的东西里有没有**已打开的
后端会话**（fastrpc PD / dspqueue / rpcmem block）。有就放弃，改成「一个什么都不做的 launcher
先 fork 一批 worker，再各自 init」—— 也就是本仓库的形状。**先 fork 再 init，永远不要反过来。**

> ⚠️ **别把这条当成「整机 hang 的解释」（11-11 更正）**：`qwen-image-split` 那两次死机拔掉
> `SYSTEM_LAST_KMSG` 后确认是 **OOM → OOM killer 杀 SystemUI/systemservice → OPPO DeathHealer
> 主动 kernel panic**（`I-029`），与 fork 语义无关。本项目撞不上那条，真正原因是**内存量级**
> —— 每段 ~1.4 GB、三段合计 ~4.5 GB；对面每个进程都加载完整 TE（4.8 GB），三份就顶满 23.6 GB。
> 结论：**fork 前别碰 backend 是代码级规则（防 SIGSEGV），内存预算才是「会不会死机」的规则。**

### 2.2 共享内存布局

一个文件（默认 `/data/local/tmp/llama-split.shm`）`mmap` 进所有进程：

```
偏移 0                    hdr_bytes (4 KiB 对齐)
├─ shm_hdr                全局：cmd / next_token / done / 采样参数
│
├─ slot[0]  ─┬ slot_hdr   n_tok / pos / sample        (64 B 对齐)
│            ├ hidden     float32 × n_embd × n_batch
│            └ toks       int32  × n_batch
├─ slot[1]  同上
└─ slot[K-1] 同上
```

**每个 stage 一个槽**，槽大小按 `n_batch` 定：9B（n_embd 4096 / batch 256）单槽 4.0 MiB。

两条设计约束是踩出来的，不是设计出来的：

1. **元数据必须跟槽走，不能放全局 `shm_hdr`。** 三段时 A 段一边推下一个 prefill 块，
   下游还在算上一块；全局字段会被上游覆写，出现 `decode failed n_tok 133 pos 1024`（`L-047`）。
2. **槽位头要在收到事件时立刻快照。** 一旦 `post(ev_free_in)` 之后再去读，
   读到的就是下一块的值 —— tg 输出乱序、答非所问的真正原因（`L-047`）。

### 2.3 同步：每个 token 的时序

事件分三类：

| 事件 | 方向 | 语义 |
|---|---|---|
| `ev_down[i]` | i → i+1 | 槽里有新数据（含长度、位置、是否要采样） |
| `ev_up[i]` | i+1 → i | 槽已消费完，可以覆写 |
| `ev_tok` | 末段 → A | 新 token；reset/quit 的应答也走它 |

三段 decode 一个 token 的完整时序：

| 步 | 谁 | 动作 | 发什么事件 |
|---|---|---|---|
| 1 | A | decode 11 层 → 写 hidden + slot_hdr | `ev_down[0]` |
| 2 | B | 醒 → 快照 slot_hdr → 拷 hidden | `ev_up[0]`（A 可覆写槽） |
| 3 | B | decode 11 层 → 写 hidden | `ev_down[1]` |
| 4 | C | 醒 → 拷 → decode 10 层 → 采样 | `ev_tok` |
| 5 | A | 醒 → 取 `next_token` → 输出 → 下一轮 | — |

**唯一的空闲点是「等 `ev_tok`」**，所以 CANCEL 只能断在 token 边界 —— 那时三段都空闲、
事件计数平衡，停完管道仍然一致（`L-052`）。这是设计上的一个硬约束，不是偷懒。

### 2.4 控制面：两层协议

**外层（TCP 行协议）**，客户端 → 段 0：

```
INFO                                  → INFO <model> <n_ctx> <n_stage> <n_embd> <n_vocab> <split>
GEN <rid> <n_predict> <temp> <top_k> <top_p> <seed> <nprompt> <tok...>
                                      → P <rid> <hex-text> ...  T <rid> <tok> ...  E <rid> ok <n> <ms>
CANCEL                                → E <rid> cancelled <n> <ms>
```

**内层（shm 里的 `cmd`）**，段 0 → 下游：`1` = 一块数据 / `2` = reset（清 KV、重建 sampler）/
`3` = quit。内层是广播式的：段 0 写 `cmd` 后 post 自己的 `ev_down`，其余段各自转发。

### 2.5 层的可见性（llama.cpp 侧的改造）

`LLAMA_LAYER_SPLIT=first:last` 环境变量，改了三处：

| 位置 | 改了什么 | 为什么 |
|---|---|---|
| 模型加载 | 只建 `[first,last)` 的层张量 | 省 VA（这是切分的目的） |
| `llama_kv_cache::init` / `llama_memory_recurrent::init` | `il` 不在范围内 → `continue` | 不建不属于自己的 KV/状态（`L-038`） |
| 建图 | `first>0` 时把最后一层输出当图输出；中间段**全行**标 logits | 否则 prefill 卡在 768 行（`L-047`） |

混合注意力模型（如 qwen35）只有 1/4 层有 KV，所以 9B/ctx8192 的 KV 只 128 MiB —— 这也是
8k 上下文能跑起来的前提。

### 2.6 张量归属

`tensor_buft_overrides` 把非层张量按正则钉到 CPU：

```
.*per_layer.*      → CPU   （PLE，gemma4 的 [10752,262144] 巨表）
.*token_embd.*     → CPU
.*output\.weight   → CPU   （9B 是 796 MiB，放 HTP 会挤爆 A/B 段）
```

`SPLIT_OUTPUT_HTP=1` 可放回 NPU，`SPLIT_OT_EXTRA` 追加正则。

**这是当前最"特化"的一处**：它只能表达「进 CPU 还是进 NPU」，表达不了「进第 3 段」。

## 3. 成本结构（全部实测）

### 3.1 每 token 的账：IPC 只占 1%

三段 9B（`--split 11,22 -c 1024 -n 12`），同一次运行的打点：

| 项 | 耗时 | 占比 |
|---|---|---|
| A 段 decode（11 层，HTP） | 72.5 ms | 29% |
| B 段 decode（11 层，HTP） | 66 ms | 26% |
| C 段 decode（10 层 + lm_head，CPU） | 121 ms | 48% |
| **Σ 计算** | **259.5 ms** | — |
| **端到端** | **250 ms（4.00 t/s）** | — |

Σ 与端到端在测量噪声内相等。IPC 那部分是独立微基准测的（`--ipc-bench`）：

| 项目 | 耗时 |
|---|---|
| 同尺寸本地 memcpy（16 KiB） | <0.05 µs，测不出 |
| eventfd 往返（对端停在 poll） | 129.8 µs |
| 往返 + 16 KiB 跨进程拷贝 | 178.2 µs |
| **对端空闲 20 ms 后被唤醒** | **828.8 µs** |

decode 时下一段总是「睡了几十毫秒」才被叫醒，所以要按 828.8 µs 算：
**3 次真实唤醒 ≈ 2.5 ms，占 1%**。采样 `llama_sampler_sample` 只 0.3 ms。

→ **结论：IPC 已经不是瓶颈**（`L-053`）。换 futex / 自旋等待 / hidden 改 f16 / 去掉 `ev_free`
全加起来最多省 3 ms（1.2%），见 §7。

### 3.2 段边界的 15 ms 是什么

4B 上做过干净的对照：两段 **7.05 t/s** → 三段 **6.37 t/s**，多一个边界 = **15 ms/token**。

这 15 ms 里 IPC 只占 ~2.5 ms，剩下是**每段都完整跑一次 `llama_decode`** 的固定开销：

- graph build（每 token 重建整个子图：几百个 op 的拓扑 + 内存规划）
- HTP submit + fastrpc 同步
- 图边界的 cache 失效（上一段的输出要回到 DDR 再被下一段读）

**段数 = 调用次数。** 单进程 32 层付 1 次，三段付 3 次。
所以省它的唯一办法是**少调用**（减段数 / 一次算多个 token），不是**加快调用**。

### 3.3 初始化：每进程付一次，不是每请求

冷启动 2.8 s（wall = `max(A, B)`，两段并行 fork 后各干各的），其中 `llama_init_from_model()`
占一半以上。服务模式下请求之间只清 KV，实测落在噪声内。详见 `create-overhead.md`。

---

## 4. 特化在哪：七条

要谈通用化，先得把「现在写死了什么」列清楚：

| # | 特化点 | 现状 | 后果 |
|---|---|---|---|
| 1 | 切分点人工填 | `--split 11,22` | 换模型/换设备要重新人肉凑 VA |
| 2 | 只切 transformer block | `embd`/`lm_head` 靠正则钉 CPU | 表达不了「这个张量去第 3 段」 |
| 3 | 每段必须是同一 gguf 的层子集 | 同构、同量化 | 异构量化/异构后端做不到 |
| 4 | stage 固定是 llama.cpp | 没有 ABI | 接不了 QNN 子图 / ORT / 远端 |
| 5 | 固定串行链、单请求 | 每段 1 槽 | **任意时刻只有 1/3 的设备在算** |
| 6 | 只有「多进程」一种形态 | 写死 fork | VA 够的机器上白亏 per-call |
| 7 | 一段死全链死 | 外部 `sup9b.sh` 3 s 重启整链 | 重建 2.8 s，且丢请求 |

---

## 5. 通用化：四层 + 两个附条件

一句话：**把它从「分段加载器」做成一个真正的 pipeline-parallel 引擎。**

```
┌─────────────────────────────────────────────────────────────┐
│ ① 规划层   设备画像 + 代价模型 → DP → (tensor → stage) 映射    │
├─────────────────────────────────────────────────────────────┤
│ ② 切分粒度  层粒度 → tensor 粒度（一个模型可以切在任意边界上）  │
├─────────────────────────────────────────────────────────────┤
│ ③ 执行层   stage ABI：与后端解耦，llama.cpp 只是其中一种实现    │
├─────────────────────────────────────────────────────────────┤
│ ④ 调度层   串行链 → 请求流水（1F1B），把空闲的段填满           │
└─────────────────────────────────────────────────────────────┘
        附：⑤ 形态可切换（进程/线程/单进程多 context）  ⑥ 单段自愈
```

### 5.1 规划层：把人工凑数换成求解

输入三样：

```c
struct device_profile {
    size_t va_window;          // 2048 MiB（v73） / 7.4 GiB（V75）
    size_t ram_budget;         // 各后端可用内存
    double ddr_bw_gbps;        // 决定性指标：decode 是带宽受限
};

struct layer_profile {         // 由 -v 打点直接采，不用估
    size_t weight_bytes;       // 逐层实际字节（Q4_K_M 的层并不等重）
    double ms_per_token;       // 该层在某后端上的单 token 耗时
};
```

代价模型（v1，够用）：

```
cost(split) = Σ_i  T_compute(layers_i, backend_i)
            + (K-1) × T_boundary
            + (Σ_i bytes_i > va_window ? ∞ : 0)     // 硬约束

T_boundary = 15 ms      // §3.2 实测；含 per-call 固定开销 + 唤醒 + 拷贝
```

两个不同的目标函数切法完全不同，必须显式选：

- **最小化单请求延迟** → 按计算量均分，让最慢的段尽量短（现在是这个）
- **最大化吞吐** → 按每段耗时比例切，让流水线的瓶颈段最短（配合 ④ 才有意义）

输出不是 `--split 11,22`，而是一张映射表：

```c
struct stage_desc {
    int  id;
    int  layer_begin, layer_end;   // -1 = 不承担层
    int  n_embd_in, n_embd_out;    // 不再假设全链同宽
    bool owns_embedding;           // 查表段
    bool owns_output;              // lm_head 段
    bool emits_logits;             // 是否采样
    const char * backend;          // "htp" / "cpu" / "qnn:<path>" / "rpc:<host>"
};
```

### 5.1.1 已实现：`plan_split.py`

`python plan_split.py <model.gguf> [--ctx 8192] [--batch 64] [--slots 2] [--lmhead auto|cpu|htp] [--goal latency|throughput] [--all]`

读 GGUF 里**真实的逐层字节**（Qwen3.5-9B Q4_K_M 是 112.5 ~ 133.7 MiB，层与层不一样重），
把 `token_embd.weight` 当作钉 CPU、把 `output.weight` 当作**可选的落点**（`--lmhead`），
再加上 KV 份额和 compute buffer，然后 DP 搜切点。

| 输入 | 输出 |
|---|---|
| 每层真实字节 + 三段各自的预算 + ctx + batch | 每段层区间、VA 占用、预估 ms、**可直接复制的 `--split` + `--route`** |

代价模型的标定系数是 **`--ms-per-mib 0.048`**（v73 实测 66 ms / 11 层 × 125 MiB，lm_head 在 CPU 时）。
按字节缩放而不是按层数，所以同一个系数能跨模型用；lm_head 挪到 DSP 时整体乘 0.77（见 `L-058`）。

| 模型 | 规划器给 | 实测 | 预估单请求 | 实测单请求 |
|---|---|---|---|---|
| Qwen3.5-**9B** Q4_K_M | K=3，`--split 11,24` + lm_head 上 DSP | 同左 | 5.02 t/s | **5.05 t/s** |
| 同左，`--lmhead cpu` | K=3，`--split 11,22` | 同左 | 3.36 t/s | 3.36 t/s |
| Qwen3.5-**4B** Q4_0 | K=2，`--split 14` | 人工 `-k 16`（5.95 vs 7.05 的偏差来自 tied embedding） | 5.40 t/s | — |
| Qwen3.5-**0.8B** Q8_0 | **K=1，不用切** | — | 10.07 t/s | — |

三个方向都对：9B 能准确复现实测最优（含 lm_head 落点），4B 给出均衡两段，0.8B 判定「装得下就别切」。

**`--goal` 的语义**（两者切法不同，详见 §5.1 的说明）：

- `latency`（默认）：取**最少可行段数** —— 每多一段就多 15 ms 边界税，而层耗时总和不变
- `throughput`：取**最慢段最短**的切法

切点本身一律按「最慢段最短」求，因为即使目标是延迟，均衡的切法也不会让某一段成为瓶颈。

### 5.1.2 修正：VA 预算不是「每段 2048 MiB」

规划器第一版用 `--va 2048`（对外公布的窗口大小）。实测发现**三个位置的可用上限完全不同**，
而「起来的了」不等于「装得下」——大 buffer 是**首次 decode 才 mmap**，init 阶段全部 ready
也会在第一次真实生成时 `fastrpc_mmap failed`。

| 位置 | 通过 | 崩 |
|---|---|---|
| 首段（带 embedding 分支） | 11 层 **1496 MiB** | 12 层 **1618 MiB** |
| 中间段 | 13 层 **1713 MiB** | — |
| 末段（8 层 + lm_head 796 MiB） | **1902 MiB** | 9 层 **2023 MiB** |

（9B Q4_K_M，ctx 8192 × 2 slots。首段最紧是因为 `build_inp_embd` 多一条 GET_ROWS 分支。）

据此规划器改成三个独立预算 `--va-first / --va / --va-last`（默认 **1500 / 1800 / 1950**），
对 9B 复现出实测最优的 `--split 11,24`，预估 **5.02 t/s** vs 实测 **5.05 t/s**（误差 0.6%）。

**操作纪律**：换设备或换模型时，`plan_split.py` 给的是上界估计，必须再跑一次真实生成确认；
八段全 ready 后立刻解码一次，比看日志可靠。

### 5.2 切分粒度：层 → tensor

llama.cpp 已经有 `tensor_buft_overrides`（**tensor 名 → buffer type**）。
推广成两级就够：**先决定 tensor → stage，再在 stage 内决定 buffer type**。

```c
struct tensor_route {              // 顺序匹配，先中先生效
    const char * name_regex;
    int          stage_id;         // -1 = 跟随所在层
    const char * buffer_type;      // 可空，空则沿用默认
};
```

收益不止「lm_head 能独立成段」：

- 异构量化（embedding 走 Q8、FFN 走 Q4）在一个模型里能表达
- 异构后端（attention 走 NPU、FFN 走 CPU）能表达
- **MoE 的专家按段分布**（这是 2.51 才加 MoE 内核之后最值得试的一件事）

#### 5.2.1 已实现：`--route`（buffer 轴），以及为什么先做这条轴

```sh
--route 'PATTERN=cpu|htp'         # 可重复，也可用 ';' 在一个参数里给多条
```

它插在三个内置默认（`per_layer` / `token_embd` / `output` → CPU）**之前**，靠 llama.cpp
「首个匹配生效」的语义覆盖默认值；`htp` 目标由 `ggml_backend_dev_buffer_type(HTP0)` 解析。
`SPLIT_OUTPUT_HTP` / `SPLIT_OT_EXTRA` / `SPLIT_PLE_ON_HTP` 三个硬编码 env 从此只是它的语法糖。

**为什么这条轴先做**：它当天就换到了 1.50×。`output.weight` 是 Q6_K / 796 MiB，每 token 读一遍，
放 CPU 要 75 ms；放 DSP 只要 21 ms。但真正的大头是副作用——CPU 上那个大矩阵会让**每一层**都变慢
（6.0~6.6 → 4.6~4.7 ms/层，见 `L-058`）。所以「把 tensor 放到对的硬件上」不只是省它自己那点时间，
它能改变整条链的单位成本。

代价是末段必须瘦下来：10 层 + lm_head = 2051 MiB 装不下，8 层 + lm_head = 1902 MiB 才行。
**这就是规划层和切分粒度必须一起动的原因**——放哪儿是规划结果，不是常量。

| 配置 | decode | 双并发 |
|---|---|---|
| `--split 11,22`（lm_head 默认在 CPU） | 3.36 t/s | 4.84 t/s |
| `--split 11,24 --route '.*output\.weight=htp'` | **5.05 t/s** | **5.77 t/s** |

#### 5.2.2 还没做：stage 轴（层内切）

`--route` 目前**拒绝** `stage:N` 目标。原因不是没时间，而是它和 buffer 轴不是一回事：

- buffer 轴：tensor 还在同一个进程里，只是落在另一块内存，ggml 自动在切分点插拷贝 —— 无需改图。
- stage 轴：要把**一个 block 拦腰截断**（比如 attention 留 A、FFN 送 B），意味着图要在
  `ffn_norm` 这个 tensor 边界分成两张图，还要为「半个 block」定义新的段间契约（现在契约是
  `(hidden, pos, ids)`，半个 block 的中间态没有对应字段）。

所以 stage 轴要等 §5.3 的 ABI 先落地，否则会做出一个只有 llama.cpp 能表达的特例。
在那之前，`lm_head` 独立成段可以用「末段层数 = 0」的退化切分凑出来（`--split a,b,32`），
但收益被多一个边界的 15 ms 吃掉，实测不如让 lm_head 和 8 层同居。

### 5.3 执行层：stage ABI

把段间契约定死，后端就不重要了：

```c
// 一个 stage 只需要实现三个动作
int  stage_open (const stage_desc *, const char * model_path);
int  stage_run  (hidden_view in,  hidden_view out, const slot_meta * m);
void stage_close(void);

// hidden_view = { void *data; int dtype; int n_embd; int n_tokens; }
```

`dtype` 是关键：一旦 ABI 里带 dtype，f16 传输、int8 隐藏态、甚至分块稀疏都可以只改一个字段。

ABI 解耦之后，一个 stage 可以是：

| 实现 | 状态 |
|---|---|
| llama.cpp 的层子集 | ✅ 现在就是这个 |
| 任意 `.so`（stage ABI） | **✅ 已跑通**（`stage_pass.so`，见 §5.3.1） |
| QNN 编译好的子图 | 已有先例（Qwen-Image 那条链就是这么做的），接口已就位 |
| ONNXRuntime / MNN | 待接 |
| 远端机器（RPC） | 拓扑自动变「跨设备流水」 |

**切分协议不该绑定 llama.cpp** —— 这是「通用」和「只是参数化了一下」的分界线。

#### 5.3.1 已实现：`stage_abi.h` + `--ext-after`

三个文件，加起来不到 200 行：

| 文件 | 作用 |
|---|---|
| `stage_abi.h` | 契约本身：`split_stage`（版本 + open/run/close）、`split_hidden`（data/dtype/n_embd/n_tok）、`split_slot_meta`（n_tok/pos0/sample/rid/chunk_seq/ids） |
| `stage_pass.c` | 参考实现（同时是测试件）：`ldd` 只依赖 libc/libm，**不链接 llama.cpp 任何东西** |
| `llama-split --ext-after N:PATH` | 把 `PATH` 插在 llama 段 N 之后；它复用同一套共享内存环和 eventfd，只替换「算」这一步 |

```sh
bash build_stage.sh stage_pass.c                      # 产出 stage_pass.so
llama-split ... --split 11,24 --ext-after 0:/path/stage_pass.so
# 启动日志会把链的形状打出来：
#   [M] chain: llama0 ext(stage_pass.so) llama1 llama2
```

**已验证（9B，ctx1024，单槽）**：

| 测试件行为 | 预期 | 实测 |
|---|---|---|
| `STAGE_PASS_BIAS=0`（恒等） | 与无 ext 的链逐 token 相同 | **`11751 13 198 760 6511 314 9338 369` 完全一致** |
| `STAGE_PASS_BIAS=5000` | 环里必须变样 | `78398 78398 …`（8 个同号） |
| `.so` 不存在 | 立刻失败，不许挂住 | 段日志 `dlopen … not found`，整链 8 ms 退出 |

三个测试各自排除一种失败：恒等 → 数据/元数据搬运正确（若下游读到的是清过零的 shm，输出必然不同）；
强扰动 → 输出确实被下游消费（不是旁路）；缺库 → 校验在启动期，不会变成运行期卡死。

**两条设计约束**（都在代码里强制）：

1. **外部 stage 不能是最后一段**：只有持有词表的段才能采样。`--ext-after K-1` 直接报错。
2. **`--ext-after` 会强制走 v2 路径**（否则单槽时会静默忽略这个参数 —— 这是我自己踩的坑，已改成显式选择）。

**还没做的（下一棒）**：接一个真干活的 stage。QNN 子图需要处理三件事 ——
`dtype` 目前只有 f32（f16 字段已留）、行主序的 `n_tok × n_embd` 布局、以及
`run()` 每 chunk 一次调用（decode 时 n_tok=1，QNN 侧要容忍 batch=1 的反复 submit）。

> **一个教训**：第一版测试件用的是「hidden × 2」，结果输出一模一样 —— 因为每个 block 开头就是
> RMSNorm，均匀缩放被精确抵消。测试件必须扰动模型真正敏感的方向（后来改成给第 0 维加偏置）。
> 这跟 `L-058` 里「CPU lm_head 拖慢所有段」是同一类错误：**看起来在测 A，其实测了个不变量。**

### 5.4 调度层：串行链 → 请求流水

现在每段 1 槽、一个请求占满整链，各段自己的时间占比是 A 29% / B 26% / C 48%：

```
单请求（v1）：A ███░░░░░░  B ░░░███░░░  C ░░░░░░░███
双请求（v2）：A ████████   B ████████   C ████████
```

> ⚠️ **别把「每段占比」当成「设备利用率」。** 三段跑在同一颗 CDSP 上，单请求时 DSP 本来就是
> 满负荷（A 算完 B 算、B 算完 C 算，中间没有空闲）。流水的真实收益只有一处：**C 段里
> lm_head 走 CPU 的那 ~58 ms 可以和别的请求的 HTP 工作重叠**。所以上限是
> `1 / (1 - CPU 占比) ≈ 1.3×`，不是 3×。这条是实测撞出来的（见 §5.4.1），写在这里免得再被算错。

### 5.4.1 已实现（`--slots N`）与实测

v2 路径已落地：每段 N 个 context、每段 N 个槽、`slot_hdr` 换成 `v2_hdr`（带 `rid` / 采样参数 /
`chunk_seq`）、段 0 从阻塞驱动改成「吸收 → 推 → 等」的异步状态机、每连接一个会话线程。

| 项 | 单请求 | 双并发 | 结论 |
|---|---|---|---|
| 聚合吞吐 | **3.36 t/s** | **4.84 t/s** | **1.44×** |
| 单请求延迟 | 297 ms/tok | 405 ms/tok | 并发时每个请求慢 36% |
| 数值 | `11751 13 198 760 6511 314 9338 369 11751 13 198 760` | **逐位一致** | 请求之间不串 |

1.44× 已经**顶到 §5.4 说的天花板**（1.3~1.5×）。证据：并发时 B 段 67 → 100~160 ms、C 段
128 → 100~176 ms，A 段反而不变 —— 三段在抢同一颗 DSP，不是「填满空闲」。

还有两条硬约束是实测出来的：

- **`--slots 2` 必须配 `--batch 64`。** batch 256 时第二个 context 的初始化直接卡死
  （每 ctx 52 MiB compute buffer ×2 装不下），batch 64 时每 ctx 13 MiB 才过。
  代价：prefill 块变小。
- 单请求从 v1 的 4.00 t/s 降到 3.36 t/s（−16%）。这是 v2 的调度开销 + batch 64 的代价，
  **不是为了并发必须付出的**，还能收回一部分（见下方 TODO）。

**所以 ④ 单独做，收益就是 1.44×。** 想要更多必须和 ②③ 一起：让不同段真正用不同硬件
（比如 C 段整体落 CPU），流水才有多条独立的算力线可叠。这也是为什么顺序虽然是 ④→①→②→③，
但 ④ 的价值要到 ②③ 才完全兑现。

要改四处，都是具体可落地的：

1. **槽数从 `n_stage` 改成 `n_stage × n_slot`**（无锁环形，`ev_up` 变计数信号量）
2. **`slot_hdr` 加 `seq_id`** —— 现在没有，多请求一定会串
3. **每段 worker 从「处理一个 chunk」变成「带队列的状态机」**
4. **续批**：段 0 的 prefill 可以和新到的 decode 请求交错提交（prefill 断续，正好填 decode 的空）

经典调度是 **1F1B**（one forward one backward 的推理版）：请求先依次流过所有段（填充流水线），
稳态下每段交替处理「新请求的 prefill」和「老请求的 decode」，最后再排空。

**收益：吞吐 2~3×**（受瓶颈段限制），单请求延迟不变甚至略增。
**这一步不依赖模型、不依赖后端，纯调度改动，是四层里投入产出比最高的。**

⚠️ 注意区分两种"批量"：

- **单请求内一次 decode 多个 token** → 不行，自回归本质串行
- **多请求合批**（continuous batching）→ 可行，属于这一层

### 5.5 附：形态可切换

同一个 worker 接口，三种实现：

| 形态 | 条件 | 代价 |
|---|---|---|
| 单进程多 context | VA 装得下整链 | 0 IPC（V75 属于这档） |
| 多线程 | 能共享地址空间、VA 不够 | 省 memcpy 和唤醒 syscall |
| 多进程 | VA 各自独立（v73 唯一的选项） | 每 token 2.5 ms + 每边界 15 ms |

现在只实现了第三种。规划层要能把「形态」也当成一个决策变量。

### 5.6 附：单段自愈

现在一段死 → 整链死 → 外部脚本 3 s 后重启全链（权重要重读 2.8 s）。

通用做法：supervisor 记录每段 `(model, layer range, ctx, backend)`，检测到某段退出 →
**只重建这一段**（权重还在页缓存，约 1~2 s）→ 重放当前请求。整链不必重来。

---

## 6. 里程碑（按投入产出比排）

| 里程碑 | 内容 | 验收标准 | 状态 |
|---|---|---|---|
| **M1 调度** | 5.4 的四处改造 | ≥2 并发，吞吐 ≥1.4×（原写 1.6×，按 §5.4 的天花板下修），请求间数值逐位一致 | **✅ 已完成**（1.44×） |
| **M2 规划器** | 5.1 的 cost model + DP | 换一个新模型，自动方案与人工方案差 **≤5%**，且不违反 VA | **✅ 已完成**（9B 误差 0.6%，见 §5.1.2） |
| **M3 tensor 粒度** | 5.2 的 route 表 | ~~`lm_head` 独立成段~~ / 异构量化能表达 | **一半完成**：buffer 轴（`--route`）✅ 换到 1.50×；stage 轴（层内切）推迟到 M4 之后（§5.2.2） |
| **M4 ABI** | 5.3 的 `stage_run` | 至少接一个**非 llama.cpp** 的 stage | **✅ 已完成**（`stage_pass.so`，恒等/强扰动/缺库三测；真干活的 QNN 子图是下一棒，见 §5.3.1） |

M1 的实现已进 `llama-split.cpp`（`--slots N`，默认 1 走原路径），
验证脚本 `v2test.py` / `bench_v2.py` / `verify_v2.py` 在 `test/`。

**M1 的三条待收尾**：

1. ~~**前端还是串行的。**~~ **✅ 已改（11-10）**：`serve_http.py` 改为**每请求一条设备连接**
   （生成走独立的 `dev`），`/cancel` 改成对「活跃连接集合」广播。全局 `CLIENT` 只留给
   INFO / PING / health 这类轻量命令。HTTP 层实测双并发 **1.31×**、`/cancel` 后 0.3 s 内两个请求
   同时停、取消后新请求正常 —— 详见 §5.4.2。

   > 顺带修掉一个**潜伏 bug**：`SplitClient.__init__(port=DEVICE_PORT)` 的默认参数在类定义时
   > 就绑定了，`main()` 里用命令行改 `DEVICE_PORT` 对它无效 —— 于是「指定 18081」实际连的还是
   > 18080。默认值全代码都是 18080 时看不出来，一换端口就现形。改成 `port=None` + 运行时读取。
2. **单请求 −16% 还能收回。** v2 的调度开销（每块一次 poll + 跨线程唤醒）和 batch 64 是两笔账，
   前者可以用「推之前先批量收事件」摊掉，后者只在 slots>1 时需要。
3. **会话对象故意不释放**（引擎可能还持有指针，释放会踩 mutex）。每个连接泄漏几百字节。

### 5.4.2 HTTP 层实测（前端已改成多连接）

| 项 | 单请求 | 双并发 |
|---|---|---|
| HTTP 端到端吞吐 | 3.41 t/s | **4.47 t/s（1.31×）** |
| `/cancel` 广播 | — | **0.3 s 内两个请求同时停**，`cancelled: true` |
| 取消后新请求 | — | 正常（8 tok / 2.63 s） |

比裸协议测的 1.44× 略低：HTTP 端到端多了一层 Python 中转和 socket 往返。

---

## 7. 明确不做（别再试一遍）

| 想法 | 为什么不划算 | 上限收益 |
|---|---|---|
| eventfd → futex / 自旋等待 | 总 IPC 只 2.5 ms，还要烧 CPU 抢 DDR | ≤1.2% |
| hidden 改 f16 传输 | decode 只省 8 KiB；prefill 才 2 MiB | <0.1% |
| 去掉 `ev_free` 往返 | 两次非阻塞 syscall | ~0.04% |
| 单请求内 batch > 1 | 自回归必须串行，批内成本反而翻倍 | 负 |
| 段内 create∥exec 流水 | 已在 Qwen-Image 链上判死（VA 不够） | — |
| 加线程数 | decode 是带宽受限，4 线程已最优 | 负 |
| 父进程先加载模型、再 fork 共享权重 | 子进程首次分配 session 就 SIGSEGV（`L-033`）；跨项目上更狠，直接整机 hard hang（`L-067`） | 负 |
| 给子进程「错开启动」加 sleep | 并发 init 本来就是安全的（`L-044`③），错峰只把并行退化成串行 | 负 |

**唯一值得继续投的方向是「少一次调用」和「让每次调用摊更多 token」** —— 也就是减段数、
投机解码（MTP 4 token/往返，把 15 ms 摊成 3.75 ms）、请求流水。

---

## 8. 证据索引

| 结论 | 台账 |
|---|---|
| 单进程 VA 上限 ~2048 MiB；切分只扩容不加速 | `L-026`、`L-029` |
| 按层切分的内存分配改造 | `L-038` |
| 每段一个槽、多段流水 | `L-045` |
| 8k 的墙是 compute buffer 不是 KV | `L-042` |
| 槽位头竞态与快照 | `L-047` |
| 看门狗边界、父进程监护 | `L-050` |
| CANCEL 只能在 token 边界 | `L-052` |
| **IPC 只占 1%，边界成本 15 ms** | `L-053` |
| **多进程省的是 VA 不是带宽** | `L-054` |
| 初始化 2.8 s 的分解 | `L-044` + `create-overhead.md` |

代码位置：驱动器 `llama-split.cpp`、服务前端 `serve_http.py`（均在本仓库根目录）、
llama.cpp 侧改造见 `patches/`（基线 `7d7f028`）、设备运行时 `/data/local/tmp/llama.cpp`。
