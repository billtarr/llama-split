# 双进程服务的初始化开销：怎么把 create 摊掉

层切分每次冷启动要付一笔固定的初始化代价——模型加载 + HTP session/context 创建，实测约 **2.8 s**。
服务化做的事，就是把「每请求付一次」变成「每进程付一次」。

本文只讲这件事：钱花在哪、省掉多少、具体怎么省、什么情况下不该省。
链路本身（切点、数值验证、8k 上下文）见 `runbook.md`；
结论依据见 [`ledger.md`](ledger.md) 的 `L-043`、`L-042`、`L-038`、`L-033`。

## 1. 冷启动的钱花在哪

Qwen3.5-4B Q4_0，`-c 8192 -k 16 --batch 256`，两段各加载一遍。日志打点实测：

| 阶段 | A 段（第 0–15 层） | B 段（第 16–31 层 + lm_head） |
|---|---|---|
| `llama_backend_init()`（DSP 起来） | 35 ms | 28 ms |
| `llama_model_load_from_file()`（读 2.45 GB + 分配 buffer） | 739 ms | 1314 ms |
| `llama_init_from_model()`（**建 KV/状态 + graph reserve**） | **1270 ms** | **1424 ms** |
| 合计 | 2044 ms | 2766 ms |

两个关键读数：

**ctx init 比模型加载还贵。** A 段 62%、B 段 51% 的时间花在 `llama_init_from_model()` 上。
这一块才是狭义的「create 开销」——它要按段大小建 KV cache 和循环状态、给每个后端分配
compute buffer、跑一次 worst-case 的 `sched_reserve`。模型文件有页缓存可以复用，这一块不行。

**两段是并行初始化的，所以 wall 只有 2.8 s。** A、B 各自 `fork` 之后就各干各的，wall 取
`max(2044, 2766) = 2766 ms`，不是 `2044 + 2766 = 4810 ms`。等于免费省掉近一半。

## 2. 每请求还剩多少

服务模式下，请求之间只做 `llama_memory_clear()`（清 KV 元数据 + 数据）+ 重建 sampler 链，
不碰模型、不重建 context。实测（HTTP 端到端，含 tokenize 与 detokenize）：

| 场景 | 端到端 |
|---|---|
| 短请求（23 prompt + 1 out），KV 近空 | 300 / 316 / 331 / 391 ms |
| 先灌满 7148 token KV，再发同一个短请求 | 344 / 316 / 327 ms |

**清 KV 的成本测不出来。** 灌满 7k 上下文之后紧接着的短请求，和 KV 为空时是同一档。
原因是按层切分后每段只有 `KV 128 MiB / RS 25 MiB`（`L-038`），而且 qwen35 是混合注意力，
32 层里只有每 4 层一个 full attention 有 KV，其余是线性注意力、状态大小与上下文无关（`L-042`④）。

对比一下「不这么做」的代价：

| 做法 | 每请求付出的时间 |
|---|---|
| 每请求从头跑一遍 | 进程 fork + 2.8 s |
| 模型常驻、每请求 `llama_init_from_model()` 重建 ctx | **1.3~1.4 s** |
| **模型常驻 + ctx 常驻 + 清 KV** | **~0.1 s**（扣除 prefill/decode 后的净开销） |

也就是说，即使模型已经加载在内存里，只要每请求重建一次 context，就还要多付 1.3 s；
把这个也省掉，才有 10 倍量级的差别。

## 3. 具体怎么省

### 3.1 A 段变成服务器，模型只加载一次

`--serve <port>` 模式下，A 段加载完就走 `accept()` 循环，B 段在 `ev_wait()` 上等活。
进程一次起、长期活，请求从 socket 进来。这是省掉 create 的前提——没有常驻，其他都无从谈起。

### 3.2 请求间只清 KV，不重建 context

```cpp
// 每个请求开始
pc.hdr->cmd = 2; ev_post(pc.ev_ab); ev_wait(pc.ev_ba);   // 通知 B 重置
llama_memory_clear(llama_get_memory(ctx), true);
```

sampler 链（top_k / top_p / temp / dist）每请求重建，但那是纯 CPU 的几个 malloc，量级在微秒。
真正贵的是 context，一律复用。

### 3.3 不要让父进程碰 llama

理论上最省的做法是：父进程先加载一次模型，`fork()` 出去让两个子进程共享同一份权重
（copy-on-write），加载时间直接砍半。**这条在 v73 上走不通**：

父进程只要碰过 llama（哪怕只是 `llama_backend_init()`），`fork()` 出来的子进程继承了
fastrpc / DSP 的全局状态，第一次分配 session 就 SIGSEGV（`L-033`）。

所以驱动器里 `n_embd` 是必填参数——父进程连模型元数据都不能去读，tokenize 也只能在 A 段里做。
代价是两段各加载一份 2.45 GB；收益是能跑。并行初始化把这个代价压到了 `max` 而不是 `sum`。

### 3.4 顺带省掉的几项

这几条不是为 create 做的，但都落在初始化路径上：

| 项 | 省了什么 |
|---|---|
| 切分时**不建 MTP 层**（`blk.32`，q4_0 ≈100 MB） | 每段少读 100 MB、少分一份 buffer |
| **KV / 循环状态按层切分**（`L-038`） | ctx init 的分配量直接减半（`KV 8→4 MiB`、`RS 50→25 MiB` @ctx64） |
| `--batch 256` 而非 `n_ctx`（`L-042`） | `sched_reserve` 从 459 ms（bs=8192）降到几十 ms，且不撞 3.14 GB 的 `max_vmem` |
| `token_embd` / `per_layer_*` 钉 CPU（`-ot` 等价） | 这两块合计 >2 GB，放 HTP 侧会让每段都贴窗口上限 |

### 3.5 一次性模式别硬套

`llama-split -p "..." -c 64 -n 8 -k 16` 这种跑法**不该**走服务路径：它跑完就退出，
常驻的那套逻辑（accept、清 KV、sampler 重建）全是空转。测数值、做 A/B 对照时用一次性模式，
只有「同一个模型要回答很多次」才值得上服务。

## 4. 什么时候这套不划算

- **只跑一两次**：初始化的 2.8 s 本来就只付一次，常驻没有意义。
- **模型本来装得下**：0.8B 这种，切分纯亏 decode 16%（`L-030`），不切就没有两个 create。
- **需要多客户端并发**：现在服务端单连接串行，第二个客户端会等到第一个断开。
  并发要靠排队而不是再加载一份——两份额外权重会把 IOVA 窗口吃穿。

## 5. 复现

```bash
# 设备侧：一次加载，常驻
adb shell "sh /data/local/tmp/serve.sh 18080"
adb forward tcp:18080 tcp:18080

# PC 侧
python serve_http.py 7788 18080
```

冷启动分解直接看日志开头的三行：

```bash
adb shell "grep -E '^\[[AB]\] (backend init|model load|ctx init)' /data/local/tmp/split-A.log /data/local/tmp/split-B.log"
```

每请求开销用 `history/probe_reuse.py`：先跑四个短请求，灌一个 7k prompt，再跑三个短请求，
比较两组耗时——差值就是「清一个满 KV」的代价。

## 6. 坑

- **`adb forward` 会随 adb server 重启丢失**，表现为 `ConnectionRefused`。重跑一次
  `start_server.sh` 即可，设备侧的常驻进程不受影响。
- **服务端一次只服务一个客户端**（模型单实例）。第二个连接不会报错，会静静地等。
- **客户端中途断开，服务端仍会把当前请求生成完**——没有取消机制，4k 输出的请求会白跑 6 分钟。
- 改了 `--batch` 或 `-c` 必须重启设备侧进程，那是初始化参数，不属于每请求可调的。
