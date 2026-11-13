# 层切分（多进程）操作手册

面向 v73 / 8 Gen 2 的「一个模型拆成两个进程跑」链路：绕过单进程 ~2 GB IOVA 窗口。
结论依据见 [`ledger.md`](ledger.md) 的 `L-028` ~ `L-036`。

## 为什么要有它

v73 上 DSP 侧可映射的权重总量约 **2 GB**（`L-026`）。模型超过这条线就两种后果：

| 模型 | 单进程 | 两段切分 |
|---|---|---|
| qwen35-0.8b q8（764 MiB） | 正常 | 正常，decode −16%（`L-030`） |
| qwen35-4b q4_0（2.28 GiB） | **ctx≤64 能跑，ctx256 直接 `fastrpc_mmap failed`** | **ctx256 正常，decode −6.5%**（`L-036`） |
| gemma4-e4b q4_0（4.5 GB） | 需 `GGML_HEXAGON_HOSTBUF=0`，pp 131 / tg 6.6 | 能跑、已确定复现，但**数值仍错且与切点无关**（`L-039`，病在 B 段构造） |

切分只扩容、不加速：两个进程共用一颗 CDSP，算力不叠加（`L-029`）。

## 代码构成

| 位置 | 作用 |
|---|---|
| `patches/`（`llama-layer-split.patch`） | llama.cpp 侧：`LLAMA_LAYER_SPLIT=first:last`，只建/只跑这段层的张量，`first>0` 时把最后一层输出当图输出 |
| `llama-split.cpp` | 驱动器：fork 两进程 + 共享内存传隐藏态 + eventfd 乒乓，A 段出隐藏态、B 段出 token |
| `quantize_gguf.py` | 纯 numpy GGUF 量化器（没有可用的 x86 `llama-quantize` 时用它） |
| `serve_http.py` | PC 侧 HTTP 前端：流式、`/cancel`、每请求一条独立设备连接 |
| `stage_abi.h` / `stage_pass.c` | 外部 stage 的 ABI 契约 + 参考实现（见「扩展它」） |
| `lib/` | 运行时库 12 个 .so，换机器直接推，见 `lib/README.md` |

## 构建与部署

仓库自带 `lib/`（12 个 .so）和编好的 `llama-split`，所以**默认不需要编译任何东西**：

```bash
export MSYS_NO_PATHCONV=1        # Git Bash 会改写 /data/... 这类路径
ADB=adb                          # 或你的 adb 绝对路径
$ADB shell "mkdir -p /data/local/tmp/llama.cpp/bin /data/local/tmp/llama.cpp/lib"
$ADB push lib/*.so /data/local/tmp/llama.cpp/lib/
$ADB push llama-split /data/local/tmp/llama.cpp/bin/llama-split
$ADB shell "chmod 755 /data/local/tmp/llama.cpp/bin/llama-split"
```

什么时候才需要重新编译：

| 改了什么 | 要重编 | 代价 |
|---|---|---|
| `llama-split.cpp` / `stage_pass.c` | `./build.sh` / `./build_stage.sh` | ~30 s（docker + NDK） |
| `patches/` 里的上游改动 | llama.cpp 全量（`scripts/snapdragon/build.py`） | ~8 min 增量，产物替换 `lib/*.so` |
| 都没改 | 不用编 | — |

重编 llama.cpp 后**必须同时更新 `lib/*.so`**：host 库和 DSP applet 出自同一次构建，混用会崩。

## 跑

```bash
cd /data/local/tmp/llama.cpp
export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib

# 单进程参考（同时给出 top5 logits 与 token 序列）
./bin/llama-split -m model.gguf -p "..." -c 64 -n 16 -t 4

# 两段切分：-k 切点、--embd 模型 n_embd（必须显式给，父进程不能碰 llama）
./bin/llama-split -m model.gguf -p "..." -c 64 -n 16 -t 4 -k 16 --embd 2560

# 单段回放（调试 B 段用）：把 A 段 --dump 出来的隐藏态直接喂给 B
./bin/llama-split -m model.gguf -c 64 -t 4 -k 16 --embd 2560 --inject /data/local/tmp/inj.bin
```

其它开关：`--cpu`（纯 CPU 真值基线）、`--dump <file>`（导出 A 段隐藏态）、`--logits <file>`、
`SPLIT_OT_EXTRA=<正则>`（额外把某些层钉到 CPU，做「部分层溢出」对照）、
`--tokens 1|2`（带真实 id / 带全 0 id；**默认 1，别关掉**，理由见下）。

## ★ 数值验证的铁律：跑三次

**B 段只要收到纯 embd 的 ubatch，就会读到未初始化/未同步的 host 内存，logits 每次都不同**
（top1 都会换，幅度 ±1.5）。这条踩过一次，误判了整轮结论（`L-035`）。

所以：

1. 切分结果必须**连跑 3 次**，top5 要逐位相同；
2. 再和**单进程基线**比 top5 与 token 序列，要逐位相同；
3. 不要用「A 段隐藏态 md5 相同」代替 B 段的检查——A 稳定不代表 B 稳定；
4. 驱动器默认 `g_send_tokens=1`（随隐藏态一起送 token id），改成 0 会立刻复现抖动。

## 切点怎么选

| 架构 | 约束 |
|---|---|
| qwen35（含 0.8B / 4B） | 无结构约束，任意 K；线性注意力状态逐层独立，KV 不跨层共享 |
| gemma4-e4b | `K ≤ 22`（`n_layer_kv_from_start=24`，末尾共享层要读本进程写出的 KV） |
| 其它 | 只要层之间不共享跨层状态就能切；切完先按上面的铁律验证 |

避坑：B 段只剩 1 层曾经会崩（`n_rs==0` 时 `s_copy` 无 buffer 的断言），**已修**（`L-038`③a），
0.8B k=23 现在正常且逐位一致。

## 三段及以上（9B 这个量级）

两段只够 4B。9B Q4_K_M 每段要 2009 MiB（16 层），直接撞墙：

```
libc++abi: terminating: ggml-hex: fastrpc_mmap failed    # --split 16
```

`--split` 接任意切点列表，`-k K` 是它的单切点特例：

```bash
# 9B Q4_K_M，32 层切成 11 / 11 / 10
adb shell "sh /data/local/tmp/serve9b.sh 18080"
# 等价于 llama-split -c 8192 --split 11,22 --embd 4096 --batch 256 --serve 18080
```

| 模型 | 切点 | 每段权重 | 结果 |
|---|---|---|---|
| 4B Q4_0 | — / 16 / 11,22 | 1.22 / 1.22 GB | 都能跑，40-token 实测两段 7.05 / 三段 6.37 t/s |
| 9B Q4_K_M | 16 | A 2009 MiB | **`fastrpc_mmap failed`** |
| 9B Q4_K_M | **11,22** | **1393 / 1360 / 1262 MiB** | **正常，数值与 CPU 逐位一致** |

### 切点怎么定

按**字节**而不是层数均分——层大小差 20%（9B 是 113~134 MiB/层）。用 GGUF 累加每层的
`n_bytes`，让每段落在 1.3~1.5 GB。9B 的 32 层累加后，11/22 正好切在 1393 / 1360 / 1262。

### 空闲会自杀？不会了（但历史版本会）

看门狗只能管「请求内部」的等待。**等待下一个请求的空闲期必须是无限等待**——早期版本把
300 s 超时套在了 worker 主循环顶上，服务空闲 5 分钟两段 worker 就退出，而段 0 还占着端口，
表现就是「页面点生成没反应」。现在：空闲走 `ev_wait_forever()`，请求内部 `ev_free`/`ev_tok`
保留 300 s；父进程 `waitpid(-1)` 监护任一段，任一段死了就整链退出，交给守护脚本重启。

设备侧守护：`sh /data/local/tmp/sup9b.sh 18080`（整链退出后 3 s 重启；`touch
/data/local/tmp/sup.stop` 停止）。PC 侧一键：`start_server.sh`。

### 卡死怎么查

`ev_wait` 带 300 s 看门狗：丢事件会打印 `[M] ev_wait: no event on fd N ... giving up` 并退出，
不会再出现「客户端一直等、设备一声不吭」。排查顺序：先 `strings split-A/B/C.log | grep chunk`
看最后推进到哪一段，再看各段进程还在不在（`ps -ef | grep llama-split`，`state=Z` 说明有一段已经死了）。

### 多段特有的五个坑

这四个都只在「多段 + 长 prompt」同时出现时才炸，单块短 prompt 完全测不出来：

**(a) 最后段没有下游。** `ev_out = -1`，`ev_post(-1)` 报 `Bad file descriptor`。所有向下游发信号
的地方都要过 `to_down()` 守卫。

**(b) reset 要有人回应。** 段 0 reset 后阻塞在 `ev_wait(ev_tok)`，最后段收到 `cmd=2` 只转发不回应
就会死锁——最后段必须 `post(ev_tok)`。

**(c) 元数据不能放在全局头里。** 段 0 连续推 prefill 分块时，全局 `hdr` 里的 `n_tok/pos` 会被
下一块覆盖，下游读到的是别人的数据。症状很怪：`decode failed (n_tok 133 pos 1024)`。
修法是每个槽位带自己的 `slot_hdr`，全局头只留 `cmd / next_token / done / 采样参数`。

**(e) 槽位头里的字段必须在收到时快照，不能「用的时候再读」。** 上游推完一块就去解码下一块，
期间的槽位头已经被改成下一块的参数；下游如果这时才去读 `sample/pos`，读到的是**下一块**的值。
症状是「prompt 一超过 256 token 输出就把词序打乱」——因为末段在非最后一块上就采样了，
而且这些提前的 `ev_tok` 会在管道里堆积，让段 0 之后完全失去同步（还会显得更快）。
规则：收到块的那一刻把 `n_tok / pos / sample` 全拷进本地变量，转发只用本地值。
配套：段 0 必须**先等 `ev_free` 再往槽位里写**；每段复位时 `ev_drain()` 清掉残留计数
（`ev_drain` 要临时 `O_NONBLOCK`，否则在阻塞 fd 上循环 read 会挂死）。

**(d) 中间段必须把每一行都标记为输出。** 中间段的任务是输出**整段** hidden，只标最后一行会触发
`embeddings required but some input tokens were not marked as outputs -> overriding`，
并且 prefill 会在几百个 token 处卡死。规则：

```cpp
b.logits[i] = pc.is_last ? ((i == n_tok - 1) ? 1 : 0) : 1;   // 中间段全 1
```

### 9B 的额外约束：独立 output 表

9B 不是 tied embedding（4B 是），`token_embd`（546 MiB）和 `output.weight`（796 MiB）是两张
独立的表。`output` 默认被钉到 CPU（否则把最后一段顶爆），代价是每 token 一次 796 MiB 的
CPU matmul。想让 lm_head 回到 NPU 用 `SPLIT_OUTPUT_HTP=1`，前提是层侧还装得下。

### 9B 实测（ctx8192 / batch256 / 11,22）

| 项 | 数值 |
|---|---|
| prefill 3437 tok | 13.2 s → **260 t/s** |
| prefill 6860 tok | 28.1 s → **244 t/s** |
| decode | **3.77 t/s**（200 token；早期记的 4.6~5.1 是失同步时的虚高） |
| 冷启动（三段并行） | ~8 s（每段各读 5.2 GB） |
| 数值 | 与 CPU 逐位一致 `561 1118 21461 28974 11 279 55438 472` |

对比 4B：参数 2.07×，prefill 掉到 0.61×、decode 掉到 0.46×。

## 内存：每个 stage 只分配自己的 KV / 状态

`layer-split.patch` 里 `llama_kv_cache::init` 与 `llama_memory_recurrent::init` 都加了层范围过滤，
不开切分时与上游行为完全一致。实测（4B q4_0，K=16，ctx64）：每段 `KV 8.00→4.00 MiB`、
`RS 50.25→25.12 MiB`，正好一半；数值逐位不变。

以前两个进程各分一整份状态，长上下文 / 大模型会拖死；现在这条路通了。

## 三条路怎么选（4B + ctx1024 实测）

| 方案 | prefill | decode | 进程数 | 代码量 |
|---|---|---|---|---|
| 单进程全 HTP | **装不下（mmap 失败）** | — | 1 | — |
| 单进程 HTP + Adreno 分层<br>`-dev HTP0,GPUOpenCL -sm layer` | 82.5 t/s | **9.0 t/s** | 1 | **0 行** |
| 单进程 HTP + 末 8 层钉 CPU<br>`SPLIT_OT_EXTRA='blk\\.(2[4-9]|3[01])\\.'` | 117.6 t/s | 5.36 t/s | 1 | 0 行 |
| **两进程切 16/16** | **323~426 t/s** | **10.47 t/s** | 2 | 本仓那套补丁 + 驱动 |

- 要吞吐 → 两进程（prefill 快 4 倍、decode 高 14%）。
- 要「一条命令跑起来、不改代码」→ HTP+GPU 分层，decode 已经够用。
- CPU-spill 只在「只差一点点」时有意义（decode 掉一半）。
- v73 只有 1 个 CDSP domain，所以「一个进程两个 session」是硬件级封死的，多窗口只能多进程（`L-040`⑥）。

## 当服务跑（常驻 + 8k 上下文）

驱动器有 `--serve <port>` 模式：模型只加载一次，A 段变服务器，B 段做 worker，请求之间清 KV 复用。
PC 侧 `serve_http.py` 把设备上的行协议转成 HTTP + SSE。

```bash
# 设备侧（一次加载，常驻）
adb shell "sh /data/local/tmp/serve.sh 18080"     # 内容是 llama-split -c 8192 -k 16 --embd 2560 --batch 256 --serve 18080
adb forward tcp:18080 tcp:18080

# PC 侧
python serve_http.py 7788 18080
# 打开 http://127.0.0.1:7788
```

```bash
curl -N -X POST http://127.0.0.1:7788/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"用三句话解释 KV cache","max_tokens":256,"no_think":true}'
```

请求字段：`prompt`、`system`、`max_tokens`（≤4096）、`temperature`、`top_k`、`top_p`、`seed`、
`chat`（默认 true，套模型自己的 chat 模板）、`no_think`（默认 true）、`stream`（默认 true）。
超限（prompt + max_tokens > 8192）返回 400 和明确原因，不会崩。

### 两条硬约束

**`--batch` 必须远小于 ctx。** 8k 上下文时 B 段含 lm_head，中间张量是 `n_batch × n_vocab × 4`：
`n_batch=8192` 要 8.1 GB，而 v73 的 `max_vmem` 只有 3.14 GB，ctx init 直接失败。取 256 后
8k prompt 分 32 块推给 B，乒乓开销总共几毫秒。

**KV 不随 ctx 线性涨。** qwen35 是混合注意力，只有每 4 层一个 full attention 有 KV；
ctx8192 时每段 `KV 128 MiB / RS 25 MiB`。真正吃内存的是 compute buffer，不是缓存。

### 实测（Qwen3.5-4B Q4_0，K=16/16，ctx8192，batch256）

| 项 | 数值 |
|---|---|
| prefill（2850 tok） | 6.84 s → **417 t/s** |
| prefill（7985 tok） | 24.0 s → **332 t/s** |
| decode（ctx 短） | **10.4~10.8 t/s** |
| 首 token 延迟 | **0.36 s** |
| 4014 prompt + 2403 输出 | 276 s，自然 EOS，无异常 |
| 8014 prompt + 100 输出 | 贴 8192 边界正常 |

### 停止生成

页面上「生成」旁边有红色「停止」按钮。底层是 `POST /cancel` → 设备 `CANCEL` 命令。
停止只能发生在 **token 边界**——那是三段唯一都空闲、事件计数平衡的时刻（见 `L-047`），
所以点下去会等到当前 token 算完，通常 0.1~0.3 s。停止后上下文**不会**保留：下一次请求照常
`cmd=2` 全链清 KV。关掉标签页等价于取消（socket 断开 → 驱动 `recv` 返回 0 → 停止）。

### 界面显示的是设备真实模型

页面标题不再硬编码，加载时用 `INFO` 命令向设备要：model / n_ctx / 段数 / 切点 / hidden / vocab。
浏览器也可以直接看：

```bash
curl -s http://127.0.0.1:7788/info
# {"ok": true, "model": "/data/local/tmp/gguf/qwen35-9b-q4km.gguf", "n_ctx": 8192,
#  "n_stage": 3, "n_embd": 4096, "n_vocab": 248320, "split": "11,22"}
```

### 行协议（调试用）

设备端 `:18080` 上是一行一条命令的文本协议，`serve_http.py` 就是它的客户端：

```
PING                                             -> PONG
TOK <hex> | TOKN <hex>                           -> TOKS <n> <id>...
CHAT <hex-system|-> <hex-user>                   -> CHATT <hex-rendered>
DETOK <n> <id>...                                -> TEXT <hex>
GEN <rid> <max> <temp> <top_k> <top_p> <seed> <nprompt> <id>...  -> P <rid> <tok> <hex> ...
                                                                 -> E <rid> <status> <nout> <ms>
CANCEL                                           -> 无回复；正在生成的请求会在下一个 token 边界停下
                                                     （客户端断开连接同样被当作 CANCEL）
QUIT                                             -> BYE
```

`TOK` 会加 BOS，`TOKN` 不加（已经套过 chat 模板的文本必须用后者）。
服务端**一次只服务一个客户端**，模型只有一个实例；第二个连接会一直等到第一个断开。

### 关思考

模板里 `enable_thinking=false` 会输出一个空的 `<think>\n\n</think>\n\n` 块，
但 `llama_chat_apply_template()` 没有传这个变量的入口。所以 `serve_http.py` 在渲染完 assistant
前缀后自己补上这段字节，流式侧再用状态机把开头的空块吃掉。

## 量化（HF → GGUF）

```bash
cd <llama.cpp>
python convert_hf_to_gguf.py <模型目录> --outfile <model>-f16.gguf --outtype f16
cd <repo>
python quantize_gguf.py <model>-f16.gguf <model>-Q4_0.gguf q4_0
```

需要 `torch` / `transformers` / `safetensors` / `gguf`（python 3.13 可用）。
Q4_0 结果：**4B → 2.45 GB（441 张量，257 Q4_0 + 184 F32）**，见 `L-034`。

## 顺带记住的坑

- 驱动里 `llama_batch_init(n, n_embd, n_seq)` 在 `embd != 0` 时**不分配 token 数组**，写 `b.token[i]` 会 SIGSEGV（`L-033`③）。
- 不加 `-dev HTP0` 时权重会被分到 OpenCL，pp 掉 4 倍（`L-024`）；驱动器已强制只列 HTP。
- `token_embd` / `per_layer_*` 一律钉 CPU（`.*token_embd.*` / `.*per_layer.*` override），否则单张量就超 512 MiB 单次映射上限。
