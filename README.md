# llama-split

把一个 llama.cpp 模型按**层**切成多段，每段一个独立进程，段间用共享内存 + eventfd 串成流水。

存在的唯一理由：8 Gen 2 的 HTP 地址窗口只有 **2048 MiB**，装不下 5.2 GiB 的 9B 模型。
分进程是为了让每段的权重各自映射，不是并行提速（单 CDSP 算力不叠加）。

设备：SM8550 / 8 Gen 2，QNN HTP 后端，Android aarch64。

## 现状（Qwen3.5-9B Q4_K_M，32 层）

| 项 | 数值 |
|---|---|
| 切分 | `--split 11,24` → 11 / 13 / 8 层 |
| 单请求 decode | **5.05 t/s**（198 ms/token） |
| 双并发（`--slots 2`） | **5.77 t/s** |
| 数值 | 与 CPU 单进程逐 token 一致 |
| 上下文 | 8192（混合注意力，仅 1/4 层有 KV） |

## 目录

```
llama-split.cpp     驱动：切分、进程编排、共享内存协议、HTTP 服务端、以及全部三段 worker
stage_abi.h         外部 stage 的 ABI 契约（版本化；一个 stage 可以不链接 llama.cpp）
stage_pass.c        参考 stage 实现，同时是测试件（恒等 / 强扰动两种模式）
build.sh            交叉编译驱动 -> ./llama-split
build_stage.sh      编译 stage 库 -> *.so

plan_split.py       规划器：读 GGUF 真实逐层字节，DP 求切点 + lm_head 落点，输出 --split/--route
quantize_gguf.py    一次性量化器（本机没有认 qwen35 的 x86 llama-quantize 时用，纯 numpy）

serve_http.py       PC 侧 HTTP 前端（SSE 流式、/cancel、每请求一条独立设备连接）
serve9b.sh          设备侧启动参数（9B 三段 + 双槽 + lm_head 在 DSP）
sup9b.sh            设备侧守护：任一段挂了 3 s 后整链重启
start_server.sh     PC 侧一键：起设备链 + adb forward + HTTP 前端

test/v2test.py      单请求冒烟测试（打印 token 序列与耗时）
test/verify_v2.py   并发一致性：单请求 vs 双并发，必须逐 token 相同
test/bench_v2.py    吞吐基准：单请求 / 双并发 / 加速比

lib/                运行时依赖：12 个 .so（88 MB），含 5 个 SoC 的 DSP applet。见 lib/README.md

docs/               设计文档（架构 / 操作 / 冷启动成本 / 抽出的台账）。**权威版本在工作区**
                    `docs/60-llama-split/`，这里是副本，用 sync_docs.py 同步，别直接改
sync_docs.py        从工作区同步 docs/ 并重新抽取台账（幂等，源文档漂移时会报错）

history/            11-09 一次性实验脚本（4B 时代的分步排查），**不是现役**，留作证据
```

## 怎么跑

驱动和设备脚本都要在设备上，模型也要：

```sh
ADB=${ADB:-adb}          # 指定你的 adb 路径，或让它在 PATH 里
export MSYS_NO_PATHCONV=1

"$ADB" push llama-split /data/local/tmp/llama.cpp/bin/
"$ADB" push lib/*.so /data/local/tmp/llama.cpp/lib/       # 12 个，一次推完
"$ADB" push serve9b.sh sup9b.sh /data/local/tmp/
"$ADB" push stage_pass.so /data/local/tmp/llama.cpp/     # 只有用 --ext-after 才需要
```

`lib/` 里是完整的运行时库（含 `libggml-hexagon.so` 和 5 个 SoC 的 applet），**换机器不需要重编 llama.cpp**。
`libc.so` / `libc++_shared.so` / `libOpenCL.so` / `libcdsprpc.so` 由 ROM 提供，不要拷；细节见
[`lib/README.md`](lib/README.md)。

`start_server.sh` **假设上面这些已经在设备上**，它只负责起链、转发、起前端：

```sh
bash start_server.sh          # 全程前台，Ctrl-C 只停 HTTP 前端
bash start_server.sh 9000     # 换 HTTP 端口
```

- HTTP: `http://127.0.0.1:7788`（`/generate` 流式、`/cancel`、`/info`、`/health`）
- 设备端口 18080；停设备侧：`adb shell "touch /data/local/tmp/sup.stop; pkill -f llama-split"`

直接跑一次生成（不起服务）：

```sh
"$ADB" shell "cd /data/local/tmp/llama.cpp && \
  LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib ./bin/llama-split \
  -m /data/local/tmp/gguf/qwen35-9b-q4km.gguf -p 'The capital of France is' \
  -c 1024 -n 16 -t 4 --split 11,24 --embd 4096 --batch 64"
```

## 换模型

先让规划器算，不要手凑：

```sh
python plan_split.py model.gguf --ctx 8192 --batch 64
```

它输出 `--split` 和 `--route`，以及预估 t/s。9B 上预测 5.02 vs 实测 5.05（误差 0.6%）。
小模型会直接告诉你「K=1，不用切」。

## 四条硬约束（违反即崩或静默变慢）

1. **`--slots 2` 必须配 `--batch 64`**。batch 256 时第二个 context 起不来（每 ctx 52 MiB compute buffer ×2 装不下）。
2. **三个位置的 VA 天花板不一样**：首段 1496 MiB 过 / 1618 崩（它还带 embedding 分支）、中间段 1713 过、末段 1902 过 / 2023 崩。
   **崩在首次 decode 而不是 init**，所以「起来全 ready」不等于装得下，必须跑一次真实生成。
3. **lm_head 放 DSP 时末段只能 8 层**（`--route '.*output\.weight=htp'`）。放 CPU 会慢 13%，而且会让每一层都慢 23%。
4. **`history/` 里的脚本不要在现役链上用**。它们的启动参数（4B / 两段 / ctx 64）和现在的协议不兼容。

## 写自己的 stage

```sh
bash build_stage.sh my_stage.c
# 设备上：./bin/llama-split ... --split 11,24 --ext-after 0:/data/local/tmp/llama.cpp/my_stage.so
```

契约见 `stage_abi.h`：`open` / `run` / `close` 三个函数，加上带 `dtype` 的 `split_hidden` 和
带 `n_tok/pos0/sample/rid/chunk_seq/ids` 的 `split_slot_meta`。外部 stage **不能是最后一段**
（只有持词表的段能采样）。

**接之前先跑 `stage_pass` 自检**：`STAGE_PASS_BIAS=0` 必须与不插 ext 时逐 token 相同，
`=5000` 必须输出变样。两条都过才说明接缝是通的。

## 更细的东西

都在 `docs/`：

| 想知道 | 读 |
|---|---|
| 它现在是什么形状、通用化还缺什么 | [`docs/architecture.md`](docs/architecture.md) ★ 主线 |
| 怎么编、怎么跑、怎么验数值、切点怎么选 | [`docs/runbook.md`](docs/runbook.md) |
| 冷启动 3 s 花在哪、怎么摊掉 | [`docs/create-overhead.md`](docs/create-overhead.md) |
| 一句话结论（40 条 L + E-016，含被否决的方案） | [`docs/ledger.md`](docs/ledger.md) |
| 索引与入口 | [`docs/README.md`](docs/README.md) |

`docs/` 是工作区 `docs/60-llama-split/` 的副本（那边还有 PMS / Qwen-Image / gemma4 等块）。
**不要在 `docs/` 里直接改**，改完工作区跑 `python sync_docs.py` 同步回来。
