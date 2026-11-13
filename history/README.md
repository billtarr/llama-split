# history/ — 一次性实验脚本，不参与现役链

这些都是 2026-11-09 到 11-10 排查阶段写的**分步验证脚本**，目标是搞清楚「层切分到底成不成立」，
不是拿来跑的。它们硬编码了当时的配置：**Qwen3.5-4B / 两段 / `-k 16` / ctx 64~1024**，协议也和现在的
v2 流水不兼容。留在仓库里是因为台账把它们当证据指针（`L-036`、`L-037`、`L-043`、`L-044`、`L-047`、`L-048`）。

| 文件 | 当时在验证什么 |
|---|---|
| `runsingle.sh` | 单进程基线，重复 N 次取稳定值 |
| `runsplit.sh` / `runsplit2.sh` / `runsplit3.sh` | 切分跑通、参数化模型/embd/k、加额外参数 |
| `dumpcheck.sh` | dump A 段导出的 hidden 再 md5，查切分是否确定性 |
| `injectcheck.sh` | 把 A 的 hidden 注回单进程，隔离「切分错」还是「B 段错」 |
| `coresidency.sh` | 第二个常驻 DSP 进程会不会让单进程结果变非确定 |
| `longctx.sh` | ctx 1024 / 220 token 下切分是否仍成立 |
| `bench4b.sh` | 4B 的 pp/tg 对照（单进程 vs 切分，ctx 64/256） |
| `t460.sh` | 每 token 打点取中位（差分法会被 model load 波动污染） |
| `exp2.sh` | 11-10 候选切法遍历 + lm_head 落点隔离实验 |
| `serve.sh` | 最早的 4B 两进程服务（被 `serve9b.sh` 取代） |
| `probe_http.py` / `probe_long.py` / `probe_reuse.py` / `test_proto.py` | 前端连通性、长上下文、KV 复用、协议握手探针 |
| `_main.inl` / `_stage_worker.inl` | 重构前的代码片段备份，`main` 与 `stage_worker` 已并入 `llama-split.cpp` |

现役的验证脚本在 `../test/`。
