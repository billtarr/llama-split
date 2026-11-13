#!/usr/bin/env python3
"""plan_split.py -- lay out layer stages for llama-split automatically.

Replaces "guess a --split, start the chain, watch it die on fastrpc_mmap failed, guess again".

It reads the GGUF and uses the *real* per-layer byte counts -- Qwen3.5-9B Q4_K_M ranges from
121 to 134 MiB per layer, so a cut that looks even in layers is not even in bytes -- then picks
cut points so that every stage fits the HTP VA window while the slowest stage is as short as
possible.

    python plan_split.py Qwen3.5-9B-Q4_K_M.gguf
    python plan_split.py model.gguf --va 7400 --ctx 8192 --batch 256   # 8 Gen 3
    python plan_split.py model.gguf --all                               # every K, to choose by hand

The cost model is deliberately small (see docs/60-llama-split/architecture.md 5.1):

    stage_ms  = sum(ms_per_layer of its layers) + boundary_ms (except the first stage)
    stage_mib = weight bytes + KV share + compute buffer

`ms_per_layer` is scaled by the layer's byte count, so a fatter layer costs proportionally more.
The defaults are calibrated on v73 / 9B Q4_K_M (L-053, L-055); override them for another SoC.
"""

import argparse
import collections
import re
import sys

MIB = 1048576.0

CPU_PINNED = ("token_embd.weight", "per_layer")
# output.weight is NOT pinned: it can go either way, and where it goes changes both the budget and
# the cost model (see L-058). It is modelled separately.
OUTPUT_NAME = "output.weight"


def read_layers(path):
    try:
        import gguf
    except ImportError:
        sys.exit("需要 gguf 包：pip install gguf")

    r = gguf.GGUFReader(path)
    per = collections.defaultdict(int)
    other = {}
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.(.+)$", t.name)
        if m:
            per[int(m.group(1))] += t.n_bytes
        else:
            other[t.name] = t.n_bytes

    n_layer = len(per)
    if n_layer == 0:
        sys.exit("这个 gguf 里没有 blk.N.* 张量，不是 transformer 模型？")
    layers = [per[i] / MIB for i in range(n_layer)]

    pinned = {k: v / MIB for k, v in other.items() if any(p in k for p in CPU_PINNED)}
    out_mib = sum(v / MIB for k, v in other.items() if k == OUTPUT_NAME)
    # anything else not under blk.* still lives in the graph: count it against the VA budget
    loose = {k: v / MIB for k, v in other.items() if k not in pinned and k != OUTPUT_NAME}
    return layers, pinned, loose, out_mib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    # 2048 is the published VA window, but a stage that reaches ~2000 MiB dies at the first
    # concurrent decode (`fastrpc_mmap failed`), so the useful budget is lower. Measured on v73:
    # 1879 MiB works with 2 slots, 2012 MiB aborts. 1900 is the safe default; pass --va to override.
    # The stages do NOT share one ceiling, so there are three budgets. Every default is a measured
    # boundary on v73 with two slots (L-058): stage 0 carries the embedding branch and dies first.
    ap.add_argument("--va", type=float, default=1800, help="budget for a middle stage (v73: 1800)")
    ap.add_argument("--va-first", type=float, default=1500,
                    help="stage 0 only: measured 1504 MiB ok, 1618 aborts")
    ap.add_argument("--va-last", type=float, default=1950,
                    help="last stage only: 8 layers + lm_head 1902 MiB works, 9 layers / 2023 aborts")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--stages", type=int, default=0, help="force K stages (0 = pick the smallest that fits)")
    ap.add_argument("--all", action="store_true", help="print every K from 1 to 6")
    ap.add_argument("--ms-per-mib", type=float, default=0.048,
                    help="HTP decode ms per MiB of layer weights (v73 measured 0.048; "
                         "66 ms / 11 layers x 125 MiB). Scales across model sizes.")
    ap.add_argument("--ms-per-layer", type=float, default=0,
                    help="override: fixed ms per layer, ignoring weight size")
    ap.add_argument("--boundary-ms", type=float, default=15.0,
                    help="fixed cost of one extra stage boundary (measured, L-053)")
    ap.add_argument("--kv-mib-per-1k", type=float, default=16.0,
                    help="KV cache MiB per 1k context (9B hybrid attention: 128 MiB @ 8k)")
    ap.add_argument("--cb-mib", type=float, default=0,
                    help="compute buffer MiB per stage (0 = estimate from batch)")
    ap.add_argument("--goal", choices=("latency", "throughput"), default="latency",
                    help="latency = fewest stages that fit; throughput = shortest worst stage")
    ap.add_argument("--lmhead-ms", type=float, default=75.0,
                    help="lm_head + sampling on the CPU, serial with the last stage "
                         "(9B Q6_K 796 MiB: 75 ms measured, L-058)")
    ap.add_argument("--lmhead-htp-ms", type=float, default=21.0,
                    help="the same matmul with output.weight on the DSP (21 ms measured)")
    ap.add_argument("--lmhead", choices=("auto", "cpu", "htp"), default="auto",
                    help="auto = put output.weight on the DSP when the last stage still fits")
    ap.add_argument("--slots", type=int, default=2,
                    help="contexts per stage; KV is reserved per slot (default 2, see L-055)")
    args = ap.parse_args()

    layers, pinned, loose, out_mib = read_layers(args.gguf)
    n = len(layers)
    avg = sum(layers) / n

    # compute buffer scales with batch: measured 13 MiB @64, 52 MiB @256 on v73
    cb = args.cb_mib if args.cb_mib > 0 else max(8.0, 13.0 * args.batch / 64.0)
    kv_total = args.kv_mib_per_1k * args.ctx / 1000.0 * args.slots
    loose_total = sum(loose.values())

    # Moving the lm_head to the DSP makes *every* layer cheaper, not just the last one: measured
    # 6.0~6.6 ms/layer with the CPU matmul in the loop, 4.6~4.7 ms with it moved (L-058). The
    # mechanism is not settled (LLC eviction of DSP weights vs a power/clock effect), so this is an
    # empirical factor, not a derivation.
    CPU_LM_PENALTY = 0.77

    print("model   : %s" % args.gguf)
    print("layers  : %d, weights %.1f MiB (%.2f avg, %.1f ~ %.1f min/max)"
          % (n, sum(layers), avg, min(layers), max(layers)))
    print("pinned  : %s (not counted against VA)"
          % (", ".join("%s %.1f MiB" % (k, v) for k, v in sorted(pinned.items())) or "none"))
    print("lm_head : output.weight %.1f MiB, %s" % (out_mib, args.lmhead))
    print("budget  : VA %0.f/%0.f/%0.f MiB (first/middle/last), ctx %d x%d slots "
          "(KV %.0f MiB total), batch %d (CB %.1f MiB)"
          % (args.va_first, args.va, args.va_last, args.ctx, args.slots, kv_total, args.batch, cb))
    print()

    INF = float("inf")

    def solve(lm_on_htp, K, objective):
        """best cut for K stages with the lm_head either on the DSP or on the CPU"""
        # ms_per_mib was calibrated with the lm_head on the CPU (6.0~6.6 ms per 125 MiB layer);
        # moving it to the DSP makes every layer ~23% cheaper as well (L-058)
        ms = [args.ms_per_mib * (CPU_LM_PENALTY if lm_on_htp else 1.0) * b for b in layers]

        def seg(i, j):
            last = (j == n)
            b = sum(layers[i:j]) + kv_total * (j - i) / n + cb
            if i == 0:
                b += loose_total                           # embedding etc. live in stage 0
            if last and lm_on_htp:
                b += out_mib                               # lm_head shares the last stage's window
            t = sum(ms[i:j]) + (0.0 if last else args.boundary_ms)
            return b, t

        best = [[INF] * (n + 1) for _ in range(K + 1)]
        prev = [[-1] * (n + 1) for _ in range(K + 1)]
        best[0][0] = 0.0
        for k in range(1, K + 1):
            for j in range(1, n + 1):
                for i in range(k - 1, j):
                    if best[k - 1][i] == INF:
                        continue
                    b, t = seg(i, j)
                    cap = args.va_first if i == 0 else (args.va_last if j == n else args.va)
                    if b > cap:
                        continue
                    c = (best[k - 1][i] + t) if objective == "latency" else max(best[k - 1][i], t)
                    if c < best[k][j]:
                        best[k][j] = c
                        prev[k][j] = i
        if best[K][n] == INF:
            return None
        cuts, j, k = [], n, K
        while k > 0:
            i = prev[k][j]
            cuts.append((i, j))
            j, k = i, k - 1
        cuts.reverse()
        lm_ms = args.lmhead_htp_ms if lm_on_htp else args.lmhead_ms
        return cuts, ms, lm_ms

    def best_for(lm_on_htp):
        """(K, cuts, ms, lm_ms, serial) for the K the goal asks for, or None"""
        cand = {}
        for K in range(1, 7):
            r = solve(lm_on_htp, K, "throughput")
            if r:
                cand[K] = r
        if not cand:
            return None
        if args.stages:
            K = args.stages if args.stages in cand else None
        elif args.goal == "latency":
            K = min(cand)                       # fewest stages = fewest boundary taxes
        else:
            def total_ms(k):
                cuts_k, ms_k, lm_k = cand[k]
                return sum(sum(ms_k[i:j]) + (0.0 if j == n else args.boundary_ms)
                           for i, j in cuts_k) + lm_k
            K = min(cand, key=total_ms)
        if K is None:
            return None
        cuts, ms, lm_ms = cand[K]
        stage_ms = [sum(ms[i:j]) + (0.0 if j == n else args.boundary_ms) for i, j in cuts]
        return K, cuts, ms, lm_ms, sum(stage_ms) + lm_ms, cand

    options = {}
    for lm_on_htp, name in ((True, "htp"), (False, "cpu")):
        if lm_on_htp and out_mib <= 0:
            continue        # tied embeddings: the lm_head *is* token_embd, which stays on the CPU
        r = best_for(lm_on_htp)
        if r:
            options[name] = r
    if not options:
        sys.exit("!! 1~6 段全都装不下：权重 %.0f MiB 远超 VA %.0f MiB。"
                 "先降 batch/ctx，或换更小的量化。" % (sum(layers), args.va))

    if args.lmhead == "auto":
        pick = min(options, key=lambda k: options[k][4])
    else:
        if args.lmhead not in options:
            sys.exit("!! lm_head=%s 装不下：末段放不进 %.0f MiB。用 --lmhead auto 让它重新切。"
                     % (args.lmhead, args.va))
        pick = args.lmhead

    K, cuts, ms, lm_ms, serial, cand = options[pick]
    print("%-4s %-9s %8s %6s %6s %8s %6s %8s"
          % ("seg", "layers", "weights", "+KV", "+CB", "total", "VA%", "ms"))
    for idx, (i, j) in enumerate(cuts):
        w = sum(layers[i:j])
        kv = kv_total * (j - i) / n
        extra = out_mib if (j == n and pick == "htp") else 0.0
        tot = w + kv + cb + (loose_total if i == 0 else 0.0) + extra
        t = sum(ms[i:j]) + (0.0 if j == n else args.boundary_ms)
        cap = args.va_first if i == 0 else (args.va_last if j == n else args.va)
        print("  %-4s %-9s %7.0fM %5.0fM %5.0fM %7.0fM %5.0f%% %8.1f%s"
              % (chr(65 + idx), "%d-%d" % (i, j - 1), w, kv, cb, tot, 100.0 * tot / cap, t,
                 "  (含 lm_head %.0f MiB)" % out_mib if extra else ""))
    print("  K = %d, 单请求 %.0f ms/token -> %.2f t/s  (各段串行 + lm_head %.0f ms on %s)"
          % (K, serial, 1000.0 / serial, lm_ms, "DSP" if pick == "htp" else "CPU"))
    print("  双请求: 约 %.2f t/s (实测 1.1~1.4x；单 DSP 满负荷，加请求主要提吞吐)"
          % (1.15 * 1000.0 / serial))
    split = ",".join(str(cuts[k][1]) for k in range(len(cuts) - 1))
    print("  --split %s%s" % (split or "(none)", "   # 单段，无需切分" if not split else ""))
    if pick == "htp":
        print("  --route '.*output\\.weight=htp'        # 末段要留出 %.0f MiB" % out_mib)
    if K > 1:
        print("  建议: --batch %d --slots %d%s" % (args.batch, args.slots,
              "   (双槽必须配小 batch, 见 L-055)" if args.slots > 1 else ""))

    other = "cpu" if pick == "htp" else "htp"
    if other in options:
        print("  另一种放法 (%s): K=%d, %.0f ms/token -> %.2f t/s"
              % (other, options[other][0], options[other][4], 1000.0 / options[other][4]))
    print()

    if args.all:
        print("所有可行段数（lm_head on %s）：" % ("DSP" if pick == "htp" else "CPU"))
        for k in sorted(cand):
            cuts_k, ms_k, lm_k = cand[k]
            s = sum(sum(ms_k[i:j]) + (0.0 if j == n else args.boundary_ms) for i, j in cuts_k) + lm_k
            print("  K=%d  --split %s   %.0f ms -> %.2f t/s"
                  % (k, ",".join(str(cuts_k[x][1]) for x in range(len(cuts_k) - 1)) or "(none)",
                     s, 1000.0 / s))


if __name__ == "__main__":
    main()
