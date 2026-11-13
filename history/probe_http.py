#!/usr/bin/env python3
"""Long-context / long-output probes over the HTTP frontend."""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:7788"

PARA = ("在移动端部署大语言模型时，最先撞到的是内存带宽和访存容量这两道墙。"
        "模型权重通常有数 GB，而 NPU 侧的 IOVA 窗口只有两 GB 出头，一旦权重加上 KV 缓存超出这个窗口，"
        "映射就会直接失败。按层切分把网络拆成前后两段，放进两个独立进程，各自拿到一份地址窗口，"
        "从而让原本装不下的模型能够加载。代价是每生成一个 token 都要在两段之间传递一次隐藏状态，"
        "并且两段共享同一个 DSP 核，吞吐的提升主要来自 prefill 阶段，解码阶段基本只是打平。")


def call(prompt, max_tokens, stream=False, **kw):
    body = {"prompt": prompt, "max_tokens": max_tokens, "stream": stream}
    body.update(kw)
    req = urllib.request.Request(BASE + "/generate",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            raw = r.read().decode()
    except urllib.error.HTTPError as exc:
        return {"error": exc.read().decode(), "http": exc.code, "wall": time.time() - t0}
    dt = time.time() - t0
    if stream:
        return {"raw": raw, "wall": dt}
    out = json.loads(raw)
    out["wall"] = dt
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "prefill"

    if mode == "prefill":
        sizes = [int(x) for x in sys.argv[2:]] or [1500, 3000, 5000]
        for chars in sizes:
            p = (PARA * 200)[:chars]
            r = call(p, 1)
            if "error" in r:
                print("prompt %5d chars -> ERROR %s" % (chars, r["error"]))
                continue
            print("prompt %5d chars -> %5d tok | %6.2f s | %6.1f tok/s"
                  % (chars, r["n_prompt"], r["wall"], r["n_prompt"] / r["wall"]), flush=True)

    elif mode == "decode":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 800
        chars = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        if chars:
            p = (PARA * 200)[:chars] + "\n\n请详细讲解移动端 NPU 推理的瓶颈，并给出优化思路。"
        else:
            p = "请详细讲解移动端 NPU 推理的瓶颈，并给出优化思路。"
        r = call(p, n, stream=True)
        # parse SSE
        toks = 0
        first = None
        text = []
        t0 = time.time()
        for line in r["raw"].splitlines():
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                continue
            ev = json.loads(d)
            if "token" in ev:
                toks += 1
                if first is None:
                    first = time.time() - t0
                text.append(ev["token"])
            elif ev.get("done"):
                print("done event:", ev)
        print("decode: %d tok in %.2f s | first token %.2f s | %.2f tok/s"
              % (toks, r["wall"], first or 0, (toks - 1) / max(r["wall"] - (first or 0), 1e-9)))
        print("---")
        print("".join(text)[:600])

    elif mode == "repeat":
        # several requests back to back: checks that KV is reset and nothing leaks
        for i in range(4):
            r = call("回答一个字：%d 加 1 等于几？只说数字。" % i, 24)
            print("req %d: %s | %.2f s | %d prompt + %d out"
                  % (i, r.get("text", r.get("error", ""))[:40].replace("\n", " "),
                     r["wall"], r.get("n_prompt", -1), r.get("n_out", -1)), flush=True)


if __name__ == "__main__":
    main()
