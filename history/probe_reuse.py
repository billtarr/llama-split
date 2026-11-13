#!/usr/bin/env python3
"""Measure per-request reuse cost, and how much of it is clearing a full 8k KV."""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:7788"

PARA = ("在移动端部署大语言模型时，最先撞到的是内存带宽和访存容量这两道墙。"
        "模型权重通常有数 GB，而 NPU 侧的 IOVA 窗口只有两 GB 出头，一旦权重加上 KV 缓存超出这个窗口，"
        "映射就会直接失败。按层切分把网络拆成前后两段，放进两个独立进程，各自拿到一份地址窗口，"
        "从而让原本装不下的模型能够加载。代价是每生成一个 token 都要在两段之间传递一次隐藏状态。")


def gen(prompt, max_tokens=1, **kw):
    body = {"prompt": prompt, "max_tokens": max_tokens, "no_think": True, "stream": False}
    body.update(kw)
    req = urllib.request.Request(BASE + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=600).read().decode())
    return r, time.time() - t0


def main():
    short = "1+1 等于几？只回答数字。"

    print("--- 短请求 x4（KV 近空）")
    warm = None
    for i in range(4):
        r, dt = gen(short, 1)
        if warm is None:
            warm = dt
        print("  #%d: %d prompt + %d out | %6.0f ms" % (i, r["n_prompt"], r["n_out"], dt * 1000))

    print("--- 填满 KV（长 prompt）")
    r, dt = gen((PARA * 200)[:12000], 1)
    print("  %d prompt | %6.0f ms" % (r["n_prompt"], dt * 1000))

    print("--- 紧跟的短请求 x3（需要清掉 8k KV）")
    for i in range(3):
        r, dt = gen(short, 1)
        print("  #%d: %d prompt + %d out | %6.0f ms" % (i, r["n_prompt"], r["n_out"], dt * 1000))


if __name__ == "__main__":
    main()
