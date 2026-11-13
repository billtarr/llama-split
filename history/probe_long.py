#!/usr/bin/env python3
"""Long-context probe: prefill a large prompt and measure the rate."""
import socket
import sys
import time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18080

PARA = ("在移动端部署大语言模型时，最先撞到的是内存带宽和访存容量这两道墙。"
        "模型权重通常有数 GB，而 NPU 侧的 IOVA 窗口只有两 GB 出头，一旦权重加上 KV 缓存超出这个窗口，"
        "映射就会直接失败。按层切分把网络拆成前后两段，放进两个独立进程，各自拿到一份地址窗口，"
        "从而让原本装不下的模型能够加载。代价是每生成一个 token 都要在两段之间传递一次隐藏状态，"
        "并且两段共享同一个 DSP 核，吞吐的提升主要来自 prefill 阶段，解码阶段基本只是打平。")


def cmd(f, line):
    f.write((line + "\n").encode())
    f.flush()
    return f.readline().decode().rstrip()


def main():
    s = socket.create_connection(("127.0.0.1", PORT), timeout=30)
    s.settimeout(1200)
    f = s.makefile("rwb")

    targets = [int(x) for x in (sys.argv[2:] or ["1500", "3000"])]
    for target in targets:
        text = (PARA * 60)[:target]
        ids = [int(x) for x in cmd(f, "TOK " + text.encode().hex()).split()[2:]]
        n = len(ids)
        t0 = time.time()
        f.write(("GEN 7 1 0.0 1 1.0 1 %d %s\n" % (n, " ".join(map(str, ids)))).encode())
        f.flush()
        status = None
        while True:
            line = f.readline().decode().rstrip()
            if line.startswith("E "):
                status = line
                break
        dt = time.time() - t0
        print("prompt %5d chars -> %5d tokens | %6.2f s | %6.1f tok/s | %s"
              % (len(text), n, dt, n / dt, status), flush=True)
    s.close()


if __name__ == "__main__":
    main()
