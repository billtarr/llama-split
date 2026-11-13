#!/usr/bin/env python3
"""Smoke test for the device-side line protocol (through adb forward)."""
import socket
import sys
import time

HOST, PORT = "127.0.0.1", 18080


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "用一句话说明什么是 KV cache。"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    s = socket.create_connection((HOST, PORT), timeout=30)
    s.settimeout(600)
    f = s.makefile("rwb")

    def cmd(line):
        f.write((line + "\n").encode())
        f.flush()
        return f.readline().decode().rstrip("\r\n")

    print("PING  ->", cmd("PING"))

    t0 = time.time()
    r = cmd("TOK " + prompt.encode().hex())
    ids = [int(x) for x in r.split()[2:]]
    print("TOK   -> %d tokens in %.0f ms" % (len(ids), (time.time() - t0) * 1000))
    print("        ids =", ids[:12], "..." if len(ids) > 12 else "")

    r = cmd("DETOK %d %s" % (len(ids), " ".join(map(str, ids))))
    text = bytes.fromhex(r.split(" ", 1)[1]).decode()
    print("DETOK ->", repr(text))
    assert text == prompt, "roundtrip mismatch"

    print("GEN   -> %d max_tokens" % max_tokens)
    t0 = time.time()
    f.write(("GEN 1 %d 0.7 40 0.95 1234 %d %s\n" % (max_tokens, len(ids), " ".join(map(str, ids)))).encode())
    f.flush()

    out = []
    n = 0
    first = None
    status = None
    while True:
        line = f.readline().decode().rstrip("\r\n")
        if not line:
            break
        parts = line.split(" ", 4)
        if parts[0] == "P" and len(parts) >= 3 and parts[2] != "-1":
            n += 1
            if first is None:
                first = time.time() - t0
            piece = bytes.fromhex(parts[3]).decode("utf-8", "replace") if len(parts) > 3 and parts[3].strip() else ""
            out.append(piece)
        elif parts[0] == "E":
            status = line
            break

    dt = time.time() - t0
    print("        first token %.0f ms, %d tokens in %.1f s (%.2f tok/s)"
          % ((first or 0) * 1000, n, dt, n / dt if dt else 0))
    print("        END:", status)
    print()
    print("".join(out))
    s.close()


if __name__ == "__main__":
    main()
