"""HTTP smoke test: one POST /generate round trip through serve_http.py.

Deliberately separate from v2test.py / verify_v2.py: those speak the raw device protocol
(`GEN ...` on port 18081) and have no notion of max_tokens. This one goes through the PC-side
front end, so it exercises the same path the browser and any HTTP client use.

    python test/smoke_http.py              # port 7788, 256-token budget
    python test/smoke_http.py 7788 64      # explicit port and budget

max_tokens is passed explicitly on purpose. The server-side default is meant for interactive
use (1024); a smoke test should stay cheap, and at ~5 t/s 256 tokens is already ~50 s while
1024 would be ~3.5 min.
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 7788
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 256
PROMPT = "用三句话解释 KV cache"

# A system-wide http_proxy (Clash and friends) makes urllib send 127.0.0.1:7788 to the proxy,
# which answers 502 and looks exactly like a dead server. Loopback must never be proxied.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def main():
    body = json.dumps({
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "no_think": True,
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d/generate" % PORT,
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    try:
        with OPENER.open(req, timeout=900) as r:
            out = json.load(r)
    except Exception as exc:                              # noqa: BLE001
        print("FAIL: request error: %s" % exc)
        return 1
    wall = time.time() - t0

    n_out = out.get("n_out", 0)
    seconds = out.get("seconds", 0) or 0
    rate = n_out / seconds if seconds else 0.0

    print("prompt tokens : %d" % out.get("n_prompt", 0))
    print("output tokens : %d  (budget %d)" % (n_out, MAX_TOKENS))
    print("server time   : %.2fs  (%.2f t/s)" % (seconds, rate))
    print("wall time     : %.2fs" % wall)
    print("text          : %s" % out.get("text", "")[:200].replace("\n", " "))

    ok = 0 < n_out <= MAX_TOKENS
    print("VERDICT:", "PASS" if ok else "FAIL (n_out=%d, budget=%d)" % (n_out, MAX_TOKENS))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
