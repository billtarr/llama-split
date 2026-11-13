#!/usr/bin/env python3
"""HTTP frontend for the two-process layer-split llama server running on the phone.

    设备侧                                   本机
    llama-split --serve 18080   <-- adb forward tcp:18080 --  serve_http.py :7788

The device speaks a tiny line protocol; this script turns it into HTTP + SSE:

    GET  /health                 -> {"ok": true}
    GET  /                       -> a small test page
    POST /generate               -> SSE stream  (default) or JSON when "stream": false
         body: {"prompt": str, "max_tokens": int, "temperature": float,
                "top_k": int, "top_p": float, "seed": int, "stream": bool}
"""
import json
import re
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEVICE_HOST = "127.0.0.1"
DEVICE_PORT = 18080
LISTEN_PORT = 7788
CTX = 8192            # device side -c
MAX_OUT = 4096        # hard cap on max_tokens


class SplitError(RuntimeError):
    pass


IDLE_TIMEOUT = 120


class SplitClient:
    """One long-lived TCP connection; the device accepts a single client at a time."""

    def __init__(self, host=None, port=None):
        # NOTE: do not default these to DEVICE_HOST / DEVICE_PORT directly -- default arguments
        # are bound when the class body executes, so a port given on the command line (which
        # main() assigns later) would be ignored and the client would silently dial the old one.
        self.host = host if host else DEVICE_HOST
        self.port = port if port else DEVICE_PORT
        self.sock = None
        self.fh = None
        self.lock = threading.Lock()
        self.rid = 0
        self.cancelled = False
        self.truncated = False

    def _ensure(self):
        if self.sock is not None:
            return
        s = socket.create_connection((self.host, self.port), timeout=30)
        # a token takes ~0.3 s and one prefill chunk a few seconds; 120 s of silence means the
        # device chain is wedged (or restarted), and we want that as an error, not a hang
        s.settimeout(IDLE_TIMEOUT)
        self.sock = s
        self.fh = s.makefile("rwb")

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.fh = None

    def _cmd(self, line):
        self._ensure()
        try:
            self.fh.write((line + "\n").encode())
            self.fh.flush()
        except OSError:
            self.close()
            raise SplitError("device connection lost")

    def _readline(self, what="reply"):
        try:
            raw = self.fh.readline()
        except socket.timeout:
            self.close()
            raise SplitError("device silent for %d s while waiting for %s" % (IDLE_TIMEOUT, what))
        except OSError:
            self.close()
            raise SplitError("device connection lost")
        if not raw:
            self.close()
            raise SplitError("device closed the connection")
        return raw.decode("utf-8", "replace").rstrip("\r\n")

    def ping(self):
        with self.lock:
            self._cmd("PING")
            return self._readline()

    def info(self):
        """What the device actually loaded (model / ctx / stage split)."""
        with self.lock:
            self._cmd("INFO")
            parts = self._readline("model info").split()
        if not parts or parts[0] != "INFO":
            raise SplitError("bad info reply: %r" % (parts[:3],))
        return {
            "ok": True,
            "model": bytes.fromhex(parts[1]).decode("utf-8", "replace") if len(parts) > 1 else "",
            "n_ctx": int(parts[2]) if len(parts) > 2 else 0,
            "n_stage": int(parts[3]) if len(parts) > 3 else 0,
            "n_embd": int(parts[4]) if len(parts) > 4 else 0,
            "n_vocab": int(parts[5]) if len(parts) > 5 else 0,
            "split": parts[6] if len(parts) > 6 else "?",
        }

    def tokenize(self, text, add_special=False):
        with self.lock:
            self._cmd(("TOK " if add_special else "TOKN ") + text.encode("utf-8").hex())
            parts = self._readline().split()
        if not parts or parts[0] != "TOKS":
            raise SplitError("bad tokenize reply: %r" % (parts[:3],))
        n = int(parts[1])
        return [int(x) for x in parts[2:2 + n]]

    def chat(self, user, system=None):
        """Render the model's chat template on the device."""
        with self.lock:
            sh = "-" if not system else system.encode("utf-8").hex()
            self._cmd("CHAT %s %s" % (sh, user.encode("utf-8").hex()))
            parts = self._readline().split(" ", 1)
        if parts[0] != "CHATT":
            raise SplitError("bad chat reply")
        return bytes.fromhex(parts[1] if len(parts) > 1 else "").decode("utf-8", "replace")

    def detokenize(self, ids):
        with self.lock:
            self._cmd("DETOK %d %s" % (len(ids), " ".join(str(t) for t in ids)))
            parts = self._readline().split(" ", 1)
        if parts[0] != "TEXT":
            raise SplitError("bad detokenize reply")
        return bytes.fromhex(parts[1] if len(parts) > 1 else "").decode("utf-8", "replace")

    def cancel(self):
        """Ask the device to stop the running request.

        Deliberately does NOT take self.lock: generate() holds it for the whole request, and the
        whole point is to interrupt that. A cancel that lands while nothing is running is consumed
        silently by the device and does not disturb the reply stream.
        """
        s = self.sock
        if s is None:
            return False
        try:
            s.sendall(b"CANCEL\n")
            return True
        except OSError:
            return False

    def generate(self, ids, max_tokens, temperature, top_k, top_p, seed):
        """Yields (token_id, text_piece). Raises SplitError on failure."""
        self.cancelled = False
        self.truncated = False
        with self.lock:
            self.rid += 1
            rid = self.rid
            cmd = "GEN %d %d %.4f %d %.4f %d %d %s" % (
                rid, max_tokens, temperature, top_k, top_p, seed & 0x7FFFFFFF,
                len(ids), " ".join(str(t) for t in ids))
            self._cmd(cmd)
            while True:
                line = self._readline()
                parts = line.split(" ", 4)
                if not parts:
                    continue
                kind = parts[0]
                if kind == "P":
                    if len(parts) < 3:
                        continue
                    tok = int(parts[2])
                    hexed = parts[3] if len(parts) > 3 else ""
                    piece = bytes.fromhex(hexed).decode("utf-8", "replace") if hexed else ""
                    if tok >= 0:
                        yield tok, piece
                elif kind == "E":
                    status = parts[2] if len(parts) > 2 else "error"
                    if status == "cancelled":
                        self.cancelled = True
                        return
                    if status == "length":
                        # Ran out of max_tokens. That is how a request is supposed to end when
                        # the answer is longer than the budget -- not a failure. The device uses
                        # "error" / "skew" for the real ones.
                        self.truncated = True
                        return
                    if status != "ok":
                        raise SplitError("generation failed: " + status)
                    return
                elif line in ("OK", "CANCEL"):
                    continue        # a cancel that raced the end of the request
                else:
                    raise SplitError("unexpected reply: %r" % line[:80])


class ThinkSkip:
    """Swallow a leading '<think>...</think>' block while streaming.

    With enable_thinking=false the model emits an empty think block first; showing the raw
    tag would be noise, and it cannot be removed afterwards in a stream.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self):
        self.buf = ""
        self.done = False

    def feed(self, s):
        if self.done:
            return s
        self.buf += s
        if self.OPEN.startswith(self.buf[:len(self.OPEN)]) and len(self.buf) < len(self.OPEN):
            return ""
        if self.buf.startswith(self.OPEN):
            i = self.buf.find(self.CLOSE)
            if i < 0:
                return ""
            rest = self.buf[i + len(self.CLOSE):]
            self.done = True
            self.buf = ""
            return rest
        self.done = True
        out, self.buf = self.buf, ""
        return out

    def flush(self):
        out, self.buf = self.buf, ""
        self.done = True
        return out


# Long-lived connection for the light commands (INFO / PING / health). Generation does NOT use
# it: one device connection per HTTP request, otherwise a shared socket (and its lock) would
# serialise exactly the requests the device is now able to overlap.
CLIENT = SplitClient()

ACTIVE = set()
ACTIVE_LOCK = threading.Lock()


def cancel_all():
    """Ask the device to stop every request currently generating."""
    with ACTIVE_LOCK:
        conns = list(ACTIVE)
    if not conns:
        return CLIENT.cancel()
    ok = False
    for c in conns:
        ok = c.cancel() or ok
    return ok

TEST_PAGE = """<!doctype html>
<meta charset="utf-8"><title>layer-split 服务</title>
<style>
 body{font:15px/1.6 -apple-system,"Segoe UI",sans-serif;max-width:780px;margin:36px auto;padding:0 18px;color:#222;background:#fafafa}
 h1{font-size:20px;font-weight:600}
 textarea{width:100%;height:110px;padding:10px;font:14px/1.5 ui-monospace,monospace;border:1px solid #d0d0d0;border-radius:6px;background:#fff}
 .row{display:flex;gap:12px;margin:12px 0;flex-wrap:wrap;align-items:center}
 input[type=number]{width:76px;padding:5px 7px;border:1px solid #d0d0d0;border-radius:5px}
 button{padding:8px 20px;border:0;border-radius:6px;background:#2563eb;color:#fff;font-size:14px;cursor:pointer}
 button:disabled{background:#9aa8c7;cursor:default}
 #out{white-space:pre-wrap;background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:14px;min-height:150px;font:14px/1.65 ui-monospace,monospace}
 .meta{color:#777;font-size:12px;margin-top:8px}
</style>
<h1 id="h1">layer-split · 连接中…</h1>
<div class="meta" id="modelinfo"></div>
<textarea id="s" placeholder="system（可选，留空则不发送）" style="height:46px;margin-bottom:8px"></textarea>
<textarea id="p" placeholder="输入 prompt…">用三句话解释为什么 KV cache 能加速自回归解码。</textarea>
<div class="row">
  <label>max_tokens <input id="n" type="number" value="1024" min="1" max="4096"></label>
  <label>temp <input id="t" type="number" value="0.7" step="0.1" min="0" max="2"></label>
  <label>top_k <input id="k" type="number" value="40" min="0"></label>
  <label>top_p <input id="pp" type="number" value="0.95" step="0.05" min="0" max="1"></label>
  <label><input id="nt" type="checkbox" checked> 关闭思考</label>
  <button id="go">生成</button>
  <button id="stop" disabled style="background:#b91c1c">停止</button>
</div>
<div id="out"></div>
<div class="meta" id="meta"></div>
<script>
const $ = id => document.getElementById(id);
let busy = false;

async function loadInfo() {
  try {
    const r = await fetch('/info');
    const j = await r.json();
    if (!j.ok) throw new Error(j.error || 'device unreachable');
    const name = j.model.split('/').pop();
    $('h1').textContent = 'layer-split · ' + name + ' · ' + j.n_ctx + ' ctx';
    $('modelinfo').textContent =
      j.n_stage + ' 段 (切点 ' + j.split + ') · hidden ' + j.n_embd +
      ' · vocab ' + j.n_vocab + ' · ' + j.model;
  } catch (e) {
    $('h1').textContent = 'layer-split · 设备未连接';
    $('modelinfo').textContent = String(e) + ' — 服务可能正在重启，稍后刷新';
  }
}
loadInfo();
$('stop').onclick = async () => {
  $('stop').disabled = true;
  $('meta').textContent = '正在停止…';
  try { await fetch('/cancel', {method: 'POST'}); } catch (e) {}
};

$('go').onclick = async () => {
  if (busy) return;
  busy = true; $('go').disabled = true; $('stop').disabled = false;
  $('out').textContent = ''; $('meta').textContent = '';
  const t0 = performance.now(); let n = 0;
  try {
    const r = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: $('p').value,
        system: $('s').value || null,
        no_think: $('nt').checked,
        max_tokens: +$('n').value,
        temperature: +$('t').value,
        top_k: +$('k').value,
        top_p: +$('pp').value,
        stream: true,
      }),
    });
    if (!r.ok) { $('out').textContent = 'HTTP ' + r.status + ': ' + await r.text(); busy=false; $('go').disabled=false; $('stop').disabled=true; return; }
    const reader = r.body.getReader();
    const dec = new TextDecoder(); let buf = '';
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let i;
      while ((i = buf.indexOf('\\n\\n')) >= 0) {
        const raw = buf.slice(0, i).trim(); buf = buf.slice(i + 2);
        if (!raw.startsWith('data:')) continue;
        const data = raw.slice(5).trim();
        if (data === '[DONE]') continue;
        const ev = JSON.parse(data);
        if (ev.token) {
          n++;
          $('out').textContent += ev.token;
        }
        if (ev.done) {
          const secs = (performance.now() - t0) / 1000;
          const how = ev.cancelled ? '已停止' : '完成';
          $('meta').textContent = `${how} · ${n} tokens · ${secs.toFixed(1)}s · ${(n/secs).toFixed(2)} tok/s · prompt ${ev.n_prompt} tok`;
        }
      }
    }
  } catch (e) {
    $('out').textContent += '\\n[error] ' + e;
  }
  busy = false; $('go').disabled = false; $('stop').disabled = true;
};
</script>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, TEST_PAGE, "text/html")
        if self.path == "/cancel":
            return self._send(200, json.dumps({"ok": cancel_all()}))
        if self.path == "/info":
            try:
                return self._send(200, json.dumps(CLIENT.info(), ensure_ascii=False))
            except Exception as exc:                      # noqa: BLE001
                CLIENT.close()
                try:
                    return self._send(200, json.dumps(CLIENT.info(), ensure_ascii=False))
                except Exception:                         # noqa: BLE001
                    return self._send(503, json.dumps({"ok": False, "error": str(exc)}))
        if self.path == "/health":
            # a cached socket dies whenever the adb server restarts, so retry once with a fresh one
            err = None
            for _ in range(2):
                try:
                    return self._send(200, json.dumps({"ok": CLIENT.ping() == "PONG"}))
                except Exception as exc:                  # noqa: BLE001
                    err = exc
                    CLIENT.close()
            return self._send(503, json.dumps({"ok": False, "error": str(err)}))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path == "/cancel":
            return self._send(200, json.dumps({"ok": cancel_all()}))
        if self.path not in ("/generate", "/v1/completions"):
            return self._send(404, json.dumps({"error": "not found"}))

        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:                          # noqa: BLE001
            return self._send(400, json.dumps({"error": "bad json: %s" % exc}))

        # one device connection per request: with --slots N the device runs several requests
        # concurrently, so sharing a socket would put the serialisation right back
        dev = SplitClient()
        with ACTIVE_LOCK:
            ACTIVE.add(dev)
        try:
            self._generate(dev, req)
        finally:
            with ACTIVE_LOCK:
                ACTIVE.discard(dev)
            dev.close()

    def _generate(self, dev, req):
        prompt = req.get("prompt", "")
        system = req.get("system") or None
        chat_mode = bool(req.get("chat", True))
        no_think = bool(req.get("no_think", True))
        # Keep in sync with the page default at the `max_tokens` <input> below (currently 1024).
        max_tokens = int(req.get("max_tokens", 1024))
        temperature = float(req.get("temperature", 0.7))
        top_k = int(req.get("top_k", 40))
        top_p = float(req.get("top_p", 0.95))
        seed = int(req.get("seed", -1))
        stream = bool(req.get("stream", True))

        if seed < 0:
            seed = int(time.time() * 1000) & 0x7FFFFFFF
        max_tokens = max(1, min(max_tokens, MAX_OUT))

        def render():
            if not chat_mode:
                return prompt, False
            text = prompt
            out = dev.chat(text, system)
            if no_think:
                # the model's jinja template emits an empty think block when enable_thinking
                # is false; llama_chat_apply_template() cannot take that flag, so append the
                # same bytes it would have produced.
                out += "<think>\n\n</think>\n\n"
            return out, True

        try:
            try:
                rendered, special = render()
            except SplitError:
                dev.close()                     # reconnect once, the device may have restarted
                rendered, special = render()
            ids = dev.tokenize(rendered, add_special=not special)
        except Exception as exc:                          # noqa: BLE001
            return self._send(503, json.dumps({"error": "tokenize failed: %s" % exc}))

        if len(ids) + max_tokens > CTX:
            return self._send(400, json.dumps({
                "error": "context overflow",
                "prompt_tokens": len(ids),
                "max_tokens": max_tokens,
                "context": CTX,
            }))

        t0 = time.time()

        if not stream:
            acc = []
            try:
                for tok, piece in dev.generate(ids, max_tokens, temperature, top_k, top_p, seed):
                    acc.append((tok, piece))
            except Exception as exc:                      # noqa: BLE001
                return self._send(500, json.dumps({"error": str(exc)}))
            text = "".join(p for _, p in acc)
            if chat_mode and no_think:
                text = re.sub(r"^\s*<think>\s*</think>\s*", "", text, count=1)
            return self._send(200, json.dumps({
                "text": text,
                "tokens": [t for t, _ in acc],
                "n_prompt": len(ids),
                "n_out": len(acc),
                "seconds": round(time.time() - t0, 3),
                "finish_reason": "length" if dev.truncated else "stop",
            }))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        n_out = 0
        gone = False          # the browser went away; keep draining, but stop writing
        filt = ThinkSkip() if (chat_mode and no_think) else None
        gen = dev.generate(ids, max_tokens, temperature, top_k, top_p, seed)
        try:
            while True:
                try:
                    tok, piece = next(gen)
                except StopIteration:
                    break
                n_out += 1
                if gone:
                    continue
                if filt is not None:
                    piece = filt.feed(piece)
                    if not piece:
                        continue
                payload = {"token": piece, "id": tok}
                try:
                    self.wfile.write(("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    # closing the tab should stop the device, not let it generate into the void
                    gone = True
                    dev.cancel()
            if filt is not None and not gone:
                tail = filt.flush()
                if tail:
                    self.wfile.write(("data: " + json.dumps({"token": tail, "id": -1},
                                                            ensure_ascii=False) + "\n\n").encode())
                    self.wfile.flush()
        except Exception as exc:                          # noqa: BLE001
            if not gone:
                try:
                    self.wfile.write(("data: " + json.dumps({"error": str(exc)}) + "\n\n").encode())
                    self.wfile.flush()
                except OSError:
                    pass
            return

        if gone:
            return

        done = {"done": True, "n_out": n_out, "n_prompt": len(ids),
                "cancelled": bool(dev.cancelled),
                "seconds": round(time.time() - t0, 3)}
        self.wfile.write(("data: " + json.dumps(done) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    global LISTEN_PORT, DEVICE_PORT
    if len(sys.argv) > 1:
        LISTEN_PORT = int(sys.argv[1])
    if len(sys.argv) > 2:
        DEVICE_PORT = int(sys.argv[2])
    # rebuild the shared connection now that the real port is known
    global CLIENT
    CLIENT = SplitClient()

    print("device 127.0.0.1:%d  ->  http://127.0.0.1:%d" % (DEVICE_PORT, LISTEN_PORT), flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
