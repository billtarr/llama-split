import socket, sys, time, threading

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18081

class C:
    def __init__(self):
        self.s = socket.create_connection(("127.0.0.1", PORT), timeout=180)
        self.f = self.s.makefile("rwb")
    def cmd(self, line):
        self.f.write((line + "\n").encode()); self.f.flush()
    def line(self):
        l = self.f.readline()
        if not l: raise RuntimeError("eof")
        return l.decode().strip()

def tok(c, text):
    c.cmd("TOK " + text.encode().hex())
    r = c.line().split()
    assert r[0] == "TOKS", r
    return r[2:]

def gen(c, ids, n=16, temp=0.0, rid=1):
    t0 = time.time()
    c.cmd("GEN %d %d %.4f %d %.4f %d %d %s" % (rid, n, temp, 0, 1.0, 0, len(ids), " ".join(ids)))
    toks = []
    while True:
        l = c.line()
        p = l.split(" ", 4)
        if p[0] == "P":
            toks.append(int(p[2]))
        elif p[0] == "E":
            return toks, time.time() - t0, l

c = C()
c.cmd("INFO"); print("INFO:", c.line(), flush=True)
ids = tok(c, "The capital of France is")
print("prompt tokens:", len(ids), flush=True)
toks, dt, eline = gen(c, ids, n=8)
print("tokens:", toks, flush=True)
print("elapsed: %.2fs  %s" % (dt, eline), flush=True)
