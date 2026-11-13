import socket, sys, time, threading

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18081
N = 12
P = "The capital of France is"

class C:
    def __init__(self):
        self.s = socket.create_connection(("127.0.0.1", PORT), timeout=300)
        self.f = self.s.makefile("rwb")
    def cmd(self, l):
        self.f.write((l + "\n").encode()); self.f.flush()
    def line(self):
        l = self.f.readline()
        if not l: raise RuntimeError("eof")
        return l.decode().strip()

def tok(c, text):
    c.cmd("TOK " + text.encode().hex())
    return c.line().split()[2:]

def one(rid, res):
    c = C(); ids = tok(c, P)
    c.cmd("GEN %d %d %.4f %d %.4f %d %d %s" % (rid, N, 0.0, 0, 1.0, 0, len(ids), " ".join(ids)))
    toks = []
    while True:
        l = c.line(); p = l.split(" ", 3)
        if p[0] == "P": toks.append(int(p[2]))
        elif p[0] == "E":
            res[rid] = (toks, l); return

r1 = {}
one(10, r1)
base = r1[10][0]
print("single   :", base, flush=True)

r2 = {}
ts = [threading.Thread(target=one, args=(20 + i, r2)) for i in range(2)]
for t in ts: t.start()
for t in ts: t.join()
ok = True
for k in sorted(r2):
    same = r2[k][0] == base
    ok &= same
    print("conc rid %d: %s  %s" % (k, r2[k][0], "SAME" if same else "*** DIFF ***"), flush=True)
print("VERDICT:", "PASS (concurrent requests do not interfere)" if ok else "FAIL", flush=True)
