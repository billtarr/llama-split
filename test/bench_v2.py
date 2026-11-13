import socket, sys, time, threading

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18081
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
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
    r = c.line().split()
    return r[2:]

def one(rid, n, res):
    try:
        c = C()
        ids = tok(c, P)
        c.cmd("GEN %d %d %.4f %d %.4f %d %d %s" % (rid, n, 0.0, 0, 1.0, 0, len(ids), " ".join(ids)))
        toks = []; t0 = time.time()
        while True:
            l = c.line(); p = l.split(" ", 4)
            if p[0] == "P": toks.append(int(p[2]))
            elif p[0] == "E":
                res[rid] = (len(toks), time.time() - t0, l); break
    except Exception as e:
        res[rid] = (0, 0, "EXC %s" % e)

def run(nreq, label):
    res = {}
    t0 = time.time()
    th = [threading.Thread(target=one, args=(i, N, res)) for i in range(nreq)]
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    tot = sum(r[0] for r in res.values())
    print("%-12s %d req x %d tok: wall %.2fs  total %d tok  AGG %.2f t/s   per-req: %s"
          % (label, nreq, N, wall, tot, tot / wall,
             " ".join("%.2fs/%d" % (res[k][1], res[k][0]) for k in sorted(res))), flush=True)
    for k in sorted(res): print("   rid %d: %s" % (k, res[k][2]), flush=True)
    return tot / wall

run(1, "warmup")
a = run(1, "single")
b = run(2, "two-conc")
print("speedup: %.2fx" % (b / a), flush=True)
