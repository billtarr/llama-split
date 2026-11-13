// llama-split: two-process layer-split pipeline for llama.cpp, plus a line-protocol server
//
// stage A computes layers [0, K) and exports the hidden state; stage B computes [K, n_layer),
// samples the next token and hands it back. the two stages live in separate processes so each
// gets its own NPU session / IOVA window (required on Snapdragon 8 Gen 2, see docs/LEDGER.md).
//
// modes:
//   one-shot  : llama-split -m model -p "prompt" -c 256 -n 16 -k 16 --embd 2560
//   single    : llama-split -m model -p "prompt" -c 256 -n 16            (no split, reference)
//   server    : llama-split -m model -c 8192 -k 16 --embd 2560 --serve 18080
//
// server protocol (one request per line, responses are lines):
//   TOK <hex-utf8>                                   -> TOKS <n> <id> ...
//   DETOK <n> <id> ...                               -> TEXT <hex-utf8>
//   GEN <rid> <max> <temp> <top_k> <top_p> <seed> <nprompt> <id> ...  -> P <rid> <tok> <hex>
//                                                                    -> ... -> E <rid> <status> <nout>
//   PING                                             -> PONG
//   QUIT                                             -> BYE

#include "llama.h"
#include "stage_abi.h"
#include <dlfcn.h>

#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <sys/eventfd.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>

#include <algorithm>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>

static const char * TAG_A = "[A]";
static const char * TAG_B = "[B]";

// ------------------------------------------------------------------ shared pipe

// request-wide state: one per process chain, changes only at request boundaries
struct shm_hdr {
    int32_t cmd;        // 1 chunk ready / 2 reset request / 3 quit
    int32_t next_token; // token chosen by the last stage
    int32_t done;       // 1 = EOS seen / nothing more to generate
    int32_t n_predict;
    int32_t top_k;
    float   temp;
    float   top_p;
    int32_t seed;
};

// per-slot metadata. it must live next to the data, not in shm_hdr: with three or more stages
// stage 0 keeps pushing prefill chunks while the downstream stages are still working, so a
// single shared copy would be overwritten before the last stage reads it.
struct slot_hdr {
    int32_t n_tok;
    int32_t pos;
    int32_t sample;
    int32_t pad;
};

// one stage's view of the pipeline. stage i writes slot[i] and reads slot[i-1]; the last stage
// also samples and hands the token back to stage 0 through ev_tok.
struct pipe_ctx {
    float *   hidden;    // out: this stage's slot
    int32_t * toks;      // out: token ids for the rows in `hidden`
    float *   in_hidden; // in: upstream slot (workers only)
    int32_t * in_toks;
    shm_hdr * hdr;
    slot_hdr *       out_hdr;  // metadata for this stage's slot
    const slot_hdr * in_hdr;   // metadata of the upstream slot (workers)
    int       ev_out;     // post after `hidden` is filled
    int       ev_free;    // wait before overwriting `hidden` (skipped on the first push)
    int       ev_in;      // wait for upstream data (workers only)
    int       ev_free_in; // post after copying the upstream slot out
    int       ev_tok;     // last stage posts the sampled token, stage 0 waits for it
    int       n_embd;
    int       n_ctx;
    int       n_batch;
    int       n_stage;
    int       is_first;
    int       is_last;
    int       pushed;
};

struct gen_params {
    int      max_tokens;
    float    temp;
    int      top_k;
    float    top_p;
    uint32_t seed;
};

typedef void (* token_cb)(void * ud, llama_token tok);

// blocking wait with a watchdog: a lost event used to wedge the whole chain silently, which is
// indistinguishable from "the model is thinking" on the client side
static int g_ev_timeout_ms = 300000;

// idle waits (a worker sitting between requests) must not expire: the user may leave the page
// for hours and the chain still has to be there when the next request arrives
static void ev_wait_forever(int fd) {
    uint64_t v = 0;
    for (;;) {
        struct pollfd pfd;
        pfd.fd      = fd;
        pfd.events  = POLLIN;
        pfd.revents = 0;
        const int r = poll(&pfd, 1, -1);
        if (r < 0) {
            if (errno == EINTR) continue;
            perror("poll eventfd");
            exit(1);
        }
        if (read(fd, &v, sizeof(v)) == sizeof(v)) return;
        if (errno == EINTR || errno == EAGAIN) continue;
        perror("read eventfd");
        exit(1);
    }
}

static void ev_wait(int fd) {
    uint64_t v = 0;
    for (;;) {
        struct pollfd pfd;
        pfd.fd      = fd;
        pfd.events  = POLLIN;
        pfd.revents = 0;
        const int r = poll(&pfd, 1, g_ev_timeout_ms);
        if (r < 0) {
            if (errno == EINTR) continue;
            perror("poll eventfd");
            exit(1);
        }
        if (r == 0) {
            fprintf(stderr, "[M] ev_wait: no event on fd %d after %d ms - giving up\n",
                    fd, g_ev_timeout_ms);
            exit(3);
        }
        if (read(fd, &v, sizeof(v)) == sizeof(v)) return;
        if (errno == EINTR || errno == EAGAIN) continue;
        perror("read eventfd");
        exit(1);
    }
}

static void ev_post(int fd) {
    uint64_t v = 1;
    if (write(fd, &v, sizeof(v)) != sizeof(v)) {
        perror("write eventfd");
        exit(1);
    }
}

// discard any pending posts so the waiter counts start from zero at a request boundary
static void ev_drain(int fd) {
    if (fd < 0) return;
    const int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    uint64_t v = 0;
    while (read(fd, &v, sizeof(v)) == sizeof(v)) { v = 0; }
    fcntl(fd, F_SETFL, flags);
    errno = 0;
}

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static void logf(const char * tag, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "%s ", tag);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fflush(stderr);
}

// ------------------------------------------------------------------ ipc micro-benchmark

// what a stage boundary actually costs: one wakeup round trip plus the hidden-state copy.
// --ipc-bench [n] runs it without touching a model.
static int ipc_bench(int iters, int n_embd) {
    const size_t bytes = sizeof(float) * (size_t) n_embd;
    char * shm = (char *) mmap(nullptr, bytes * 2, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (shm == MAP_FAILED) { perror("mmap bench"); return 1; }

    const int ev_a2b = eventfd(0, 0);
    const int ev_b2a = eventfd(0, 0);
    if (ev_a2b < 0 || ev_b2a < 0) { perror("eventfd bench"); return 1; }

    const pid_t pid = fork();
    if (pid == 0) {
        char * dst = (char *) malloc(bytes);
        for (;;) {
            ev_wait_forever(ev_a2b);
            memcpy(dst, shm, bytes);          // stand-in for hidden -> llama_batch.embd
            ev_post(ev_b2a);
        }
    }

    for (int i = 0; i < 64; ++i) { memset(shm, i, bytes); ev_post(ev_a2b); ev_wait_forever(ev_b2a); }

    double t0 = now_ms();
    for (int i = 0; i < iters; ++i) { ev_post(ev_a2b); ev_wait_forever(ev_b2a); }
    const double rtt = 1000.0 * (now_ms() - t0) / iters;

    t0 = now_ms();
    for (int i = 0; i < iters; ++i) {
        memset(shm, i & 0xff, bytes);
        ev_post(ev_a2b); ev_wait_forever(ev_b2a);
    }
    const double rtt_copy = 1000.0 * (now_ms() - t0) / iters;

    // the peer asleep for ~20 ms is the closest stand-in for the decode cadence
    t0 = now_ms();
    for (int i = 0; i < iters; ++i) {
        usleep(20000);
        ev_post(ev_a2b); ev_wait_forever(ev_b2a);
    }
    const double rtt_idle = 1000.0 * (now_ms() - t0) / iters - 20000.0;

    // local memcpy, no IPC, for reference
    char * a = (char *) malloc(bytes), * b = (char *) malloc(bytes);
    t0 = now_ms();
    const int N = iters * 20;
    for (int i = 0; i < N; ++i) memcpy(b, a, bytes);
    const double cpy = 1000.0 * (now_ms() - t0) / N;

    fprintf(stderr, "[bench] n_embd %d (%zu B per hidden state), %d rounds\n", n_embd, bytes, iters);
    fprintf(stderr, "[bench]   local memcpy                     %8.1f us\n", cpy);
    fprintf(stderr, "[bench]   eventfd round trip (peer parked)  %8.1f us\n", rtt);
    fprintf(stderr, "[bench]   round trip + copy through shm     %8.1f us\n", rtt_copy);
    fprintf(stderr, "[bench]   round trip after 20 ms idle       %8.1f us\n", rtt_idle);

    kill(pid, SIGTERM);
    waitpid(pid, nullptr, 0);
    free(a); free(b);
    return 0;
}

// ------------------------------------------------------------------ device selection

// restrict the model to the HTP device(s): without this, weights get split across OpenCL and HTP
// and prefill drops by ~4x (see docs/LEDGER.md L-024)
static std::vector<ggml_backend_dev_t> g_htp_devs;

static ggml_backend_dev_t * htp_devices() {
    g_htp_devs.clear();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(d);
        if (name && strstr(name, "HTP")) {
            g_htp_devs.push_back(d);
        }
    }
    g_htp_devs.push_back(nullptr);
    return g_htp_devs.data();
}

// ------------------------------------------------------------------ tensor routing (generalisation step 2)
//
// Today a clear win comes from moving `output.weight` (Q6_K, 796 MiB, read once per token) from the
// CPU to the DSP: the CPU matmul costs ~75 ms/token and, worse, it drags every other stage down
// (see docs/LEDGER.md L-058). It only fits when the last stage is small enough, so the choice is a
// planning decision, not a constant.
//
// `--route PATTERN=cpu|htp` (repeatable) makes that expressible, and subsumes the env hacks that
// used to be hardcoded here (SPLIT_OUTPUT_HTP / SPLIT_OT_EXTRA / SPLIT_PLE_ON_HTP).
// It is the first half of the "tensor -> backend" axis of the generic router; the `stage:N` target
// that would cut a layer in half is deliberately NOT accepted yet -- intra-layer stage cuts need
// the graph to split at a tensor boundary, which is a different (much bigger) change.
struct route_t {
    const char * pat;
    int          to_cpu;   // 1 = CPU, 0 = DSP
};

static route_t g_routes[24];
static int     g_n_routes = 0;

static void add_route(const char * pat, int to_cpu) {
    if (!pat || !*pat) return;
    if (g_n_routes >= (int) (sizeof(g_routes) / sizeof(g_routes[0]))) {
        fprintf(stderr, "--route: too many rules (max %zu)\n",
                sizeof(g_routes) / sizeof(g_routes[0]));
        return;
    }
    g_routes[g_n_routes].pat    = strdup(pat);
    g_routes[g_n_routes].to_cpu = to_cpu;
    g_n_routes++;
}

static int  g_output_on_htp = 0;   // set by --route '.*output\.weight=htp'

// the buft of the DSP device, needed to *override back* to the NPU after a broader cpu rule
static ggml_backend_buffer_type_t htp_buft() {
    static ggml_backend_buffer_type_t cached = nullptr;
    static bool done = false;
    if (!done) {
        done = true;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const char * name = ggml_backend_dev_name(d);
            if (name && strstr(name, "HTP")) { cached = ggml_backend_dev_buffer_type(d); break; }
        }
    }
    return cached;
}

// the shared tensors below are only used by a GET_ROWS with a handful of rows, but they are big:
// per_layer_token_embd is ~1.4 GiB per stage and token_embd ~0.7 GiB, which does not fit in the
// DSP IOVA window (~2.2 GiB) next to the layer weights. keep them on the CPU (same as -ot).
static llama_model_tensor_buft_override g_overrides[32];

static const llama_model_tensor_buft_override * cpu_buft_overrides() {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) return nullptr;

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(cpu);

    // per_layer / token_embd / output are big lookup or lm_head tables. on a tied-embedding model
    // `output` does not exist as a separate tensor, so the pattern is simply inert there.
    // SPLIT_OUTPUT_HTP=1 keeps the lm_head on the NPU instead (only if it fits next to the layers).
    const bool output_cpu = getenv("SPLIT_OUTPUT_HTP") == nullptr;
    const char * pats[] = { ".*per_layer.*", ".*token_embd.*", ".*output\\.weight" };
    const bool   keep[] = { getenv("SPLIT_PLE_ON_HTP") == nullptr, true, output_cpu };

    int n = 0;

    // user rules first: the loader takes the first match, so an explicit --route beats a default
    for (int i = 0; i < g_n_routes && n < 30; ++i) {
        ggml_backend_buffer_type_t b = g_routes[i].to_cpu ? buft : htp_buft();
        if (!b) continue;                      // no DSP: nothing to route to
        g_overrides[n].pattern = g_routes[i].pat;
        g_overrides[n].buft    = b;
        n++;
    }

    for (int i = 0; i < 3; ++i) {
        if (!keep[i]) continue;
        if (n >= 31) break;
        // skip a default identical to a user rule: first match wins, so it would be shadowed
        // anyway -- skipping just keeps the debug log readable
        bool shadowed = false;
        for (int k = 0; k < g_n_routes; ++k) {
            if (!strcmp(g_routes[k].pat, pats[i])) { shadowed = true; break; }
        }
        if (shadowed) continue;
        g_overrides[n].pattern = pats[i];
        g_overrides[n].buft    = buft;
        n++;
    }
    // SPLIT_OT_EXTRA lets a test pin arbitrary layers to the CPU (e.g. "blk\\.(2[4-9]|3[01])\\.")
    const char * extra = getenv("SPLIT_OT_EXTRA");
    if (extra && n < 31) {
        g_overrides[n].pattern = extra;
        g_overrides[n].buft    = buft;
        n++;
    }
    g_overrides[n] = { nullptr, nullptr };

    return g_overrides;
}

static int  g_verbose = 0;

// a client may send CANCEL at any time; we only honour it between tokens, which is the one point
// where every stage is idle and the pipeline counters are consistent (see L-047)
static volatile int g_cancel_req = 0;
static int          g_cancel_fd  = -1;

static void poll_cancel() {
    if (g_cancel_fd < 0) return;
    struct pollfd pfd;
    pfd.fd      = g_cancel_fd;
    pfd.events  = POLLIN;
    pfd.revents = 0;
    const int pr = poll(&pfd, 1, 0);
    if (pr <= 0) return;
    if (g_verbose) fprintf(stderr, "[dbg] poll_cancel readable revents=%d\n", pfd.revents);
    char buf[64];
    const ssize_t n = recv(g_cancel_fd, buf, sizeof(buf) - 1, 0);
    if (n == 0) { g_cancel_req = 1; return; }   // client closed the socket: stop generating
    if (n < 0) return;
    buf[n] = 0;
    if (strstr(buf, "CANCEL")) g_cancel_req = 1;
}

// what is actually loaded - reported over the wire so a client never has to guess (the web page
// used to hard-code the model name and ended up lying about it)
static const char * g_info_model   = "";
static char         g_info_split[96] = "single";
static int          g_info_n_stage = 1;
static int  g_cpu_only = 0;    // reference run: no HTP, pure CPU (ground truth for correctness)
static const char * g_inject = nullptr;   // run stage B alone on a dumped hidden state (prefill only)
static std::vector<float> g_inject_buf;
static int  g_inject_n_tok = 0;
static const char * g_dump = nullptr;
static const char * g_logits_dump = nullptr;
static int  g_send_tokens = 1;  // always carry the ids: an embd-only ubatch makes stage B
                                // non-deterministic on HTP (see docs/LEDGER.md L-035), and models
                                // with per-layer embeddings need them anyway. 2 = send zeros.
static int  g_serve_port = 0;
static int  g_slots      = 1;   // contexts per stage: >1 enables request pipelining (④)
// external stages (stage ABI): g_ext_after[i] = library inserted immediately after llama stage i
static const char * g_ext_after[8] = {nullptr};   // capped like V2_MAX_STAGE
static int          g_n_ext = 0;
static int  g_n_batch    = 256; // physical batch; the 8k context reserves compute buffers per
                                // ubatch, and 8192 needs ~9 GB (see L-042)

static void dump_logits(const char * tag, const float * logits, int n_vocab) {
    if (!logits) { logf(tag, "no logits\n"); return; }
    int t[5] = {0,0,0,0,0};
    for (int i = 0; i < n_vocab; ++i) {
        for (int k = 0; k < 5; ++k) {
            if (logits[i] > logits[t[k]]) {
                for (int j = 4; j > k; --j) t[j] = t[j-1];
                t[k] = i;
                break;
            }
        }
    }
    logf(tag, "top5:");
    for (int k = 0; k < 5; ++k) fprintf(stderr, " %d:%.4f", t[k], logits[t[k]]);
    fprintf(stderr, "\n");
    if (g_logits_dump) {
        FILE * f = fopen(g_logits_dump, "wb");
        if (f) { fwrite(logits, sizeof(float), n_vocab, f); fclose(f); }
    }
}

// ------------------------------------------------------------------ stage A: generation driver

// pushes the prompt through layers [0,K) in n_batch chunks, then drives the token loop.
// returns 0 on success; *out_n receives the number of tokens emitted through cb.
static int a_generate(llama_context * ctx, pipe_ctx & pc, const llama_vocab * vocab,
                      const std::vector<llama_token> & prompt, const gen_params & gp,
                      token_cb cb, void * ud, int * out_n) {
    (void) vocab;
    *out_n = 0;
    g_cancel_req = 0;

    const int n_embd  = pc.n_embd;
    const int n_batch = pc.n_batch;
    const int n_prompt = (int) prompt.size();
    if (n_prompt <= 0) return 1;

    // hand the sampling parameters to the last stage, then reset the whole chain
    pc.hdr->n_predict = gp.max_tokens;
    pc.hdr->top_k     = gp.top_k;
    pc.hdr->temp      = gp.temp;
    pc.hdr->top_p     = gp.top_p;
    pc.hdr->seed      = (int32_t) gp.seed;
    pc.hdr->cmd       = 2;
    ev_post(pc.ev_out);
    ev_wait(pc.ev_tok);
    llama_memory_clear(llama_get_memory(ctx), true);
    pc.pushed = 0;
    ev_drain(pc.ev_free);

    int pos   = 0;
    int done  = 0;
    int n_out = 0;
    llama_token cur = 0;

    // ---- prefill (chunked: an 8k ubatch would reserve ~9 GB of compute buffers on stage B)
    for (int off = 0; off < n_prompt; off += n_batch) {
        const int n = std::min(n_batch, n_prompt - off);

        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            b.token    [i] = prompt[off + i];
            b.pos      [i] = pos + i;
            b.n_seq_id [i] = 1;
            b.seq_id   [i][0] = 0;
            b.logits   [i] = 1;   // every row needs its hidden state
        }
        b.n_tokens = n;

        const double t0 = now_ms();
        if (llama_decode(ctx, b) != 0) {
            llama_batch_free(b);
            logf(TAG_A, "prefill chunk failed at %d\n", off);
            return 1;
        }
        const double t1 = now_ms();

        const float * e = llama_get_embeddings(ctx);
        if (!e) { llama_batch_free(b); logf(TAG_A, "no embeddings\n"); return 1; }
        // the slot may still be in use by the downstream stage: release first, then write
        if (pc.pushed) ev_wait(pc.ev_free);
        memcpy(pc.hidden, e, sizeof(float) * (size_t) n_embd * n);
        for (int i = 0; i < n; ++i) pc.toks[i] = prompt[off + i];
        llama_batch_free(b);

        logf(TAG_A, "prefill %d/%d tok in %.1f ms (%.1f t/s)\n",
             off + n, n_prompt, t1 - t0, 1000.0 * n / (t1 - t0));

        if (g_dump && off + n >= n_prompt) {
            FILE * f = fopen(g_dump, "wb");
            if (f) { fwrite(pc.hidden, sizeof(float), (size_t) n_embd * n, f); fclose(f); }
        }

        const int last = (off + n >= n_prompt) ? 1 : 0;
        pc.hdr->cmd    = 1;
        pc.out_hdr->n_tok  = n;
        pc.out_hdr->pos    = pos;
        pc.out_hdr->sample = last;
        pc.hdr->done   = 0;
        pos += n;
        pc.pushed = 1;

        ev_post(pc.ev_out);
        if (last) {
            ev_wait(pc.ev_tok);
            cur  = (llama_token) pc.hdr->next_token;
            done = pc.hdr->done;
        }
    }

    if (done) { *out_n = 0; return 0; }

    cb(ud, cur);
    n_out++;

    // ---- decode: one row per round trip
    const double td0 = now_ms();
    while (!done && n_out < gp.max_tokens && !g_cancel_req) {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token    [0] = cur;
        b.pos      [0] = pos;
        b.n_seq_id [0] = 1;
        b.seq_id   [0][0] = 0;
        b.logits   [0] = 1;
        b.n_tokens = 1;

        const double ta0 = now_ms();
        if (llama_decode(ctx, b) != 0) {
            llama_batch_free(b);
            logf(TAG_A, "decode failed at %d\n", n_out);
            break;
        }
        const double ta1 = now_ms();
        if (g_verbose) logf(TAG_A, "chunk 1 tok @%d %.1f ms\n", pos, ta1 - ta0);
        const float * e = llama_get_embeddings(ctx);
        if (!e) { llama_batch_free(b); logf(TAG_A, "no embeddings in decode\n"); break; }
        if (pc.pushed) ev_wait(pc.ev_free);
        memcpy(pc.hidden, e, sizeof(float) * n_embd);
        pc.toks[0] = cur;
        llama_batch_free(b);
        pos++;

        pc.hdr->cmd    = 1;
        pc.out_hdr->n_tok  = 1;
        pc.out_hdr->pos    = pos - 1;
        pc.out_hdr->sample = 1;
        pc.hdr->done   = 0;

        pc.pushed = 1;
        ev_post(pc.ev_out);
        ev_wait(pc.ev_tok);

        cur  = (llama_token) pc.hdr->next_token;
        done = pc.hdr->done;

        cb(ud, cur);
        n_out++;
    }
    const double td = now_ms() - td0;
    logf(TAG_A, "decode %d tok in %.1f ms (%.2f t/s)\n", n_out - 1, td,
         n_out > 1 ? 1000.0 * (n_out - 1) / td : 0.0);

    *out_n = n_out;
    return 0;
}

// ------------------------------------------------------------------ stage B

// a non-first stage: consumes the upstream hidden state, decodes it, forwards the result.
// the last stage samples instead of forwarding, and posts the token back to stage 0.
static int stage_worker(int idx, const char * model_path, int n_ctx, int n_threads, pipe_ctx & pc) {
    char tag[8];
    snprintf(tag, sizeof(tag), "[%d]", idx);

    const double t0_all = now_ms();
    llama_backend_init();
    logf(tag, "backend init %.0f ms\n", now_ms() - t0_all);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = g_cpu_only ? 0 : 99;
    mp.devices = g_cpu_only ? nullptr : htp_devices();
    mp.tensor_buft_overrides = g_cpu_only ? nullptr : cpu_buft_overrides();

    const double t0_load = now_ms();
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { logf(tag, "model load failed\n"); return 1; }
    logf(tag, "model load %.0f ms\n", now_ms() - t0_load);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = n_ctx;
    cp.n_batch   = pc.n_batch;
    cp.n_ubatch  = pc.n_batch;
    cp.n_threads = n_threads;
    cp.no_perf   = true;

    const double t0_ctx = now_ms();
    llama_context * ctx = nullptr;
    std::vector<llama_context *> extra_ctx;
    for (int s = 0; s < g_slots; ++s) {
        llama_context * c = llama_init_from_model(model, cp);
        if (!c) {
            logf(tag, "ctx init failed at slot %d/%d (n_ctx %d, batch %d)\n",
                 s, g_slots, n_ctx, pc.n_batch);
            return 1;
        }
        if (s == 0) ctx = c; else extra_ctx.push_back(c);
    }
    logf(tag, "ctx init %.0f ms for %d slot(s) (total %.0f ms)\n",
         now_ms() - t0_ctx, g_slots, now_ms() - t0_all);

    if (!pc.is_last) llama_set_embeddings(ctx, true);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd(model);
    if (n_embd != pc.n_embd) { logf(tag, "n_embd mismatch\n"); return 1; }

    logf(tag, "ready (ctx %d, batch %d%s)\n", n_ctx, pc.n_batch, pc.is_last ? ", last" : "");

    llama_sampler * smpl = nullptr;
    std::vector<llama_token> toks(pc.n_batch > 0 ? (size_t) pc.n_batch : 1);

    // the last stage has no downstream: nothing to notify
    auto to_down = [&pc]() { if (pc.ev_out >= 0) ev_post(pc.ev_out); };

    for (;;) {
        ev_wait_forever(pc.ev_in);   // idle between requests: no watchdog here
        const int32_t cmd = pc.hdr->cmd;

        if (cmd == 3) { to_down(); break; }

        if (cmd == 2) {
            llama_memory_clear(llama_get_memory(ctx), true);
            if (smpl) { llama_sampler_free(smpl); smpl = nullptr; }
            pc.pushed = 0;
            ev_drain(pc.ev_free);
            ev_drain(pc.ev_free_in);
            // stage 0 blocks on ev_tok right after the reset, so the last stage must answer
            if (pc.is_last) ev_post(pc.ev_tok);
            else            to_down();
            continue;
        }
        if (cmd != 1) { to_down(); continue; }

        // snapshot everything now: the upstream is free to reuse its slot header as soon as we
        // post ev_free_in below, so re-reading these fields later picks up the NEXT chunk's values
        const int n_tok     = pc.in_hdr->n_tok;
        const int bpos      = pc.in_hdr->pos;
        const int in_sample = pc.in_hdr->sample;
        const int do_sample = (in_sample != 0) && pc.is_last;

        llama_batch b = llama_batch_init(n_tok, n_embd, 1);
        memcpy(b.embd, pc.in_hidden, sizeof(float) * (size_t) n_embd * n_tok);
        for (int i = 0; i < n_tok; ++i) {
            toks[i] = g_send_tokens ? ((g_send_tokens == 2) ? 0 : pc.in_toks[i]) : 0;
            b.pos[i] = bpos + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            // a mid stage must emit the hidden state of EVERY row - that is what selects the
            // embedding outputs. only the last stage wants logits, and only for the final row.
            b.logits[i] = pc.is_last ? ((i == n_tok - 1) ? 1 : 0) : 1;
        }
        if (g_send_tokens) b.token = toks.data();
        b.n_tokens = n_tok;

        ev_post(pc.ev_free_in);   // upstream may reuse its slot from here on

        const double t0 = now_ms();
        const int rc = llama_decode(ctx, b);
        const double t1 = now_ms();
        b.token = nullptr;
        llama_batch_free(b);

        if (rc != 0) {
            logf(tag, "decode failed (n_tok %d pos %d)\n", n_tok, bpos);
            pc.hdr->done = 1;
            if (pc.is_last) { pc.hdr->next_token = 0; ev_post(pc.ev_tok); }
            else            { to_down(); }
            continue;
        }

        if (pc.is_last) {
            if (do_sample) {
                if (!smpl) {
                    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
                    smpl = llama_sampler_chain_init(sp);
                    if (pc.hdr->temp <= 0.0f) {
                        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(1));
                    } else {
                        if (pc.hdr->top_k > 0) {
                            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(pc.hdr->top_k));
                        }
                        if (pc.hdr->top_p > 0.0f && pc.hdr->top_p < 1.0f) {
                            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(pc.hdr->top_p, 1));
                        }
                        llama_sampler_chain_add(smpl, llama_sampler_init_temp(pc.hdr->temp));
                    }
                    llama_sampler_chain_add(smpl, llama_sampler_init_dist((uint32_t) pc.hdr->seed));
                    if (g_verbose) logf(tag, "sampler: temp %.2f top_k %d top_p %.2f seed %d\n",
                                        pc.hdr->temp, pc.hdr->top_k, pc.hdr->top_p, pc.hdr->seed);
                }
                const double ts0 = now_ms();
                const llama_token t = llama_sampler_sample(smpl, ctx, -1);
                if (g_verbose) logf(tag, "sample %.1f ms\n", now_ms() - ts0);
                pc.hdr->next_token = (int32_t) t;
                pc.hdr->done = (t == llama_vocab_eos(vocab)) ? 1 : 0;
                ev_post(pc.ev_tok);
            }
            // no sampling (mid-prefill chunk): stage 0 is not waiting on anything
        } else {
            const float * e = llama_get_embeddings(ctx);
            if (!e) { logf(tag, "no embeddings\n"); pc.hdr->done = 1; to_down(); continue; }
            if (pc.pushed) ev_wait(pc.ev_free);
            if (pc.out_hdr) {
                pc.out_hdr->n_tok  = n_tok;
                pc.out_hdr->pos    = bpos;
                pc.out_hdr->sample = in_sample;
            }
            memcpy(pc.hidden, e, sizeof(float) * (size_t) n_embd * n_tok);
            for (int i = 0; i < n_tok; ++i) pc.toks[i] = toks[i];
            pc.pushed = 1;
            to_down();
        }

        if (g_verbose) logf(tag, "chunk %d tok @%d %.1f ms%s\n", n_tok, bpos, t1 - t0,
                            do_sample ? " (sample)" : "");
    }

    if (smpl) llama_sampler_free(smpl);
    logf(tag, "exit\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

// replay a dumped hidden state through one stage (debug aid, no pipeline)
static int stage_replay(const char * model_path, int n_ctx, int n_threads, pipe_ctx & pc) {
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = g_cpu_only ? 0 : 99;
    mp.devices = g_cpu_only ? nullptr : htp_devices();
    mp.tensor_buft_overrides = g_cpu_only ? nullptr : cpu_buft_overrides();

    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { logf("[B]", "model load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = pc.n_batch; cp.n_ubatch = pc.n_batch;
    cp.n_threads = n_threads; cp.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { logf("[B]", "ctx init failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd(model);

    const int n_tok = g_inject_n_tok;
    llama_batch b = llama_batch_init(n_tok, n_embd, 1);
    memcpy(b.embd, g_inject_buf.data(), sizeof(float) * (size_t) n_embd * n_tok);
    std::vector<llama_token> toks((size_t) n_tok, 0);
    for (int i = 0; i < n_tok; ++i) {
        b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
        b.logits[i] = (i == n_tok - 1) ? 1 : 0;
    }
    if (g_send_tokens) b.token = toks.data();
    b.n_tokens = n_tok;
    llama_decode(ctx, b);
    b.token = nullptr;
    llama_batch_free(b);

    dump_logits("[B]", llama_get_logits_ith(ctx, -1), n_vocab);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

// ------------------------------------------------------------------ server (runs inside stage A)

static std::string to_hex(const char * p, size_t n) {
    static const char * D = "0123456789abcdef";
    std::string s;
    s.resize(n * 2);
    for (size_t i = 0; i < n; ++i) {
        const unsigned char c = (unsigned char) p[i];
        s[2*i]     = D[c >> 4];
        s[2*i + 1] = D[c & 15];
    }
    return s;
}

static std::string from_hex(const std::string & s) {
    auto val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    std::string out;
    out.reserve(s.size() / 2);
    for (size_t i = 0; i + 1 < s.size(); i += 2) {
        const int hi = val(s[i]), lo = val(s[i+1]);
        if (hi < 0 || lo < 0) break;
        out.push_back((char) ((hi << 4) | lo));
    }
    return out;
}

// length of the longest prefix that ends on a UTF-8 code point boundary
static size_t utf8_complete(const std::string & s) {
    const size_t n = s.size();
    size_t i = n;
    size_t back = 0;
    while (i > 0 && back < 4) {
        --i;
        ++back;
        const unsigned char c = (unsigned char) s[i];
        if ((c & 0xC0) != 0x80) {
            const size_t need = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
            return (n - i >= need) ? n : i;
        }
    }
    return n;
}

struct stream_ud {
    FILE * out;
    const llama_vocab * vocab;
    int    rid;
    std::string pending;   // bytes held back because they end mid code point
};

static void stream_cb(void * ud_, llama_token tok) {
    stream_ud * u = (stream_ud *) ud_;
    poll_cancel();
    char buf[512];
    const int n = llama_token_to_piece(u->vocab, tok, buf, sizeof(buf), 0, false);

    if (n > 0) u->pending.append(buf, (size_t) n);

    const size_t cut = utf8_complete(u->pending);
    if (cut > 0) {
        const std::string piece = u->pending.substr(0, cut);
        u->pending.erase(0, cut);
        fprintf(u->out, "P %d %d %s\n", u->rid, (int) tok, to_hex(piece.data(), piece.size()).c_str());
    } else {
        // nothing complete yet, still report the token id so the client can count
        fprintf(u->out, "P %d %d \n", u->rid, (int) tok);
    }
    fflush(u->out);
}

static void serve_client(FILE * fh, llama_context * ctx, pipe_ctx & pc,
                         const llama_vocab * vocab, int n_ctx, const char * tmpl) {
    char * line = nullptr;
    size_t cap = 0;
    ssize_t len;

    while ((len = getline(&line, &cap, fh)) > 0) {
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        if (len == 0) continue;

        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        if (cmd == "PING") {
            fprintf(fh, "PONG\n");
            fflush(fh);
            continue;
        }
        if (cmd == "INFO") {
            const std::string hex = to_hex(g_info_model, strlen(g_info_model));
            fprintf(fh, "INFO %s %d %d %d %d %s\n",
                    hex.c_str(), n_ctx, g_info_n_stage, pc.n_embd,
                    llama_vocab_n_tokens(vocab), g_info_split);
            fflush(fh);
            continue;
        }
        if (cmd == "CANCEL") {
            continue;   // nothing running; stay silent so the reply stream keeps its shape
        }
        if (cmd == "QUIT") {
            fprintf(fh, "BYE\n");
            fflush(fh);
            break;
        }
        if (cmd == "TOK" || cmd == "TOKN") {
            // TOK adds BOS, TOKN does not (already-formatted chat prompts must not get one)
            const bool add_special = (cmd == "TOK");
            std::string hex;
            iss >> hex;
            const std::string text = from_hex(hex);
            int n = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0,
                                    add_special, true);
            std::vector<llama_token> toks(n > 0 ? n : 0);
            if (n > 0) llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), n,
                                      add_special, true);
            fprintf(fh, "TOKS %d", n > 0 ? n : 0);
            for (auto t : toks) fprintf(fh, " %d", (int) t);
            fprintf(fh, "\n");
            fflush(fh);
            continue;
        }
        if (cmd == "CHAT") {
            // CHAT <hex-system|-> <hex-user>  ->  CHATT <hex-rendered>
            std::string hsys, husr;
            iss >> hsys >> husr;
            const std::string sys = (hsys == "-") ? std::string() : from_hex(hsys);
            const std::string usr = from_hex(husr);

            llama_chat_message msgs[3];
            int n_msg = 0;
            if (!sys.empty()) { msgs[n_msg].role = "system"; msgs[n_msg].content = sys.c_str(); n_msg++; }
            msgs[n_msg].role = "user"; msgs[n_msg].content = usr.c_str(); n_msg++;

            std::string rendered;
            int32_t need = llama_chat_apply_template(tmpl ? tmpl : "chatml", msgs, n_msg, true, nullptr, 0);
            if (need > 0) {
                std::vector<char> buf((size_t) need + 1);
                const int32_t got = llama_chat_apply_template(tmpl ? tmpl : "chatml", msgs, n_msg, true,
                                                              buf.data(), need + 1);
                if (got > 0) rendered.assign(buf.data(), (size_t) got);
            }
            if (rendered.empty()) {
                // hard fallback: Qwen chatml
                if (!sys.empty()) rendered += "<|im_start|>system\n" + sys + "<|im_end|>\n";
                rendered += "<|im_start|>user\n" + usr + "<|im_end|>\n<|im_start|>assistant\n";
            }
            fprintf(fh, "CHATT %s\n", to_hex(rendered.data(), rendered.size()).c_str());
            fflush(fh);
            continue;
        }
        if (cmd == "DETOK") {
            int n = 0;
            iss >> n;
            std::vector<llama_token> toks(n);
            for (int i = 0; i < n; ++i) { int t = 0; iss >> t; toks[i] = (llama_token) t; }
            std::string text;
            char buf[1024];
            for (auto t : toks) {
                const int k = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
                if (k > 0) text.append(buf, (size_t) k);
            }
            fprintf(fh, "TEXT %s\n", to_hex(text.data(), text.size()).c_str());
            fflush(fh);
            continue;
        }
        if (cmd == "GEN") {
            stream_ud u;
            u.out = fh;
            u.vocab = vocab;
            u.rid = 0;
            int max_tokens = 256, top_k = 40, nprompt = 0;
            float temp = 0.7f, top_p = 0.95f;
            unsigned seed = 0;
            iss >> u.rid >> max_tokens >> temp >> top_k >> top_p >> seed >> nprompt;

            std::vector<llama_token> prompt(nprompt > 0 ? nprompt : 0);
            for (int i = 0; i < nprompt; ++i) { int t = 0; iss >> t; prompt[i] = (llama_token) t; }

            if (nprompt <= 0) {
                fprintf(fh, "E %d empty_prompt 0\n", u.rid);
                fflush(fh);
                continue;
            }
            if (nprompt + max_tokens > n_ctx) {
                fprintf(fh, "E %d ctx_overflow %d\n", u.rid, nprompt);
                fflush(fh);
                continue;
            }
            if (max_tokens <= 0) {
                fprintf(fh, "E %d bad_max 0\n", u.rid);
                fflush(fh);
                continue;
            }

            gen_params gp;
            gp.max_tokens = max_tokens;
            gp.temp = temp;
            gp.top_k = top_k;
            gp.top_p = top_p;
            gp.seed = seed;

            const double t0 = now_ms();
            int n_out = 0;
            g_cancel_fd = fileno(fh);      // the token callback polls this for "CANCEL"
            if (g_verbose) logf(TAG_A, "[dbg] cancel fd = %d\n", g_cancel_fd);
            const int rc = a_generate(ctx, pc, vocab, prompt, gp, stream_cb, &u, &n_out);
            g_cancel_fd = -1;
            const double dt = now_ms() - t0;

            // flush whatever is left (a dangling partial code point is dropped on purpose)
            if (!u.pending.empty()) {
                fprintf(fh, "P %d -1 %s\n", u.rid, to_hex(u.pending.data(), u.pending.size()).c_str());
                u.pending.clear();
            }
            const char * status = g_cancel_req ? "cancelled" : (rc == 0 ? "ok" : "error");
            fprintf(fh, "E %d %s %d %.0f\n", u.rid, status, n_out, dt);
            fflush(fh);

            logf(TAG_A, "GEN %d: %d prompt + %d out in %.0f ms (%.2f t/s)\n",
                 u.rid, nprompt, n_out, dt, n_out > 1 ? 1000.0 * (n_out - 1) / dt : 0.0);
            continue;
        }

        fprintf(fh, "ERR unknown command\n");
        fflush(fh);
    }
    free(line);
}

static int serve_loop(llama_context * ctx, pipe_ctx & pc, const llama_vocab * vocab, int n_ctx,
                      const char * tmpl) {
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) { perror("socket"); return 1; }

    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t) g_serve_port);

    if (bind(lfd, (sockaddr *) &addr, sizeof(addr)) != 0) { perror("bind"); return 1; }
    if (listen(lfd, 4) != 0) { perror("listen"); return 1; }

    logf(TAG_A, "serving on 127.0.0.1:%d\n", g_serve_port);

    for (;;) {
        const int cfd = accept(lfd, nullptr, nullptr);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            break;
        }
        logf(TAG_A, "client connected\n");
        FILE * fh = fdopen(cfd, "r+");
        if (!fh) { close(cfd); continue; }
        serve_client(fh, ctx, pc, vocab, n_ctx, tmpl);
        fclose(fh);
        logf(TAG_A, "client gone\n");
    }
    return 0;
}

// ------------------------------------------------------------------ stage A entry

static int stage_a(const char * model_path, int n_ctx, int n_threads, pipe_ctx & pc,
                   const char * prompt_str, int n_predict) {
    const double t0_all = now_ms();
    llama_backend_init();
    logf(TAG_A, "backend init %.0f ms\n", now_ms() - t0_all);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = g_cpu_only ? 0 : 99;
    mp.devices = g_cpu_only ? nullptr : htp_devices();
    mp.tensor_buft_overrides = g_cpu_only ? nullptr : cpu_buft_overrides();

    const double t0_load = now_ms();
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { logf(TAG_A, "model load failed\n"); return 1; }
    logf(TAG_A, "model load %.0f ms\n", now_ms() - t0_load);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = n_ctx;
    cp.n_batch   = pc.n_batch;
    cp.n_ubatch  = pc.n_batch;
    cp.n_threads = n_threads;
    cp.no_perf   = true;

    const double t0_ctx = now_ms();
    llama_context * ctx = nullptr;
    std::vector<llama_context *> extra_ctx;   // slots 2..N, currently only to prove they fit
    for (int s = 0; s < g_slots; ++s) {
        llama_context * c = llama_init_from_model(model, cp);
        if (!c) {
            logf(TAG_A, "ctx init failed at slot %d/%d (n_ctx %d, batch %d)\n",
                 s, g_slots, n_ctx, pc.n_batch);
            return 1;
        }
        if (s == 0) ctx = c; else extra_ctx.push_back(c);
        logf(TAG_A, "ctx[%d] ready %.0f ms\n", s, now_ms() - t0_ctx);
    }
    logf(TAG_A, "ctx init %.0f ms for %d slot(s) (total %.0f ms)\n",
         now_ms() - t0_ctx, g_slots, now_ms() - t0_all);

    llama_set_embeddings(ctx, true);

    const int n_embd = llama_model_n_embd(model);
    if (n_embd != pc.n_embd) { logf(TAG_A, "n_embd mismatch: %d vs %d\n", n_embd, pc.n_embd); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    logf(TAG_A, "ready (ctx %d, batch %d, n_embd %d)\n", n_ctx, pc.n_batch, n_embd);

    int rc = 0;
    if (g_serve_port > 0) {
        const char * tmpl = llama_model_chat_template(model, nullptr);
        logf(TAG_A, "chat template: %s\n", tmpl ? "found" : "none (chatml fallback)");
        rc = serve_loop(ctx, pc, vocab, n_ctx, tmpl);
    } else {
        std::vector<llama_token> prompt;
        int n = -llama_tokenize(vocab, prompt_str, (int32_t) strlen(prompt_str), nullptr, 0, true, true);
        prompt.resize(n > 0 ? n : 0);
        if (n > 0) llama_tokenize(vocab, prompt_str, (int32_t) strlen(prompt_str), prompt.data(), n, true, true);
        logf(TAG_A, "prompt %d tokens\n", n);

        gen_params gp;
        gp.max_tokens = n_predict;
        gp.temp   = 0.0f;
        gp.top_k  = 0;
        gp.top_p  = 1.0f;
        gp.seed   = 0;

        int n_out = 0;
        fprintf(stderr, "[A] gen:");
        a_generate(ctx, pc, vocab, prompt, gp, [](void *, llama_token t) {
            fprintf(stderr, " %d", (int) t);
        }, nullptr, &n_out);
        fprintf(stderr, "\n");
    }

    // tell the rest of the chain we are done (propagates stage to stage)
    pc.hdr->cmd = 3;
    ev_post(pc.ev_out);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}

// ------------------------------------------------------------------ single process reference

static int run_single(const char * model_path, int n_ctx, int n_threads, int n_predict,
                      const char * prompt_str, std::vector<int> * out) {
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = g_cpu_only ? 0 : 99;
    mp.devices = g_cpu_only ? nullptr : htp_devices();
    mp.tensor_buft_overrides = g_cpu_only ? nullptr : cpu_buft_overrides();

    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "[S] model load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = std::min(n_ctx, g_n_batch); cp.n_ubatch = std::min(n_ctx, g_n_batch);
    cp.n_threads = n_threads; cp.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd(model);

    if (g_dump) llama_set_embeddings(ctx, true);

    std::vector<llama_token> prompt;
    {
        int n = -llama_tokenize(vocab, prompt_str, (int32_t) strlen(prompt_str), nullptr, 0, true, true);
        prompt.resize(n > 0 ? n : 0);
        if (n > 0) llama_tokenize(vocab, prompt_str, (int32_t) strlen(prompt_str), prompt.data(), n, true, true);
        fprintf(stderr, "[S] prompt %d tokens\n", n);
    }

    const int n_prompt = (int) prompt.size();

    double t_pre = 0;
    {
        for (int off = 0; off < n_prompt; off += cp.n_batch) {
            const int n = std::min((int) cp.n_batch, n_prompt - off);
            llama_batch b = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; ++i) {
                b.token[i] = prompt[off + i]; b.pos[i] = off + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
                b.logits[i] = (off + i == n_prompt - 1) ? 1 : 0;
            }
            b.n_tokens = n;
            const double t0 = now_ms();
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "[S] prefill failed\n"); }
            t_pre += now_ms() - t0;
            if (g_dump && off + n >= n_prompt) {
                const float * e = llama_get_embeddings(ctx);
                if (e) {
                    FILE * f = fopen(g_dump, "wb");
                    if (f) { fwrite(e, sizeof(float), (size_t) n_embd * n, f); fclose(f); }
                }
            }
            llama_batch_free(b);
        }
        dump_logits("[S]", llama_get_logits_ith(ctx, -1), n_vocab);
    }

    const double t_start = now_ms();
    int pos = n_prompt;
    int n_gen = 0;
    for (int i = 0; i < n_predict; ++i) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int k = 1; k < n_vocab; ++k) if (logits[k] > logits[best]) best = k;
        if (out) out->push_back(best);
        n_gen++;
        if (best == llama_vocab_eos(vocab)) break;

        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0] = best; b.pos[0] = pos++; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
        b.n_tokens = 1;
        llama_decode(ctx, b);
        llama_batch_free(b);
    }
    const double t_dec = now_ms() - t_start;

    fprintf(stderr, "[S] prefill %d tok %.1f ms (%.1f t/s) | %d tok %.1f ms -> %.2f t/s\n",
            n_prompt, t_pre, 1000.0 * n_prompt / t_pre, n_gen, t_dec,
            n_gen > 1 ? 1000.0 * (n_gen - 1) / t_dec : 0.0);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

// ------------------------------------------------------------------ v2: pipelined serve
//
// `--slots N` turns the chain into a real pipeline. Every stage owns one context per slot, and
// stage 0 keeps a small request state machine, so it can push a chunk for request B while
// request A is still waiting for its sampled token.
//
// The gain is purely cross-request: a single request cannot use it, because decode is strictly
// serial (token n+1 needs the logits of token n). With 3 stages the per-token split is roughly
// 29% / 26% / 48%, so a single request keeps each stage idle ~2/3 of the time; two requests fill
// the chain. See docs/60-llama-split/architecture.md §5.4.

#define V2_MAX_STAGE 8
#define V2_MAX_SLOT  8

struct v2_glob {
    int32_t quit;
    int32_t n_stage;
    int32_t n_slot;
    int32_t pad;
};

// one chunk descriptor. Unlike slot_hdr this also carries the sampling parameters: with several
// requests in flight they differ per request, so they cannot live in a global header.
struct v2_hdr {
    int32_t state;      // 1 = data ready
    int32_t rid;        // request id (also used to rebuild the sampler)
    int32_t n_tok;
    int32_t pos;
    int32_t sample;     // 1 = the last stage must sample here
    int32_t n_predict;
    int32_t top_k;
    int32_t seed;
    float   temp;
    float   top_p;
    int32_t next_token;
    int32_t done;       // 1 = eos or budget exhausted
    int32_t chunk_seq;  // per-request monotonic chunk id, checked by stage 0 for skew
};

struct v2_chan {
    v2_hdr *  hdr;
    float *   hidden;
    int32_t * toks;
};

struct v2_sess {
    int      fd   = -1;
    FILE *   fh   = nullptr;
    std::mutex m;
    std::condition_variable cv;
    std::string pending;          // bytes held back mid code point
    std::vector<llama_token> out; // tokens waiting to be written to the socket
    bool     finished = false;
    const char * status = "ok";
    int      rid = 0;
    bool     cancel = false;
    bool     gone = false;        // client disconnected
    int      n_out = 0;           // filled in by the engine when the request finishes
    double   ms = 0;
};

struct v2_req {
    bool       used = false;
    int        rid = 0;
    v2_sess *  sess = nullptr;
    std::vector<llama_token> prompt;
    int        ip = 0;            // prompt tokens already pushed
    int        pos = 0;
    int        n_out = 0;
    int        max_tokens = 0;
    llama_token cur = 0;
    bool       can_push = false;  // free to compute another chunk
    float      temp = 0, top_p = 1;
    int        top_k = 0;
    uint32_t   seed = 0;
    double     t0 = 0;
    int32_t    seq = 0;        // chunks pushed so far
    int32_t    expect = 0;     // chunk id stage 0 is waiting for
    bool       waiting = false;
};

static void v2_emit(v2_req & r, llama_token t) {
    if (!r.sess) return;
    std::lock_guard<std::mutex> lk(r.sess->m);
    r.sess->out.push_back(t);
    r.sess->cv.notify_one();
}

static void v2_finish(v2_req & r, const char * status) {
    if (r.sess) {
        std::lock_guard<std::mutex> lk(r.sess->m);
        r.sess->finished = true;
        r.sess->status = status;
        r.sess->n_out = r.n_out;
        r.sess->ms = r.t0 > 0 ? now_ms() - r.t0 : 0;
        r.sess->cv.notify_one();
    }
    r.used = false;
}

// eventfd table, filled by run_v2() before fork so every stage inherits it
static int g_v2_ev_d[V2_MAX_STAGE][V2_MAX_SLOT];
static int g_v2_ev_t[V2_MAX_SLOT];

// ---------------- worker: stages 1..n-1

static int v2_worker(int idx, const char * model_path, int n_ctx, int n_threads,
                     v2_glob * glob, v2_chan * chan) {
    char tag[8];
    snprintf(tag, sizeof(tag), "[%d]", idx);

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = g_cpu_only ? 0 : 99;
    mp.devices = g_cpu_only ? nullptr : htp_devices();
    mp.tensor_buft_overrides = g_cpu_only ? nullptr : cpu_buft_overrides();

    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { logf(tag, "model load failed\n"); return 1; }

    const int n_embd = llama_model_n_embd(model);
    const int n_batch = g_n_batch;
    const bool is_last = (idx == glob->n_stage - 1);
    const int n_slot = glob->n_slot;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_batch; cp.n_ubatch = n_batch;
    cp.n_threads = n_threads; cp.no_perf = true;

    llama_context * ctx[V2_MAX_SLOT] = {nullptr};
    for (int k = 0; k < n_slot; ++k) {
        ctx[k] = llama_init_from_model(model, cp);
        if (!ctx[k]) { logf(tag, "ctx init failed at slot %d\n", k); return 1; }
        if (!is_last) llama_set_embeddings(ctx[k], true);
    }
    logf(tag, "ready, %d slots%s\n", n_slot, is_last ? ", last" : "");

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler * smpl[V2_MAX_SLOT] = {nullptr};
    int smpl_rid[V2_MAX_SLOT] = {-1, -1, -1, -1, -1, -1, -1, -1};
    std::vector<llama_token> toks((size_t) n_batch);

    for (;;) {
        struct pollfd pfd[V2_MAX_SLOT];
        for (int k = 0; k < n_slot; ++k) {
            pfd[k].fd = g_v2_ev_d[idx - 1][k];
            pfd[k].events = POLLIN;
            pfd[k].revents = 0;
        }
        if (poll(pfd, (nfds_t) n_slot, -1) < 0) {
            if (errno == EINTR) continue;
            perror("poll worker");
            return 1;
        }
        if (glob->quit) break;

        for (int k = 0; k < n_slot; ++k) {
            if (!(pfd[k].revents & POLLIN)) continue;
            uint64_t v = 0;
            if (read(pfd[k].fd, &v, sizeof(v)) != sizeof(v)) continue;
            if (glob->quit) break;

            v2_chan in  = chan[(size_t)(idx - 1) * n_slot + k];
            v2_chan out = chan[(size_t) idx       * n_slot + k];

            const int n_tok  = in.hdr->n_tok;
            const int bpos   = in.hdr->pos;
            const int sample = in.hdr->sample;
            const int32_t cseq = in.hdr->chunk_seq;

            // a chunk at position 0 starts a new request: this stage's KV still holds the
            // previous one and stage 0 cannot clear it from another process
            if (bpos == 0) llama_memory_clear(llama_get_memory(ctx[k]), true);

            llama_batch b = llama_batch_init(n_tok, n_embd, 1);
            memcpy(b.embd, in.hidden, sizeof(float) * (size_t) n_embd * n_tok);
            for (int i = 0; i < n_tok; ++i) {
                toks[i] = g_send_tokens ? ((g_send_tokens == 2) ? 0 : in.toks[i]) : 0;
                b.pos[i] = bpos + i;
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = is_last ? ((sample && i == n_tok - 1) ? 1 : 0) : 1;
            }
            if (g_send_tokens) b.token = toks.data();
            b.n_tokens = n_tok;

            const double tw0 = now_ms();
            const int rc = llama_decode(ctx[k], b);
            const double tw1 = now_ms();
            if (g_verbose) logf(tag, "chunk %d tok @%d slot %d %.1f ms%s\n",
                                n_tok, bpos, k, tw1 - tw0, (is_last && sample) ? " (sample)" : "");
            b.token = nullptr;
            llama_batch_free(b);

            if (rc != 0) {
                logf(tag, "decode failed (n_tok %d pos %d slot %d)\n", n_tok, bpos, k);
                out.hdr->rid = in.hdr->rid;
                out.hdr->done = 1;
                out.hdr->next_token = 0;
                out.hdr->state = 1;
                ev_post(g_v2_ev_t[k]);
                continue;
            }

            if (is_last) {
                out.hdr->rid  = in.hdr->rid;
                out.hdr->done = 0;
                out.hdr->next_token = 0;
                out.hdr->n_tok = n_tok;
                out.hdr->pos = bpos;
                out.hdr->chunk_seq = cseq;
                // stage 0 reads `sample` back to decide whether this chunk produced a token or
                // was only the ack for a mid-prefill chunk: it MUST be written here too
                out.hdr->sample = sample;
                if (sample) {
                    const int rid = in.hdr->rid;
                    if (smpl_rid[k] != rid) {
                        if (smpl[k]) llama_sampler_free(smpl[k]);
                        llama_sampler_chain_params sp = llama_sampler_chain_default_params();
                        smpl[k] = llama_sampler_chain_init(sp);
                        if (in.hdr->temp <= 0.0f) {
                            llama_sampler_chain_add(smpl[k], llama_sampler_init_top_k(1));
                        } else {
                            if (in.hdr->top_k > 0)
                                llama_sampler_chain_add(smpl[k], llama_sampler_init_top_k(in.hdr->top_k));
                            if (in.hdr->top_p > 0.0f && in.hdr->top_p < 1.0f)
                                llama_sampler_chain_add(smpl[k], llama_sampler_init_top_p(in.hdr->top_p, 1));
                            llama_sampler_chain_add(smpl[k], llama_sampler_init_temp(in.hdr->temp));
                        }
                        llama_sampler_chain_add(smpl[k], llama_sampler_init_dist(in.hdr->seed));
                        smpl_rid[k] = rid;
                    }
                    const llama_token t = llama_sampler_sample(smpl[k], ctx[k], -1);
                    out.hdr->next_token = (int32_t) t;
                    out.hdr->done = (t == llama_vocab_eos(vocab)) ? 1 : 0;
                }
                out.hdr->state = 1;
                ev_post(g_v2_ev_t[k]);
            } else {
                const float * e = llama_get_embeddings(ctx[k]);
                if (!e) {
                    out.hdr->rid = in.hdr->rid;
                    out.hdr->done = 1;
                    out.hdr->state = 1;
                    ev_post(g_v2_ev_t[k]);
                    continue;
                }
                out.hdr->rid = in.hdr->rid;
                out.hdr->n_tok = n_tok;
                out.hdr->pos = bpos;
                out.hdr->sample = sample;
                out.hdr->chunk_seq = cseq;
                out.hdr->done = 0;
                memcpy(out.hidden, e, sizeof(float) * (size_t) n_embd * n_tok);
                for (int i = 0; i < n_tok; ++i) out.toks[i] = toks[i];
                out.hdr->state = 1;
                ev_post(g_v2_ev_d[idx][k]);
            }
        }
    }

    for (int k = 0; k < n_slot; ++k) { if (smpl[k]) llama_sampler_free(smpl[k]); llama_free(ctx[k]); }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}



// ---------------- external stage: anything that exports split_stage_get()
//
// Same shm ring, same eventfd protocol, same forwarding rules as a llama stage -- only the "compute"
// step differs. That is the whole point of the ABI: the chain does not know (or care) that this
// process links nothing from llama.cpp.
static int v2_worker_ext(int idx, const char * lib, int n_embd, int n_ctx, int n_batch,
                         v2_glob * glob, v2_chan * chan) {
    char tag[8];
    snprintf(tag, sizeof(tag), "[%d]", idx);

    void * h = dlopen(lib, RTLD_NOW);
    if (!h) { logf(tag, "dlopen %s failed: %s\n", lib, dlerror()); return 1; }

    typedef const split_stage * (*get_fn)(void);
    get_fn get = (get_fn) dlsym(h, "split_stage_get");
    if (!get) { logf(tag, "%s does not export split_stage_get\n", lib); return 1; }

    const split_stage * st = get();
    if (!st) { logf(tag, "%s returned a null stage\n", lib); return 1; }
    if (st->abi_version != SPLIT_STAGE_ABI_VERSION) {
        logf(tag, "stage '%s' speaks ABI %d, driver speaks %d\n",
             st->name ? st->name : "?", st->abi_version, SPLIT_STAGE_ABI_VERSION);
        return 1;
    }
    if (!st->run) { logf(tag, "stage '%s' has no run()\n", st->name ? st->name : "?"); return 1; }

    const int n_slot = glob->n_slot;
    if (st->open && st->open("", n_embd, n_ctx, n_slot) != 0) {
        logf(tag, "stage '%s' open() failed\n", st->name ? st->name : "?");
        return 1;
    }
    logf(tag, "external stage '%s' ready, %d slots (abi %d)\n",
         st->name ? st->name : "?", n_slot, st->abi_version);

    std::vector<int32_t> ids;
    for (;;) {
        struct pollfd pfd[V2_MAX_SLOT];
        for (int k = 0; k < n_slot; ++k) {
            pfd[k].fd = g_v2_ev_d[idx - 1][k];
            pfd[k].events = POLLIN;
            pfd[k].revents = 0;
        }
        if (poll(pfd, (nfds_t) n_slot, -1) < 0) {
            if (errno == EINTR) continue;
            perror("poll ext");
            break;
        }
        if (glob->quit) break;

        for (int k = 0; k < n_slot; ++k) {
            if (!(pfd[k].revents & POLLIN)) continue;
            uint64_t v = 0;
            if (read(pfd[k].fd, &v, sizeof(v)) != sizeof(v)) continue;
            if (glob->quit) break;

            v2_chan in  = chan[(size_t)(idx - 1) * n_slot + k];
            v2_chan out = chan[(size_t) idx       * n_slot + k];

            const int n_tok = in.hdr->n_tok;
            if (n_tok <= 0 || n_tok > n_batch) {
                logf(tag, "bad chunk (n_tok %d)\n", n_tok);
                out.hdr->rid = in.hdr->rid;
                out.hdr->done = 1;
                out.hdr->state = 1;
                ev_post(g_v2_ev_d[idx][k]);
                continue;
            }

            ids.assign(in.toks, in.toks + n_tok);
            split_slot_meta m;
            m.n_tok = n_tok; m.pos0 = in.hdr->pos; m.n_embd = n_embd;
            m.sample = in.hdr->sample; m.rid = in.hdr->rid;
            m.chunk_seq = in.hdr->chunk_seq; m.ids = ids.data();

            split_hidden hi = { in.hidden,  SPLIT_DTYPE_F32, n_embd, n_tok };
            split_hidden ho = { out.hidden, SPLIT_DTYPE_F32, n_embd, n_tok };

            const double tw0 = now_ms();
            int32_t tok = 0;
            const int rc = st->run(&m, &hi, &ho, &tok);
            if (g_verbose) logf(tag, "chunk %d tok @%d slot %d %.1f ms (ext)\n",
                                n_tok, in.hdr->pos, k, now_ms() - tw0);

            if (rc != 0) {
                logf(tag, "stage run failed rc %d (n_tok %d pos %d slot %d)\n",
                     rc, n_tok, in.hdr->pos, k);
                out.hdr->rid = in.hdr->rid;
                out.hdr->done = 1;
                out.hdr->state = 1;
                ev_post(g_v2_ev_d[idx][k]);
                continue;
            }

            out.hdr->rid       = in.hdr->rid;
            out.hdr->n_tok     = n_tok;
            out.hdr->pos       = in.hdr->pos;
            out.hdr->sample    = in.hdr->sample;
            out.hdr->chunk_seq = in.hdr->chunk_seq;
            out.hdr->done      = 0;
            out.hdr->state     = 1;
            memcpy(out.toks, in.toks, sizeof(int32_t) * (size_t) n_tok);
            ev_post(g_v2_ev_d[idx][k]);
        }
    }

    if (st->close) st->close();
    dlclose(h);
    return 0;
}

// ---------------- stage 0: request state machine + serving

static v2_glob *  g_v2_glob = nullptr;
static v2_chan *  g_v2_chan = nullptr;
static int        g_v2_n_stage = 0, g_v2_n_slot = 0, g_v2_n_embd = 0, g_v2_n_batch = 0;
static int        g_v2_n_ctx = 0;
static int        g_v2_ev_new = -1;
static llama_context *     g_v2_ctx[V2_MAX_SLOT];
static const llama_vocab * g_v2_vocab = nullptr;
static const char *        g_v2_tmpl = nullptr;
static std::mutex          g_v2_qm;
static std::deque<v2_req *> g_v2_q;

// compute one chunk for request r on slot k and hand it to stage 1
static void v2_push(v2_req & r, int k) {
    const int n_embd = g_v2_n_embd;
    const int remain = (int) r.prompt.size() - r.ip;
    const bool prefill = remain > 0;
    const int  n = prefill ? std::min(g_v2_n_batch, remain) : 1;

    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i]   = prefill ? r.prompt[(size_t) r.ip + i] : r.cur;
        b.pos[i]     = r.pos + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]  = 1;
    }
    b.n_tokens = n;

    const double t0 = now_ms();
    const int rc = llama_decode(g_v2_ctx[k], b);
    if (g_verbose) logf(TAG_A, "push rid %d slot %d %s %d tok @%d %.1f ms\n",
                        r.rid, k, prefill ? "prefill" : "decode", n, r.pos, now_ms() - t0);

    const float * e = rc == 0 ? llama_get_embeddings(g_v2_ctx[k]) : nullptr;
    v2_chan c = g_v2_chan[k];

    c.hdr->rid  = r.rid;
    c.hdr->n_tok = n;
    c.hdr->pos   = r.pos;
    c.hdr->n_predict = r.max_tokens;
    c.hdr->top_k = r.top_k;
    c.hdr->temp  = r.temp;
    c.hdr->top_p = r.top_p;
    c.hdr->seed  = (int32_t) r.seed;
    c.hdr->done  = 0;
    c.hdr->chunk_seq = ++r.seq;
    r.expect = r.seq;
    c.hdr->state = 1;

    if (!e) {
        logf(TAG_A, "no embeddings (rid %d slot %d)\n", r.rid, k);
        c.hdr->done = 1;
        c.hdr->next_token = 0;
        llama_batch_free(b);
        ev_post(g_v2_ev_d[0][k]);
        return;
    }

    memcpy(c.hidden, e, sizeof(float) * (size_t) n_embd * n);
    for (int i = 0; i < n; ++i) c.toks[i] = b.token[i];
    llama_batch_free(b);

    r.pos += n;
    if (prefill) r.ip += n;
    c.hdr->sample = prefill ? ((r.ip >= (int) r.prompt.size()) ? 1 : 0) : 1;

    ev_post(g_v2_ev_d[0][k]);
}

// the last stage answered: either a sampled token or "prefill chunk done, send the next one"
static void v2_handle(v2_req & r, int k, v2_hdr * h) {
    if (g_verbose) logf(TAG_A, "handle rid %d slot %d seq %d/%d sample %d done %d tok %d n_out %d\n",
                        r.rid, k, h->chunk_seq, r.expect, h->sample, h->done, h->next_token, r.n_out);
    if (!r.used) return;
    if (h->chunk_seq != r.expect) {
        // a stale event: this would silently desynchronise the request (see L-047)
        logf(TAG_A, "SKEW rid %d slot %d got seq %d want %d -- dropping request\n",
             r.rid, k, h->chunk_seq, r.expect);
        v2_finish(r, "skew");
        return;
    }

    if (r.sess && r.sess->cancel) { v2_finish(r, "cancelled"); return; }

    if (h->sample) {
        const llama_token t = (llama_token) h->next_token;
        v2_emit(r, t);
        r.n_out++;
        if (h->done) { v2_finish(r, "ok"); return; }
        if (r.n_out >= r.max_tokens) { v2_finish(r, "length"); return; }
        r.cur = t;
    } else if (h->done) {
        v2_finish(r, "error");
        return;
    }
    r.can_push = true;
}

static void * v2_engine(void *) {
    v2_req R[V2_MAX_SLOT];

    for (;;) {
        // 1) admit queued requests into free slots
        for (;;) {
            std::lock_guard<std::mutex> lk(g_v2_qm);
            if (g_v2_q.empty()) break;
            int k = -1;
            for (int i = 0; i < g_v2_n_slot; ++i) if (!R[i].used) { k = i; break; }
            if (k < 0) break;                      // pipeline full, leave it queued
            v2_req * q = g_v2_q.front(); g_v2_q.pop_front();
            R[k] = *q;
            delete q;
            R[k].used = true;
            R[k].can_push = true;
            R[k].pos = 0;
            R[k].ip = 0;
            R[k].n_out = 0;
            llama_memory_clear(llama_get_memory(g_v2_ctx[k]), true);
            if (g_verbose) logf(TAG_A, "admit rid %d -> slot %d (%d prompt tok)\n",
                                R[k].rid, k, (int) R[k].prompt.size());
        }

        // 2) push every slot that is free to run — this is what fills the idle stages
        for (int k = 0; k < g_v2_n_slot; ++k) {
            if (R[k].used && R[k].can_push) {
                v2_push(R[k], k);
                R[k].can_push = false;
            }
        }

        // 3) wait for a sampled token / ack / a new request
        struct pollfd pfd[V2_MAX_SLOT + 1];
        int np = 0;
        for (int k = 0; k < g_v2_n_slot; ++k) {
            pfd[np].fd = g_v2_ev_t[k]; pfd[np].events = POLLIN; pfd[np].revents = 0; np++;
        }
        pfd[np].fd = g_v2_ev_new; pfd[np].events = POLLIN; pfd[np].revents = 0; np++;

        if (poll(pfd, (nfds_t) np, -1) < 0) {
            if (errno == EINTR) continue;
            perror("poll engine");
            break;
        }
        if (g_v2_glob->quit) break;

        for (int k = 0; k < g_v2_n_slot; ++k) {
            if (!(pfd[k].revents & POLLIN)) continue;
            uint64_t v = 0;
            if (read(g_v2_ev_t[k], &v, sizeof(v)) != sizeof(v)) continue;
            v2_hdr * h = g_v2_chan[(size_t)(g_v2_n_stage - 1) * g_v2_n_slot + k].hdr;
            v2_handle(R[k], k, h);
        }
        if (pfd[g_v2_n_slot].revents & POLLIN) {
            uint64_t v = 0;
            if (read(g_v2_ev_new, &v, sizeof(v)) != sizeof(v)) {}
        }
    }
    return nullptr;
}

// ---------------- serving

static bool v2_fill(int fd, std::string & buf, bool & gone) {
    char tmp[4096];
    for (;;) {
        const ssize_t n = recv(fd, tmp, sizeof(tmp), MSG_DONTWAIT);
        if (n > 0) { buf.append(tmp, (size_t) n); continue; }
        if (n == 0) { gone = true; return false; }
        if (errno == EAGAIN || errno == EWOULDBLOCK) return true;
        if (errno == EINTR) continue;
        gone = true;
        return false;
    }
}

static void v2_write_tokens(v2_sess * s, const std::vector<llama_token> & out) {
    for (auto t : out) {
        char buf[512];
        const int n = llama_token_to_piece(g_v2_vocab, t, buf, sizeof(buf), 0, false);
        if (n > 0) s->pending.append(buf, (size_t) n);
        const size_t cut = utf8_complete(s->pending);
        std::string hex;
        if (cut > 0) { hex = to_hex(s->pending.data(), cut); s->pending.erase(0, cut); }
        fprintf(s->fh, "P %d %d %s\n", s->rid, (int) t, hex.c_str());
    }
    fflush(s->fh);
}

static void * v2_session(void * p) {
    v2_sess * s = (v2_sess *) p;
    std::string inbuf;
    bool waiting = false;
    double t0 = 0;

    while (!s->gone) {
        struct pollfd pfd;
        pfd.fd = s->fd; pfd.events = POLLIN; pfd.revents = 0;
        const int pr = poll(&pfd, 1, waiting ? 20 : -1);
        if (pr > 0 && (pfd.revents & (POLLIN | POLLHUP | POLLERR))) {
            v2_fill(s->fd, inbuf, s->gone);
        }
        if (s->gone) break;

        // complete lines
        size_t nl;
        while ((nl = inbuf.find('\n')) != std::string::npos) {
            std::string line = inbuf.substr(0, nl);
            inbuf.erase(0, nl + 1);
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (line.empty()) continue;

            if (waiting) {
                if (line == "CANCEL") s->cancel = true;
                continue;                       // ignore everything else while generating
            }

            std::istringstream iss(line);
            std::string cmd;
            iss >> cmd;

            if (cmd == "PING") { fprintf(s->fh, "PONG\n"); fflush(s->fh); continue; }
            if (cmd == "QUIT") { fprintf(s->fh, "BYE\n"); fflush(s->fh); s->gone = true; break; }
            if (cmd == "CANCEL") { continue; }
            if (cmd == "INFO") {
                const std::string hex = to_hex(g_info_model, strlen(g_info_model));
                fprintf(s->fh, "INFO %s %d %d %d %d %s\n", hex.c_str(), g_v2_n_ctx,
                        g_v2_n_stage, g_v2_n_embd, llama_vocab_n_tokens(g_v2_vocab), g_info_split);
                fflush(s->fh);
                continue;
            }
            if (cmd == "TOK" || cmd == "TOKN") {
                const bool add_special = (cmd == "TOK");
                std::string hex;
                iss >> hex;
                const std::string text = from_hex(hex);
                int n = -llama_tokenize(g_v2_vocab, text.c_str(), (int32_t) text.size(), nullptr, 0,
                                        add_special, true);
                std::vector<llama_token> toks(n > 0 ? n : 0);
                if (n > 0) llama_tokenize(g_v2_vocab, text.c_str(), (int32_t) text.size(), toks.data(),
                                          n, add_special, true);
                fprintf(s->fh, "TOKS %d", n > 0 ? n : 0);
                for (auto t : toks) fprintf(s->fh, " %d", (int) t);
                fprintf(s->fh, "\n");
                fflush(s->fh);
                continue;
            }
            if (cmd == "CHAT") {
                std::string hsys, husr;
                iss >> hsys >> husr;
                const std::string sys = (hsys == "-") ? std::string() : from_hex(hsys);
                const std::string usr = from_hex(husr);
                llama_chat_message msgs[3];
                int n_msg = 0;
                if (!sys.empty()) { msgs[n_msg].role = "system"; msgs[n_msg].content = sys.c_str(); n_msg++; }
                msgs[n_msg].role = "user"; msgs[n_msg].content = usr.c_str(); n_msg++;
                std::string rendered;
                const char * tm = g_v2_tmpl ? g_v2_tmpl : "chatml";
                int32_t need = llama_chat_apply_template(tm, msgs, n_msg, true, nullptr, 0);
                if (need > 0) {
                    std::vector<char> buf((size_t) need + 1);
                    if (llama_chat_apply_template(tm, msgs, n_msg, true, buf.data(), need + 1) > 0)
                        rendered.assign(buf.data(), (size_t) need);
                }
                if (rendered.empty()) {
                    if (!sys.empty()) rendered += "<|im_start|>system\n" + sys + "<|im_end|>\n";
                    rendered += "<|im_start|>user\n" + usr + "<|im_end|>\n<|im_start|>assistant\n";
                }
                fprintf(s->fh, "CHATT %s\n", to_hex(rendered.data(), rendered.size()).c_str());
                fflush(s->fh);
                continue;
            }
            if (cmd == "DETOK") {
                int n = 0;
                iss >> n;
                std::vector<llama_token> toks(n);
                for (int i = 0; i < n; ++i) { int t = 0; iss >> t; toks[i] = (llama_token) t; }
                std::string text;
                char buf[1024];
                for (auto t : toks) {
                    const int k = llama_token_to_piece(g_v2_vocab, t, buf, sizeof(buf), 0, false);
                    if (k > 0) text.append(buf, (size_t) k);
                }
                fprintf(s->fh, "TEXT %s\n", to_hex(text.data(), text.size()).c_str());
                fflush(s->fh);
                continue;
            }
            if (cmd == "GEN") {
                int rid = 0, max_tokens = 256, top_k = 40, nprompt = 0;
                float temp = 0.7f, top_p = 0.95f;
                unsigned seed = 0;
                iss >> rid >> max_tokens >> temp >> top_k >> top_p >> seed >> nprompt;
                std::vector<llama_token> prompt(nprompt > 0 ? nprompt : 0);
                for (int i = 0; i < nprompt; ++i) { int t = 0; iss >> t; prompt[i] = (llama_token) t; }

                if (nprompt <= 0) { fprintf(s->fh, "E %d empty_prompt 0\n", rid); fflush(s->fh); continue; }
                if (nprompt + max_tokens > g_v2_n_ctx) {
                    fprintf(s->fh, "E %d ctx_overflow %d\n", rid, nprompt); fflush(s->fh); continue;
                }
                if (max_tokens <= 0) { fprintf(s->fh, "E %d bad_max 0\n", rid); fflush(s->fh); continue; }

                v2_req * r = new v2_req();
                r->rid = rid;
                r->sess = s;
                r->prompt = std::move(prompt);
                r->max_tokens = max_tokens;
                r->temp = temp;
                r->top_k = top_k;
                r->top_p = top_p;
                r->seed = seed;
                r->t0 = now_ms();
                s->rid = rid;
                s->finished = false;
                s->status = "ok";
                s->n_out = 0;
                s->ms = 0;
                s->cancel = false;
                {
                    std::lock_guard<std::mutex> lk(g_v2_qm);
                    g_v2_q.push_back(r);
                }
                ev_post(g_v2_ev_new);
                waiting = true;
                t0 = now_ms();
                continue;
            }
        }

        // streaming: drain whatever the engine produced for the active request
        if (waiting) {
            std::vector<llama_token> out;
            bool fin = false;
            {
                std::lock_guard<std::mutex> lk(s->m);
                out.swap(s->out);
                fin = s->finished;
            }
            if (!out.empty()) v2_write_tokens(s, out);
            if (fin) {
                if (!s->pending.empty()) {
                    fprintf(s->fh, "P %d -1 %s\n", s->rid,
                            to_hex(s->pending.data(), s->pending.size()).c_str());
                    s->pending.clear();
                }
                fprintf(s->fh, "E %d %s %d %.0f\n", s->rid, s->status, s->n_out, now_ms() - t0);
                fflush(s->fh);
                waiting = false;
            }
        }
    }

    // NOTE: the session object is deliberately NOT freed. The engine may still hold a pointer to
    // it (a request can outlive its client), and freeing here destroyed the mutex under the
    // engine -> "FORTIFY: pthread_mutex_lock called on a destroyed mutex". Leaking a few hundred
    // bytes per connection is the cheap side of that trade.
    if (s->fh) fclose(s->fh);
    return nullptr;
}

static void * v2_accept(void * arg) {
    const int lfd = (int) (intptr_t) arg;
    for (;;) {
        const int cfd = accept(lfd, nullptr, nullptr);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            break;
        }
        v2_sess * s = new v2_sess();
        s->fd = cfd;
        s->fh = fdopen(cfd, "r+");
        if (!s->fh) { close(cfd); delete s; continue; }
        logf(TAG_A, "client connected (fd %d)\n", cfd);
        std::thread(v2_session, s).detach();
    }
    return nullptr;
}

// ---------------- entry

static int v2_stage0(const char * model_path, int n_ctx, int n_threads) {
    const double t0_all = now_ms();
    llama_backend_init();
    logf(TAG_A, "backend init %.0f ms\n", now_ms() - t0_all);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = g_cpu_only ? 0 : 99;
    mp.devices = g_cpu_only ? nullptr : htp_devices();
    mp.tensor_buft_overrides = g_cpu_only ? nullptr : cpu_buft_overrides();

    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { logf(TAG_A, "model load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = g_v2_n_batch;
    cp.n_ubatch = g_v2_n_batch;
    cp.n_threads = n_threads;
    cp.no_perf = true;

    const double t0_ctx = now_ms();
    for (int k = 0; k < g_v2_n_slot; ++k) {
        g_v2_ctx[k] = llama_init_from_model(model, cp);
        if (!g_v2_ctx[k]) { logf(TAG_A, "ctx init failed at slot %d\n", k); return 1; }
        llama_set_embeddings(g_v2_ctx[k], true);
    }
    logf(TAG_A, "ready: ctx %d x %d slot(s), batch %d, n_embd %d (init %.0f ms, total %.0f ms)\n",
         n_ctx, g_v2_n_slot, g_v2_n_batch, g_v2_n_embd, now_ms() - t0_ctx, now_ms() - t0_all);

    g_v2_vocab = llama_model_get_vocab(model);
    g_v2_tmpl  = llama_model_chat_template(model, nullptr);

    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) { perror("socket"); return 1; }
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t) g_serve_port);
    if (bind(lfd, (sockaddr *) &addr, sizeof(addr)) != 0) { perror("bind"); return 1; }
    if (listen(lfd, 8) != 0) { perror("listen"); return 1; }
    logf(TAG_A, "serving on 127.0.0.1:%d (%d slots)\n", g_serve_port, g_v2_n_slot);

    std::thread(v2_accept, (void *) (intptr_t) lfd).detach();

    v2_engine(nullptr);     // stage 0's own compute happens inside the engine loop

    g_v2_glob->quit = 1;
    for (int i = 0; i + 1 < g_v2_n_stage; ++i)
        for (int k = 0; k < g_v2_n_slot; ++k)
            ev_post(g_v2_ev_d[i][k]);          // wake every worker so it sees quit
    logf(TAG_A, "engine stopped\n");
    return 0;
}

static int run_v2(const char * model_path, int n_ctx, int n_threads, int n_embd,
                  const std::vector<int> & cuts) {
    const double t0_all = now_ms();
    const int n_llama = (int) cuts.size() + 1;

    // chain plan: entry >= 0 is llama stage i; entry -(i+1) is the external stage that runs right
    // after llama stage i. An external stage may not be last -- only the stage that owns the
    // vocabulary can sample the next token.
    std::vector<int> plan;
    for (int i = 0; i < n_llama; ++i) {
        plan.push_back(i);
        if (g_ext_after[i]) {
            if (i + 1 == n_llama) {
                logf(TAG_A, "--ext-after %d: the last llama stage owns sampling, put the external "
                            "stage earlier\n", i);
                return 1;
            }
            plan.push_back(-(i + 1));
        }
    }
    const int n_stage = (int) plan.size();
    const int n_slot  = std::min(g_slots, V2_MAX_SLOT);
    const int n_batch = std::min(g_n_batch, n_ctx);

    const size_t glob_bytes   = 4096;
    const size_t hidden_bytes = sizeof(float) * (size_t) n_embd * (size_t) n_batch;
    const size_t tok_bytes    = sizeof(int32_t) * (size_t) n_batch;
    const size_t hdr_bytes    = ((sizeof(v2_hdr) + 63) / 64) * 64;
    const size_t chan_bytes   = hdr_bytes + hidden_bytes + tok_bytes;
    const size_t n_chan       = (size_t) n_stage * (size_t) n_slot;
    const size_t shm_bytes    = glob_bytes + n_chan * chan_bytes;

    const char * shm_path = getenv("LLAMA_SPLIT_SHM");
    if (!shm_path) shm_path = "/data/local/tmp/llama-split-v2.shm";

    const int fd = open(shm_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) { perror("open shm"); return 1; }
    if (ftruncate(fd, (off_t) shm_bytes) != 0) { perror("ftruncate"); return 1; }
    void * base = mmap(nullptr, shm_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    memset(base, 0, shm_bytes);

    v2_glob * glob = (v2_glob *) base;
    glob->n_stage = n_stage;
    glob->n_slot  = n_slot;

    v2_chan * chan = (v2_chan *) calloc(n_chan, sizeof(v2_chan));
    for (int i = 0; i < n_stage; ++i) {
        for (int k = 0; k < n_slot; ++k) {
            char * sp = (char *) base + glob_bytes + ((size_t) i * n_slot + k) * chan_bytes;
            chan[(size_t) i * n_slot + k].hdr    = (v2_hdr *) sp;
            chan[(size_t) i * n_slot + k].hidden = (float *)   (sp + hdr_bytes);
            chan[(size_t) i * n_slot + k].toks   = (int32_t *) (sp + hdr_bytes + hidden_bytes);
        }
    }

    for (int i = 0; i + 1 < n_stage; ++i)
        for (int k = 0; k < n_slot; ++k)
            g_v2_ev_d[i][k] = eventfd(0, 0);
    for (int k = 0; k < n_slot; ++k) g_v2_ev_t[k] = eventfd(0, 0);
    g_v2_ev_new = eventfd(0, 0);

    g_v2_glob = glob; g_v2_chan = chan;
    g_v2_n_stage = n_stage; g_v2_n_slot = n_slot;
    g_v2_n_embd = n_embd; g_v2_n_batch = n_batch;
    g_v2_n_ctx = n_ctx;

    logf(TAG_A, "v2 pipeline: %d stages x %d slots (%.1f MiB shm, batch %d)\n",
         n_stage, n_slot, shm_bytes / 1048576.0, n_batch);
    if (g_n_ext > 0) {
        fprintf(stderr, "[M] chain:");
        for (int i = 0; i < n_stage; ++i) {
            if (plan[i] >= 0) fprintf(stderr, " llama%d", plan[i]);
            else              fprintf(stderr, " ext(%s)", g_ext_after[-plan[i] - 1]);
        }
        fprintf(stderr, "\n");
    }

    // fork workers first; only stage 0 ever starts threads (fork + threads do not mix)
    std::vector<pid_t> pids((size_t) n_stage, -1);
    static const char stage_names[] = "ABCDEFGH";
    for (int i = 0; i < n_stage; ++i) {
        const int li = plan[i];
        const pid_t pid = fork();
        if (pid == 0) {
            char lp[80];
            snprintf(lp, sizeof(lp), "/data/local/tmp/v2-%c.log", stage_names[i]);
            const int lg = open(lp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (lg >= 0) { dup2(lg, 1); dup2(lg, 2); }

            int rc;
            if (li < 0) {
                // external stage: no model, no layer range
                rc = v2_worker_ext(i, g_ext_after[-li - 1], n_embd, n_ctx, n_batch, glob, chan);
            } else {
                const int start = (li == 0) ? 0 : cuts[li - 1];
                const int end   = (li + 1 == n_llama) ? 100000 : cuts[li];
                char buf[32];
                snprintf(buf, sizeof(buf), "%d:%d", start, end);
                setenv("LLAMA_LAYER_SPLIT", buf, 1);
                rc = (li == 0) ? v2_stage0(model_path, n_ctx, n_threads)
                               : v2_worker(i, model_path, n_ctx, n_threads, glob, chan);
            }
            _exit(rc);
        }
        pids[i] = pid;
    }

    int sa = 0;
    for (;;) {
        int st = 0;
        const pid_t p = waitpid(-1, &st, 0);
        if (p <= 0) break;
        if (p == pids[0]) sa = st;
        break;
    }
    for (int i = 0; i < n_stage; ++i) if (pids[i] > 0 && pids[i] != 0) kill(pids[i], SIGTERM);

    logf(TAG_A, "stage 0 exited first (status %d), wall %.0f ms\n", sa, now_ms() - t0_all);
    return 0;
}

// ------------------------------------------------------------------ main

int main(int argc, char ** argv) {
    // a client that disconnects mid-stream must not kill us with SIGPIPE: the token callback
    // notices the EOF and treats it as a cancel instead
    signal(SIGPIPE, SIG_IGN);
    const char * model_path = nullptr;
    const char * prompt_str = "The capital of France is";
    const char * split_spec = nullptr;   // "11,22" = three stages
    int n_ctx     = 512;
    int n_predict = 16;
    int n_threads = 4;
    int split_k   = 0;   // single cut point, equivalent to --split <k>
    int n_embd    = 0;
    int bench_iters = 0;

    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if      (!strcmp(a, "-m")) { model_path = argv[++i]; }
        else if (!strcmp(a, "-p")) { prompt_str = argv[++i]; }
        else if (!strcmp(a, "-c")) { n_ctx = atoi(argv[++i]); }
        else if (!strcmp(a, "-n")) { n_predict = atoi(argv[++i]); }
        else if (!strcmp(a, "-t")) { n_threads = atoi(argv[++i]); }
        else if (!strcmp(a, "-k")) { split_k = atoi(argv[++i]); }
        else if (!strcmp(a, "--split")) { split_spec = argv[++i]; }
        else if (!strcmp(a, "--embd")) { n_embd = atoi(argv[++i]); }
        else if (!strcmp(a, "--route")) {
            // PATTERN=cpu|htp[;...]  -- repeatable, and ';' separates several rules in one argument
            char * spec = strdup(argv[++i]);
            for (char * s = spec; s && *s; ) {
                char * semi = strchr(s, ';');
                if (semi) *semi = '\0';
                char * eq = strrchr(s, '=');
                if (!eq) {
                    fprintf(stderr, "--route: '%s' is not PATTERN=target\n", s);
                } else {
                    *eq = '\0';
                    const char * tgt = eq + 1;
                    if (!strcmp(tgt, "cpu")) {
                        add_route(s, 1);
                    } else if (!strcmp(tgt, "htp")) {
                        add_route(s, 0);
                        if (strstr(s, "output")) g_output_on_htp = 1;
                    } else {
                        fprintf(stderr, "--route: unknown target '%s' (cpu|htp)\n", tgt);
                    }
                }
                s = semi ? semi + 1 : nullptr;
            }
        }
        else if (!strcmp(a, "--batch")) { g_n_batch = atoi(argv[++i]); }
        else if (!strcmp(a, "--serve")) { g_serve_port = atoi(argv[++i]); }
        else if (!strcmp(a, "--slots")) { g_slots = atoi(argv[++i]); }
        else if (!strcmp(a, "--ext-after")) {
            // N:PATH -- run the stage library PATH immediately after llama stage N (0-based).
            // it receives that stage's hidden state and forwards its own output to stage N+1.
            char * spec = strdup(argv[++i]);
            char * colon = strchr(spec, ':');
            if (!colon) { fprintf(stderr, "--ext-after: expected N:PATH\n"); return 1; }
            *colon = '\0';
            const int n = atoi(spec);
            if (n < 0 || n >= V2_MAX_STAGE) {
                fprintf(stderr, "--ext-after: stage %d out of range\n", n);
                return 1;
            }
            g_ext_after[n] = colon + 1;
            g_n_ext++;
        }
        else if (!strcmp(a, "-v")) { g_verbose = 1; }
        else if (!strcmp(a, "--dump")) { g_dump = argv[++i]; }
        else if (!strcmp(a, "--logits")) { g_logits_dump = argv[++i]; }
        else if (!strcmp(a, "--tokens")) { g_send_tokens = atoi(argv[++i]); }
        else if (!strcmp(a, "--cpu")) { g_cpu_only = 1; }
        else if (!strcmp(a, "--inject")) { g_inject = argv[++i]; }
        else if (!strcmp(a, "--ipc-bench")) { bench_iters = 500; }
        else { fprintf(stderr, "unknown arg %s\n", a); return 1; }
    }
    if (g_n_routes > 0) {
        fprintf(stderr, "[M] tensor routes (first match wins):");
        for (int i = 0; i < g_n_routes; ++i) {
            fprintf(stderr, " '%s'->%s", g_routes[i].pat, g_routes[i].to_cpu ? "cpu" : "htp");
        }
        fprintf(stderr, "%s\n", g_output_on_htp ? "   [lm_head on DSP]" : "");
    }
    if (bench_iters > 0) {        llama_backend_init();   // no model needed, but keep the runtime identical to a real run
        return ipc_bench(bench_iters, n_embd > 0 ? n_embd : 4096);
    }
    if (!model_path) {
        fprintf(stderr,
            "usage: %s -m model.gguf [-p prompt] [-c ctx] [-n predict] [--split a,b] [--embd N]\n"
            "       %s -m model.gguf -c 8192 -k 16 --embd 2560 --serve 18080\n"
            "       %s -m model.gguf -c 8192 --slots 2 --split 11,24 --embd 4096 --serve 18080 \\\n"
            "           --route '.*output\\.weight=htp'    (lm_head on the DSP: -13%% latency, L-058)\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    // ---- stage layout: cuts[i] is the first layer of stage i+1
    std::vector<int> cuts;
    if (split_spec) {
        const char * p = split_spec;
        while (*p) {
            char * e = nullptr;
            const long v = strtol(p, &e, 10);
            if (e == p) break;
            if (v > 0) cuts.push_back((int) v);
            p = (*e == ',') ? e + 1 : e;
            if (!*p) break;
        }
    } else if (split_k > 0) {
        cuts.push_back(split_k);
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    const int n_stage = (int) cuts.size() + 1;

    // publish what we are about to load (read back by the web UI through the INFO command)
    g_info_model   = model_path;
    g_info_n_stage = n_stage;
    {
        std::string sp;
        for (int i = 0; i < (int) cuts.size(); ++i) {
            if (i) sp += ",";
            sp += std::to_string(cuts[i]);
        }
        snprintf(g_info_split, sizeof(g_info_split), "%s", sp.empty() ? "single" : sp.c_str());
    }

    if (g_n_batch < 1) g_n_batch = 1;

    // single process reference (no split)
    if (n_stage < 2) {
        std::vector<int> out;
        run_single(model_path, n_ctx, n_threads, n_predict, prompt_str, &out);
        printf("tokens:");
        for (int t : out) printf(" %d", t);
        printf("\n");
        return 0;
    }

    // NOTE: the parent must never touch llama before fork(): a forked child inherits the
    // fastrpc / backend global state and segfaults on the first session allocation.
    if (n_embd <= 0) {
        fprintf(stderr, "n_embd must be given with --embd (see gguf metadata)\n");
        return 1;
    }

    // standalone replay of a dumped hidden state through the last stage
    if (g_inject) {
        FILE * f = fopen(g_inject, "rb");
        if (!f) { perror("open inject"); return 1; }
        fseek(f, 0, SEEK_END);
        const long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        g_inject_n_tok = (int) (sz / (long) (sizeof(float) * n_embd));
        g_inject_buf.resize((size_t) n_embd * g_inject_n_tok);
        if (fread(g_inject_buf.data(), sizeof(float), g_inject_buf.size(), f) != g_inject_buf.size()) {
            fprintf(stderr, "short read from %s\n", g_inject);
            return 1;
        }
        fclose(f);
        fprintf(stderr, "[M] inject %d tok from %s\n", g_inject_n_tok, g_inject);

        char buf[32];
        snprintf(buf, sizeof(buf), "%d:%d", cuts.back(), 100000);
        setenv("LLAMA_LAYER_SPLIT", buf, 1);

        pipe_ctx pc;
        memset(&pc, 0, sizeof(pc));
        pc.n_batch = g_n_batch;
        return stage_replay(model_path, n_ctx, n_threads, pc);
    }

    // ---- pipelined path: one context per slot per stage, cross-request overlap.
    // an external stage only exists on this path, so --ext-after selects it too (with 1 slot)
    if (g_slots > 1 || g_n_ext > 0) {
        if (!split_spec && split_k <= 0) {
            fprintf(stderr, "--slots/--ext-after require --split (a single-stage pipeline has "
                            "nothing to overlap)\n");
            return 1;
        }
        if (n_embd <= 0) {
            fprintf(stderr, "--slots requires --embd (the parent cannot touch llama)\n");
            return 1;
        }
        if (!g_serve_port) {
            fprintf(stderr, "--slots is only meaningful in serve mode (--serve <port>)\n");
            return 1;
        }
        return run_v2(model_path, n_ctx, n_threads, n_embd, cuts);
    }

    // ---- shared memory: one slot (hidden + ids) per stage
    const int n_batch = std::min(g_n_batch, n_ctx);
    const size_t hdr_bytes      = ((sizeof(shm_hdr) + 4095) / 4096) * 4096;
    const size_t slot_hdr_bytes = ((sizeof(slot_hdr) + 63) / 64) * 64;
    const size_t hidden_bytes   = sizeof(float) * (size_t) n_embd * (size_t) n_batch;
    const size_t token_bytes    = sizeof(int32_t) * (size_t) n_batch;
    const size_t slot_bytes     = slot_hdr_bytes + hidden_bytes + token_bytes;
    const size_t shm_bytes      = hdr_bytes + (size_t) n_stage * slot_bytes;

    const char * shm_path = getenv("LLAMA_SPLIT_SHM");
    if (!shm_path) shm_path = "/data/local/tmp/llama-split.shm";

    int fd = open(shm_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) { perror("open shm"); return 1; }
    if (ftruncate(fd, (off_t) shm_bytes) != 0) { perror("ftruncate"); return 1; }

    void * base = mmap(nullptr, shm_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    memset(base, 0, shm_bytes);

    shm_hdr * hdr = (shm_hdr *) base;

    // down[i]: stage i -> i+1 (data ready)   up[i]: stage i+1 -> i (slot reusable)
    std::vector<int> ev_down((size_t) n_stage, -1);
    std::vector<int> ev_up((size_t) n_stage, -1);
    for (int i = 0; i + 1 < n_stage; ++i) {
        ev_down[i] = eventfd(0, 0);
        ev_up[i]   = eventfd(0, 0);
        if (ev_down[i] < 0 || ev_up[i] < 0) { perror("eventfd"); return 1; }
    }
    const int ev_tok = eventfd(0, 0);
    if (ev_tok < 0) { perror("eventfd"); return 1; }

    std::vector<pipe_ctx> pcs((size_t) n_stage);
    for (int i = 0; i < n_stage; ++i) {
        pipe_ctx & c = pcs[i];
        memset(&c, 0, sizeof(c));
        char * sp   = (char *) base + hdr_bytes + (size_t) i * slot_bytes;
        c.hdr       = hdr;
        c.out_hdr   = (slot_hdr *) sp;
        c.hidden    = (float *)   (sp + slot_hdr_bytes);
        c.toks      = (int32_t *) ((char *) c.hidden + hidden_bytes);
        c.in_hdr    = (i > 0) ? pcs[i-1].out_hdr : nullptr;
        c.in_hidden = (i > 0) ? pcs[i-1].hidden : nullptr;
        c.in_toks   = (i > 0) ? pcs[i-1].toks   : nullptr;
        c.ev_out     = (i + 1 < n_stage) ? ev_down[i] : -1;
        c.ev_free    = (i + 1 < n_stage) ? ev_up[i]   : -1;
        c.ev_in      = (i > 0) ? ev_down[i-1] : -1;
        c.ev_free_in = (i > 0) ? ev_up[i-1]   : -1;
        c.ev_tok     = ev_tok;
        c.n_embd   = n_embd;
        c.n_ctx    = n_ctx;
        c.n_batch  = n_batch;
        c.n_stage  = n_stage;
        c.is_first = (i == 0);
        c.is_last  = (i == n_stage - 1);
    }

    fprintf(stderr, "[M] %d stages:", n_stage);
    for (int i = 0; i < n_stage; ++i) {
        const int start = (i == 0) ? 0 : cuts[i-1];
        if (i + 1 == n_stage) fprintf(stderr, " [%d,end)", start);
        else                  fprintf(stderr, " [%d,%d)", start, cuts[i]);
    }
    fprintf(stderr, "\n");

    const double t_all = now_ms();
    std::vector<pid_t> pids((size_t) n_stage, -1);

    static const char stage_names[] = "ABCDEFGH";
    for (int i = 0; i < n_stage; ++i) {
        const int start = (i == 0) ? 0 : cuts[i-1];
        const int end   = (i + 1 == n_stage) ? 100000 : cuts[i];

        const pid_t pid = fork();
        if (pid == 0) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%d:%d", start, end);
            setenv("LLAMA_LAYER_SPLIT", buf, 1);

            char lp[80];
            snprintf(lp, sizeof(lp), "/data/local/tmp/split-%c.log", stage_names[i]);
            const int lg = open(lp, O_WRONLY|O_CREAT|O_TRUNC, 0644);
            if (lg >= 0) { dup2(lg, 1); dup2(lg, 2); }

            const int rc = (i == 0)
                ? stage_a(model_path, n_ctx, n_threads, pcs[0], prompt_str, n_predict)
                : stage_worker(i, model_path, n_ctx, n_threads, pcs[i]);
            _exit(rc);
        }
        pids[i] = pid;
    }

    // watch EVERY stage, not just stage 0: if a worker dies the chain deadlocks and stage 0 keeps
    // the port open, so the service looks alive while every request hangs. When that happens take
    // the whole chain down and let the launcher restart it.
    int sa = 0;
    int dead = -1;
    for (;;) {
        int st = 0;
        const pid_t p = waitpid(-1, &st, 0);
        if (p <= 0) break;
        dead = (int) p;
        if (p == pids[0]) sa = st;
        break;
    }

    hdr->cmd = 3;
    ev_post(ev_down[0]);
    for (int i = 0; i < n_stage; ++i) {
        if (pids[i] == dead) continue;
        kill(pids[i], SIGTERM);
    }
    for (int i = 0; i < n_stage; ++i) {
        if (pids[i] == dead) continue;
        waitpid(pids[i], nullptr, 0);
    }

    fprintf(stderr, "stage %d exited first (status %d), wall %.1f ms\n",
            dead == pids[0] ? 0 : -1, sa, now_ms() - t_all);
    return (dead == pids[0]) ? 0 : 2;
}
