int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * prompt_str = "The capital of France is";
    const char * split_spec = nullptr;   // "11,22" = three stages
    int n_ctx     = 512;
    int n_predict = 16;
    int n_threads = 4;
    int split_k   = 0;   // single cut point, equivalent to --split <k>
    int n_embd    = 0;

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
        else if (!strcmp(a, "--batch")) { g_n_batch = atoi(argv[++i]); }
        else if (!strcmp(a, "--serve")) { g_serve_port = atoi(argv[++i]); }
        else if (!strcmp(a, "-v")) { g_verbose = 1; }
        else if (!strcmp(a, "--dump")) { g_dump = argv[++i]; }
        else if (!strcmp(a, "--logits")) { g_logits_dump = argv[++i]; }
        else if (!strcmp(a, "--tokens")) { g_send_tokens = atoi(argv[++i]); }
        else if (!strcmp(a, "--cpu")) { g_cpu_only = 1; }
        else if (!strcmp(a, "--inject")) { g_inject = argv[++i]; }
        else { fprintf(stderr, "unknown arg %s\n", a); return 1; }
    }
    if (!model_path) {
        fprintf(stderr,
            "usage: %s -m model.gguf [-p prompt] [-c ctx] [-n predict] [--split a,b] [--embd N]\n"
            "       %s -m model.gguf -c 8192 -k 16 --embd 2560 --serve 18080\n"
            "       %s -m model.gguf -c 8192 --split 11,22 --embd 4096 --serve 18080\n",
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

    // ---- shared memory: one slot (hidden + ids) per stage
    const int n_batch = std::min(g_n_batch, n_ctx);
    const size_t hdr_bytes    = ((sizeof(shm_hdr) + 4095) / 4096) * 4096;
    const size_t hidden_bytes = sizeof(float) * (size_t) n_embd * (size_t) n_batch;
    const size_t token_bytes  = sizeof(int32_t) * (size_t) n_batch;
    const size_t slot_bytes   = hidden_bytes + token_bytes;
    const size_t shm_bytes    = hdr_bytes + (size_t) n_stage * slot_bytes;

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
        c.hdr       = hdr;
        c.hidden    = (float *)   ((char *) base + hdr_bytes + (size_t) i * slot_bytes);
        c.toks      = (int32_t *) ((char *) c.hidden + hidden_bytes);
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

    int sa = 0;
    waitpid(pids[0], &sa, 0);

    // stage 0 is gone: let the rest of the chain unwind instead of blocking in ev_wait
    hdr->cmd = 3;
    ev_post(ev_down[0]);
    for (int i = 1; i < n_stage; ++i) waitpid(pids[i], nullptr, 0);

    fprintf(stderr, "stage 0 exit %d (sig %d), wall %.1f ms\n",
            WEXITSTATUS(sa), WIFSIGNALED(sa) ? WTERMSIG(sa) : 0, now_ms() - t_all);
    return 0;
}
