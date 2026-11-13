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
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { logf(tag, "ctx init failed\n"); return 1; }
    logf(tag, "ctx init %.0f ms (total %.0f ms)\n", now_ms() - t0_ctx, now_ms() - t0_all);

    if (!pc.is_last) llama_set_embeddings(ctx, true);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd(model);
    if (n_embd != pc.n_embd) { logf(tag, "n_embd mismatch\n"); return 1; }

    logf(tag, "ready (ctx %d, batch %d%s)\n", n_ctx, pc.n_batch, pc.is_last ? ", last" : "");

    llama_sampler * smpl = nullptr;
    std::vector<llama_token> toks(pc.n_batch > 0 ? (size_t) pc.n_batch : 1);

    for (;;) {
        ev_wait(pc.ev_in);
        const int32_t cmd = pc.hdr->cmd;

        if (cmd == 3) { ev_post(pc.ev_out); break; }

        if (cmd == 2) {
            llama_memory_clear(llama_get_memory(ctx), true);
            if (smpl) { llama_sampler_free(smpl); smpl = nullptr; }
            ev_post(pc.ev_out);
            continue;
        }
        if (cmd != 1) { ev_post(pc.ev_out); continue; }

        const int n_tok     = pc.hdr->n_tok;
        const int bpos      = pc.hdr->pos;
        const int do_sample = (pc.hdr->sample != 0) && pc.is_last;

        llama_batch b = llama_batch_init(n_tok, n_embd, 1);
        memcpy(b.embd, pc.in_hidden, sizeof(float) * (size_t) n_embd * n_tok);
        for (int i = 0; i < n_tok; ++i) {
            toks[i] = g_send_tokens ? ((g_send_tokens == 2) ? 0 : pc.in_toks[i]) : 0;
            b.pos[i] = bpos + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = (do_sample && i == n_tok - 1) ? 1 : 0;
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
            else            { ev_post(pc.ev_out); }
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
                const llama_token t = llama_sampler_sample(smpl, ctx, -1);
                pc.hdr->next_token = (int32_t) t;
                pc.hdr->done = (t == llama_vocab_eos(vocab)) ? 1 : 0;
                ev_post(pc.ev_tok);
            }
            // no sampling (mid-prefill chunk): stage 0 is not waiting on anything
        } else {
            const float * e = llama_get_embeddings(ctx);
            if (!e) { logf(tag, "no embeddings\n"); pc.hdr->done = 1; ev_post(pc.ev_out); continue; }
            if (pc.pushed) ev_wait(pc.ev_free);
            memcpy(pc.hidden, e, sizeof(float) * (size_t) n_embd * n_tok);
            for (int i = 0; i < n_tok; ++i) pc.toks[i] = toks[i];
            pc.pushed = 1;
            ev_post(pc.ev_out);
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
