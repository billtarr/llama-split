// stage_pass.c -- reference implementation of the stage ABI, and the chain's test double.
//
// It links nothing from llama.cpp: `ldd stage_pass.so` shows only libc/libm. That is the point --
// if the chain runs with this in the middle, the seam is no longer llama.cpp-specific.
//
//   STAGE_PASS_BIAS=0.0   (default)  identity: output must be bit-identical to the chain without it
//   STAGE_PASS_BIAS=1.0              adds a bias to dim 0 of every row: proves the data really goes
//                                    through. NOTE: a *uniform scale* is a no-op here -- every block
//                                    starts with an RMSNorm, which cancels it exactly. measured.
//   STAGE_PASS_VERBOSE=1            one line per chunk
//
// It also asserts the ABI invariants it can see: dtype must be f32, n_embd must match what the
// driver announced in open(), and the chunk must not be empty.
#include "stage_abi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int    g_n_embd = 0;
static int    g_n_ctx  = 0;
static int    g_n_slot = 0;
static float  g_bias   = 0.0f;
static int    g_verbose = 0;
static long   g_chunks = 0;
static double g_sum    = 0.0;

static int st_open(const char * cfg, int n_embd, int n_ctx, int n_slot) {
    g_n_embd = n_embd;
    g_n_ctx  = n_ctx;
    g_n_slot = n_slot;

    const char * bias = getenv("STAGE_PASS_BIAS");
    if (bias && *bias) g_bias = (float) atof(bias);
    const char * v = getenv("STAGE_PASS_VERBOSE");
    if (v && *v) g_verbose = 1;

    if (g_verbose) {
        fprintf(stderr, "[X] pass stage: n_embd %d, n_ctx %d, %d slot(s), bias %.3f%s\n",
                n_embd, n_ctx, n_slot, g_bias, cfg && *cfg ? cfg : "");
    }
    return 0;
}

static int st_run(const split_slot_meta * m, const split_hidden * in,
                  split_hidden * out, int32_t * out_token) {
    (void) out_token;   // a middle stage never samples

    if (in->dtype != SPLIT_DTYPE_F32) return -2;          // f16 is not on the wire yet
    if (in->n_embd != g_n_embd)       return -3;
    if (in->n_tok <= 0)               return -4;
    if (!in->data || !out->data)      return -5;

    const size_t n = (size_t) in->n_embd * (size_t) in->n_tok;
    const float * src = (const float *) in->data;
    float * dst = (float *) out->data;

    if (dst != src) memmove(dst, src, n * sizeof(float));
    if (g_bias != 0.0f) {
        // one element per row: a uniform scale would be cancelled by the next RMSNorm
        for (int r = 0; r < in->n_tok; ++r) dst[(size_t) r * in->n_embd] += g_bias;
    }

    out->dtype  = in->dtype;
    out->n_embd = in->n_embd;
    out->n_tok  = in->n_tok;

    g_chunks++;
    if (g_verbose) {
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += dst[i];
        g_sum = s;
        fprintf(stderr, "[X] chunk %ld rid %d seq %d: %d x %d f32, sum %.6f\n",
                g_chunks, m->rid, m->chunk_seq, in->n_tok, in->n_embd, s);
    }
    return 0;
}

static void st_close(void) {
    if (g_verbose) fprintf(stderr, "[X] pass stage: %ld chunks, last sum %.6f\n", g_chunks, g_sum);
}

static const split_stage g_stage = {
    SPLIT_STAGE_ABI_VERSION,
    "pass",
    st_open,
    st_run,
    st_close,
};

const split_stage * split_stage_get(void) { return &g_stage; }
