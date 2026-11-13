// stage_abi.h -- the contract between pipeline stages.
//
// A stage is "whatever turns hidden_in into hidden_out". It does NOT have to be llama.cpp: the
// driver only knows this header. Anything that exports split_stage_get() can take a position in
// the chain (llama-split --ext-after N:libstage.so).
//
// Design notes (see docs/60-llama-split/architecture.md 5.3):
//   * one stage == one process, so the vtable is a singleton -- no `self` pointer to thread through
//   * `dtype` is in the ABI on purpose: f16 transport, int8 hidden states or a block-sparse format
//     must be a one-field change, not a protocol change
//   * only the driver's head stage may sample; a stage in the middle forwards `sample` untouched
//   * `run()` is called once per chunk; for decode a chunk is exactly one token
#ifndef SPLIT_STAGE_ABI_H
#define SPLIT_STAGE_ABI_H

#include <stdint.h>

#define SPLIT_STAGE_ABI_VERSION 1

enum split_dtype {
    SPLIT_DTYPE_F32 = 0,
    SPLIT_DTYPE_F16 = 1,   // reserved: the driver still ships f32 on the wire
};

typedef struct split_hidden {
    void * data;           // n_tok rows x n_embd elements of `dtype`
    int    dtype;          // enum split_dtype
    int    n_embd;
    int    n_tok;
} split_hidden;

typedef struct split_slot_meta {
    int32_t n_tok;
    int32_t pos0;          // absolute position of the first token in this chunk
    int32_t n_embd;
    int32_t sample;        // 1 = the last stage must emit a token; middle stages forward it
    int32_t rid;           // request id, so a stage can reset per-request state
    int32_t chunk_seq;     // monotonic per request; a gap means the driver desynchronised
    const int32_t * ids;   // token ids of this chunk, may be NULL
} split_slot_meta;

typedef struct split_stage {
    int          abi_version;   // must equal SPLIT_STAGE_ABI_VERSION
    const char * name;

    // cfg is the string given after ':' on the command line (may be empty). Return 0 on success.
    int  (*open )(const char * cfg, int n_embd, int n_ctx, int n_slot);

    // in and out may alias. Return 0 on success, negative to abort the request.
    // Only a stage that owns the vocabulary may write *out_token.
    int  (*run  )(const split_slot_meta * m, const split_hidden * in,
                  split_hidden * out, int32_t * out_token);

    void (*close)(void);
} split_stage;

// every stage library exports exactly this
const split_stage * split_stage_get(void);

#endif // SPLIT_STAGE_ABI_H
