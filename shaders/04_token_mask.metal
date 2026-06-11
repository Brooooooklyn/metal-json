// 04_token_mask.metal — K3 `token_mask`.
//
// One 256-thread threadgroup per 1024-word spine chunk, 4 words (256 input
// bytes) per thread — the threadgroup must span the whole chunk so the
// in-chunk quote-parity prefix is one cooperative scan instead of a
// per-word lookback (and 256 B/thread comfortably clears the spike-C
// >= 64 B grain floor).
//
// Per word (bit-exact to reference::stage2_tokens):
//
//   in_string  = prefix_xor64(quote_real)            (6-step ladder in uint2:
//                ^ parity carried into the word        five vector shifts +
//                                                      one seam broadcast)
//   tokens     = (candidates & ~in_string & ~quote_real) | quote_real
//
// where the carried parity = K2's chunk quote carry + the in-chunk prefix
// of quote popcounts (threadgroup scan + per-thread sequential step).
//
// `bm_tok` is read as K1's candidate bitmap and overwritten IN PLACE with
// the token bitmap (the planned bm_cand/bm_tok aliasing; same-thread
// read-then-write, no cross-thread hazard). The per-chunk token popcount is
// reduced with a second threadgroup scan and stored (single writer per
// chunk) for the K4 spine scan.

#include "common.h"
#include "bitmap_u2.h"
#include "tg_scan.h"

kernel void token_mask(
    device const uint2* bm_quote [[buffer(0)]],
    device uint2* bm_tok [[buffer(1)]],
    device const uint* chunk_quote_carries [[buffer(2)]],
    device uint* chunk_token_counts [[buffer(3)]],
    constant MjParams& params [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint parts[9];
    ulong words = params.element_count;
    ulong base = ulong(tgid) * ulong(MJ_CHUNK_WORDS) + ulong(lid) * 4;

    uint2 quote[4];
    uint quote_pc[4];
    uint quote_sum = 0u;
    for (uint j = 0u; j < 4u; ++j) {
        ulong w = base + ulong(j);
        quote[j] = w < words ? bm_quote[w] : uint2(0u, 0u);
        quote_pc[j] = popcount64_u2(quote[j]);
        quote_sum += quote_pc[j];
    }

    // Quote parity entering this thread's first word.
    uint parity = chunk_quote_carries[tgid]
        + tg_exclusive_scan_256(quote_sum, simd_lane, simd_id, parts);

    uint token_sum = 0u;
    for (uint j = 0u; j < 4u; ++j) {
        ulong w = base + ulong(j);
        if (w < words) {
            uint2 candidates = bm_tok[w];
            uint2 in_string = prefix_xor64_u2(quote[j]);
            if ((parity & 1u) != 0u) {
                in_string = ~in_string;
            }
            uint2 tokens = (candidates & ~in_string & ~quote[j]) | quote[j];
            bm_tok[w] = tokens;
            token_sum += popcount64_u2(tokens);
        }
        parity += quote_pc[j];
    }

    // Chunk token partial for K4 (the scan's grand total; single writer).
    tg_exclusive_scan_256(token_sum, simd_lane, simd_id, parts);
    if (lid == 0u) {
        chunk_token_counts[tgid] = parts[8];
    }
}
