// 05_token_scatter.metal — K5 `token_scatter`.
//
// Same shape as K3: one 256-thread threadgroup per 1024-word chunk, 4 words
// per thread. Runs in CB2, after the CPU has read header.token_total and
// allocated tok_pos/tok_kind at their exact size.
//
// Per token bit (iterated with ctz64_u2):
//
//   rank = chunk token carry (K4) + in-chunk token-prefix (threadgroup
//          scan + per-thread step) + in-word prefix popcount (bit order)
//   tok_pos[rank]  = byte position (u32)
//   tok_kind[rank] = MJ_TOK_* (mirrors reference::TokenKind discriminants):
//     - quote bits: QuoteOpen when the in-string mask (recomputed exactly
//       as in K3, from bm_quote + the K2 carries) is 1 at the bit —
//       odd parity = this quote starts a string — else QuoteClose;
//     - other bits: from the raw input byte at the position.
//
// Ranks are globally dense and in document order by construction, so the
// scatter writes each output slot exactly once.

#include "common.h"
#include "bitmap_u2.h"
#include "tg_scan.h"

// Clear the lowest set bit of a non-zero uint2 bitmap word.
static inline uint2 mj_clear_lowest(uint2 x) {
    if (x.x != 0u) {
        x.x &= x.x - 1u;
    } else {
        x.y &= x.y - 1u;
    }
    return x;
}

// Bit `b` (0..63) of a uint2 bitmap word.
static inline uint mj_bit(uint2 x, uint b) {
    return b < 32u ? (x.x >> b) & 1u : (x.y >> (b - 32u)) & 1u;
}

static inline uchar mj_token_kind_for_byte(uchar c) {
    switch (c) {
        case uchar('{'): return uchar(MJ_TOK_LBRACE);
        case uchar('}'): return uchar(MJ_TOK_RBRACE);
        case uchar('['): return uchar(MJ_TOK_LBRACKET);
        case uchar(']'): return uchar(MJ_TOK_RBRACKET);
        case uchar(':'): return uchar(MJ_TOK_COLON);
        case uchar(','): return uchar(MJ_TOK_COMMA);
        default: return uchar(MJ_TOK_SCALAR_START);
    }
}

kernel void token_scatter(
    device const uchar* input [[buffer(0)]],
    device const uint2* bm_quote [[buffer(1)]],
    device const uint2* bm_tok [[buffer(2)]],
    device const uint* chunk_quote_carries [[buffer(3)]],
    device const uint* chunk_token_carries [[buffer(4)]],
    device uint* tok_pos [[buffer(5)]],
    device uchar* tok_kind [[buffer(6)]],
    constant MjParams& params [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint parts[9];
    ulong words = params.element_count;
    ulong base = ulong(tgid) * ulong(MJ_CHUNK_WORDS) + ulong(lid) * 4;

    uint2 quote[4];
    uint2 tokens[4];
    uint quote_pc[4];
    uint quote_sum = 0u;
    uint token_sum = 0u;
    for (uint j = 0u; j < 4u; ++j) {
        ulong w = base + ulong(j);
        bool in_range = w < words;
        quote[j] = in_range ? bm_quote[w] : uint2(0u, 0u);
        tokens[j] = in_range ? bm_tok[w] : uint2(0u, 0u);
        quote_pc[j] = popcount64_u2(quote[j]);
        quote_sum += quote_pc[j];
        token_sum += popcount64_u2(tokens[j]);
    }

    uint parity = chunk_quote_carries[tgid]
        + tg_exclusive_scan_256(quote_sum, simd_lane, simd_id, parts);
    uint rank = chunk_token_carries[tgid]
        + tg_exclusive_scan_256(token_sum, simd_lane, simd_id, parts);

    for (uint j = 0u; j < 4u; ++j) {
        uint2 in_string = prefix_xor64_u2(quote[j]);
        if ((parity & 1u) != 0u) {
            in_string = ~in_string;
        }
        ulong word_start = (base + ulong(j)) * ulong(MJ_WORD_BYTES);

        uint2 bits = tokens[j];
        while ((bits.x | bits.y) != 0u) {
            uint b = ctz64_u2(bits);
            bits = mj_clear_lowest(bits);
            ulong pos = word_start + ulong(b);
            uchar kind;
            if (mj_bit(quote[j], b) != 0u) {
                kind = mj_bit(in_string, b) != 0u ? uchar(MJ_TOK_QUOTE_OPEN)
                                                  : uchar(MJ_TOK_QUOTE_CLOSE);
            } else {
                kind = mj_token_kind_for_byte(input[pos]);
            }
            tok_pos[rank] = uint(pos);
            tok_kind[rank] = kind;
            rank += 1u;
        }
        parity += quote_pc[j];
    }
}
