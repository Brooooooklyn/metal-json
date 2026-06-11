// 06_validate.metal — K6 `token_validate_footprint` + K6b `apply_tape_offsets`.
//
// CB2 grows around these two kernels (the M3 structure pipeline):
//
//   K5 token_scatter            (M2) writes tok_pos / tok_kind
//   K6 token_validate_footprint per token: Layer-1 validation + per-chunk
//                               partial counts (tape words, stringbuf bytes,
//                               skeleton / string / scalar records) + one
//                               min-reduced error word per chunk
//   K7 spine3                   (07_spine3.metal) chunk partials -> exclusive
//                               carries, totals + error fold -> header
//   ── commit, wait: CPU sync 2 reads the header. A Layer-1 error REJECTS
//      the input here (CB2b/CB3 never run; list outputs are never
//      produced). Otherwise the CPU exact-allocates tape_ofs target users,
//      the skeleton / string / scalar lists from the K7 totals ──
//   K6b apply_tape_offsets      tape_ofs[token] (+1 root-prologue base) and
//                               the deterministic document-order scatter of
//                               the three lists, via chunk carries + in-chunk
//                               ranks — never global atomics
//
// Geometry: one 256-thread threadgroup per MJ_TOK_CHUNK_TOKENS = 1024-token
// chunk, MJ_TOK_PER_THREAD = 4 tokens per thread (the K3/K5 shape in the
// token domain). Both kernels are dispatched as FULL threadgroups
// (Dispatch::Threadgroups) — the cooperative scans below are convergent.
//
// The bit-exact spec is the CPU oracle `reference::stage3_validate_local`
// (src/reference/validate.rs):
//
//   - the token-order table  -> MJ_PAIR_ALLOWED (mirrors `pair_allowed`)
//   - error kind/offset pick -> mj_check_adjacent (mirrors `check_adjacent`)
//   - the colon 4-token rule -> mj_colon_rule (mirrors `colon_rule`)
//   - object-first-member    -> mj_first_member_rule (mirrors
//                               `first_member_rule`)
//   - literal byte checks    -> mj_check_literal (mirrors
//                               `reference::scalars::check_literal`)
//   - tape footprints        -> mj_token_info (mirrors the per-kind
//                               footprint pushes in `stage3_validate_local`:
//                               brackets 1, string opens 1, literals 1,
//                               numbers 2, colon/comma/quote-close 0)
//   - skeleton records       -> the (token_index, pos, byte) triple of
//                               `SkeletonRecord`, materialized as three
//                               parallel arrays (struct-of-arrays; the
//                               field VALUES are bit-identical)
//
// Error semantics: every rule packs (offset << 32) | MJ_ERR_* and the
// chunk/threadgroup/K7 min-reduction picks the earliest offset, with the
// MjErrorCode numeric order breaking same-offset ties — see the tie-break
// contract in common.h for why that reproduces the reference's iteration
// order exactly. The reduction is deterministic (no atomic races): each
// threadgroup serializes its own min, K7 serializes the chunks.

#include "common.h"
#include "tg_scan.h"

// Virtual boundary "kind" for the start/end of the token stream
// (reference: the `None` arms of `pair_allowed` / `check_adjacent`).
constant constexpr uint MJ_TOK_NONE = 9;

// THE token-order table, one row per `prev` kind (row 9 = virtual start),
// one bit per allowed `next` kind (bit 9 = virtual end). Mirrors
// `reference::validate::pair_allowed` row for row; see that function for
// the JSONTestSuite n_*.json case each ban kills.
//
//   value starters VS = LBrace | LBracket | QuoteOpen | ScalarStart = 0x145
//   value enders   VE = RBrace | RBracket | QuoteClose | ScalarStart = 0x18A
constant uint MJ_PAIR_ALLOWED[10] = {
    /* LBrace      -> */ 0x042, // QuoteOpen | RBrace
    /* RBrace      -> */ 0x22A, // Comma | RBrace | RBracket | end
    /* LBracket    -> */ 0x14D, // VS | RBracket
    /* RBracket    -> */ 0x22A, // Comma | RBrace | RBracket | end
    /* Colon       -> */ 0x145, // VS
    /* Comma       -> */ 0x145, // VS
    /* QuoteOpen   -> */ 0x080, // QuoteClose only
    /* QuoteClose  -> */ 0x23A, // Colon | Comma | RBrace | RBracket | end
    /* ScalarStart -> */ 0x22A, // Comma | RBrace | RBracket | end
    /* (start)     -> */ 0x345, // VS | end (empty stream never reaches K6)
};

constant constexpr uint MJ_VALUE_START_MASK = 0x145;
constant constexpr uint MJ_VALUE_END_MASK = 0x18A;

static inline bool mj_pair_allowed(uint prev_kind, uint next_kind) {
    return ((MJ_PAIR_ALLOWED[prev_kind] >> next_kind) & 1u) != 0u;
}

// Apply the table to one adjacent pair; on a banned pair pick the error
// kind and offset exactly as `reference::validate::check_adjacent` does
// (the arm ORDER there is load-bearing: QuoteOpen first, then open-at-end,
// then the two-values MissingComma case, then the generic fallbacks).
// Returns the packed error, or MJ_HEADER_NO_ERROR.
static inline uint64_t mj_check_adjacent(
    uint prev_kind, ulong prev_pos,
    uint next_kind, ulong next_pos,
    ulong input_len)
{
    if (mj_pair_allowed(prev_kind, next_kind)) {
        return MJ_HEADER_NO_ERROR;
    }
    if (prev_kind == MJ_TOK_QUOTE_OPEN) {
        // Unreachable post-CB1 (odd quote totals are rejected there, and an
        // even total makes every QuoteOpen's next token its QuoteClose);
        // kept so the table stays a complete mirror of the reference.
        return mj_pack_error(prev_pos, MJ_ERR_UNTERMINATED_STRING);
    }
    if (next_kind == MJ_TOK_NONE
        && (prev_kind == MJ_TOK_LBRACE || prev_kind == MJ_TOK_LBRACKET)) {
        // Input ends right after an open bracket: report the bracket.
        return mj_pack_error(prev_pos, MJ_ERR_UNBALANCED);
    }
    if (prev_kind != MJ_TOK_NONE && next_kind != MJ_TOK_NONE
        && ((MJ_VALUE_END_MASK >> prev_kind) & 1u) != 0u
        && ((MJ_VALUE_START_MASK >> next_kind) & 1u) != 0u) {
        // Two values back to back: a comma is missing between them.
        return mj_pack_error(next_pos, MJ_ERR_MISSING_COMMA);
    }
    if (next_kind != MJ_TOK_NONE) {
        return mj_pack_error(next_pos, MJ_ERR_UNEXPECTED_TOKEN);
    }
    return mj_pack_error(input_len, MJ_ERR_UNEXPECTED_TOKEN);
}

// May byte `b` continue a scalar run? (reference::scalars::scalar_run stop
// set: whitespace, operator, `"` — anything else extends the run.)
static inline bool mj_scalar_continues(uchar b) {
    return !(mj_is_ws_byte(b) || mj_is_op_byte(b) || b == uchar('"'));
}

// Literal texts for mj_check_literal, padded to 5 bytes ("true"/"null"
// never read their 5th entry — the length below stops at 4).
constant uchar MJ_LIT_TRUE[5] = { 't', 'r', 'u', 'e', 0 };
constant uchar MJ_LIT_FALSE[5] = { 'f', 'a', 'l', 's', 'e' };
constant uchar MJ_LIT_NULL[5] = { 'n', 'u', 'l', 'l', 0 };

// Byte-exact `true` / `false` / `null` check at `pos`, including the
// boundary rule (the byte after the literal must not extend the scalar
// run — kills `truee`, `false0`, `[tru]`). Mirrors
// `reference::scalars::check_literal`: bytes at or past input_len read as
// padding spaces, which mismatch the literal text / end the run exactly
// like the reference's input.len() bounds checks.
static inline uint64_t mj_check_literal(
    device const uchar* input,
    ulong pos,
    ulong input_len,
    ulong padded_len)
{
    // b0 is one of t/f/n — the caller dispatched on it, so the first byte
    // always matches its own text.
    uchar b0 = input[pos];
    uint len = (b0 == uchar('f')) ? 5u : 4u;
    constant const uchar* text = (b0 == uchar('t')) ? MJ_LIT_TRUE
        : (b0 == uchar('f')) ? MJ_LIT_FALSE
                             : MJ_LIT_NULL;
    bool ok = true;
    for (uint k = 0u; k < len; ++k) {
        ok = ok && (mj_load_padded(input, pos + ulong(k), padded_len) == text[k]);
    }
    // Boundary: end == input_len is fine; a continuing byte is not.
    ulong end = pos + ulong(len);
    ok = ok && (end >= input_len || !mj_scalar_continues(input[end]));
    return ok ? MJ_HEADER_NO_ERROR : mj_pack_error(pos, MJ_ERR_INVALID_LITERAL);
}

// --- per-token footprints + list membership ----------------------------------

// Everything K6 counts and K6b scatters for one token. The two kernels call
// the same function on the same inputs, so the K6 chunk totals and the K6b
// ranks can never disagree (the scatter would overrun its exact-size
// buffers otherwise).
struct MjTokenInfo {
    uint footprint;  // tape words this token emits (the K7 scan component)
    uint slot_bytes; // string-buffer slot: raw_len + 5 for QuoteOpen, else 0
    uint skel;       // 1 if a skeleton record (bracket / colon / comma)
    uint str;        // 1 if a string-list record (QuoteOpen)
    uint scalar;     // 1 if a scalar-list record (ScalarStart)
    uchar skel_byte; // the SkeletonRecord byte: one of { } [ ] : ,
};

static inline MjTokenInfo mj_token_info(
    device const uchar* input,
    device const uint* tok_pos,
    device const uchar* tok_kind,
    ulong t,
    ulong n)
{
    MjTokenInfo info = { 0u, 0u, 0u, 0u, 0u, uchar(0) };
    switch (tok_kind[t]) {
        case MJ_TOK_LBRACE:   info.footprint = 1u; info.skel = 1u; info.skel_byte = uchar('{'); break;
        case MJ_TOK_RBRACE:   info.footprint = 1u; info.skel = 1u; info.skel_byte = uchar('}'); break;
        case MJ_TOK_LBRACKET: info.footprint = 1u; info.skel = 1u; info.skel_byte = uchar('['); break;
        case MJ_TOK_RBRACKET: info.footprint = 1u; info.skel = 1u; info.skel_byte = uchar(']'); break;
        case MJ_TOK_COLON:    info.skel = 1u; info.skel_byte = uchar(':'); break;
        case MJ_TOK_COMMA:    info.skel = 1u; info.skel_byte = uchar(','); break;
        case MJ_TOK_QUOTE_OPEN: {
            info.footprint = 1u;
            info.str = 1u;
            // The close quote is the very next token (stage-2 adjacency;
            // guaranteed post-CB1 by the even quote total). raw_len is the
            // byte count strictly between the quotes; the slot adds the
            // 4-byte length prefix + NUL of the string record
            // (docs/tape-format.md: raw_len + 5). The t+1 guard is
            // defensive only (a lone trailing open cannot reach K6).
            uint raw_len = (t + 1 < n) ? (tok_pos[t + 1] - tok_pos[t] - 1u) : 0u;
            info.slot_bytes = raw_len + 5u;
            break;
        }
        case MJ_TOK_QUOTE_CLOSE:
            break; // 0 tape words; the open quote owns the record
        default: { // MJ_TOK_SCALAR_START
            info.scalar = 1u;
            uchar b = input[tok_pos[t]];
            // Numbers occupy two tape words (marker + value); literals one.
            // A byte that cannot begin a scalar is a Layer-1 error (the
            // validation below reports it); its footprint value is never
            // observed — rejection keeps CB2b from running — but is pinned
            // to 1 so K6 and K6b stay bit-identical on every input.
            info.footprint = (b == uchar('-') || (b >= uchar('0') && b <= uchar('9'))) ? 2u : 1u;
            break;
        }
    }
    return info;
}

// --- Layer-1 validation of one token --------------------------------------------

// Every context-free rule of reference stage 3, evaluated for token `t`:
//   1. the token-order table on (t-1, t) — and on (t, end) for the last
//      token (the virtual end-of-input boundary);
//   2. the object-first-member rule when t is `{`;
//   3. the colon 4-token rule when t is `:`;
//   4. scalar first-byte / literal byte checks when t is a ScalarStart.
// Returns the min (earliest offset, lowest code) of everything that fired.
static inline uint64_t mj_validate_token(
    device const uchar* input,
    device const uint* tok_pos,
    device const uchar* tok_kind,
    ulong t,
    ulong n,
    ulong input_len,
    ulong padded_len)
{
    uint64_t err = MJ_HEADER_NO_ERROR;
    uint kind = uint(tok_kind[t]);
    ulong pos = ulong(tok_pos[t]);

    uint prev_kind = (t > 0) ? uint(tok_kind[t - 1]) : MJ_TOK_NONE;
    ulong prev_pos = (t > 0) ? ulong(tok_pos[t - 1]) : 0;
    err = min(err, mj_check_adjacent(prev_kind, prev_pos, kind, pos, input_len));
    if (t + 1 == n) {
        err = min(err, mj_check_adjacent(kind, pos, MJ_TOK_NONE, 0, input_len));
    }

    if (kind == MJ_TOK_LBRACE) {
        // Object-first-member rule: `{` followed by a complete key pair
        // requires `:` as token t+3. Mirrors `first_member_rule` (incomplete
        // key pairs are the table's job: `{"abc` must stay an
        // unterminated-string report, not a missing colon).
        bool key_pair = (t + 1 < n) && (uint(tok_kind[t + 1]) == MJ_TOK_QUOTE_OPEN)
            && (t + 2 < n) && (uint(tok_kind[t + 2]) == MJ_TOK_QUOTE_CLOSE);
        if (key_pair) {
            if (t + 3 < n) {
                if (uint(tok_kind[t + 3]) != MJ_TOK_COLON) {
                    err = min(err, mj_pack_error(ulong(tok_pos[t + 3]), MJ_ERR_MISSING_COLON));
                }
            } else {
                err = min(err, mj_pack_error(input_len, MJ_ERR_MISSING_COLON));
            }
        }
    } else if (kind == MJ_TOK_COLON) {
        // Colon 4-token rule: tokens (t-3..t) must be ({ or , then a
        // complete key string then :). Only t-3 needs checking here — the
        // table already pins t-1/t-2. Mirrors `colon_rule`.
        bool ok = (t >= 3)
            && (uint(tok_kind[t - 3]) == MJ_TOK_LBRACE || uint(tok_kind[t - 3]) == MJ_TOK_COMMA);
        if (!ok) {
            err = min(err, mj_pack_error(pos, MJ_ERR_UNEXPECTED_TOKEN));
        }
    } else if (kind == MJ_TOK_SCALAR_START) {
        uchar b = input[pos];
        if (b == uchar('-') || (b >= uchar('0') && b <= uchar('9'))) {
            // Number grammar is K10's job (M4); Layer 1 accepts the run.
        } else if (b == uchar('t') || b == uchar('f') || b == uchar('n')) {
            err = min(err, mj_check_literal(input, pos, input_len, padded_len));
        } else {
            // A byte that cannot begin a JSON scalar (`[True]`, `{'a':0}`,
            // `[*]`, a BOM, ...).
            err = min(err, mj_pack_error(pos, MJ_ERR_UNEXPECTED_TOKEN));
        }
    }
    return err;
}

// --- K6 ---------------------------------------------------------------------------

// One threadgroup per 1024-token chunk, 4 tokens per thread. Validates every
// token and reduces the chunk's
//   uint4 (tape words, skeleton, string, scalar) counts -> chunk_counts,
//   ulong string-slot byte sum                          -> chunk_string_bytes,
//   ulong packed-error min                              -> chunk_error,
// all single-writer plain stores (thread 0), no device atomics. K7 scans /
// folds these. The three chunk buffers are fully overwritten for every
// chunk in the grid, so they carry no zero/init precondition.
kernel void token_validate_footprint(
    device const uchar* input [[buffer(0)]],
    device const uint* tok_pos [[buffer(1)]],
    device const uchar* tok_kind [[buffer(2)]],
    device uint4* chunk_counts [[buffer(3)]],
    device ulong* chunk_string_bytes [[buffer(4)]],
    device ulong* chunk_error [[buffer(5)]],
    constant MjParams& params [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint4 parts4[9];
    threadgroup ulong lanes[THREADGROUP_SIZE];

    ulong n = params.element_count; // token_total
    ulong input_len = params.input_len;
    ulong padded_len = ((input_len + ulong(MJ_WORD_BYTES) - 1) / ulong(MJ_WORD_BYTES))
        * ulong(MJ_WORD_BYTES);
    ulong base = ulong(tgid) * ulong(MJ_TOK_CHUNK_TOKENS) + ulong(lid) * ulong(MJ_TOK_PER_THREAD);

    uint4 counts = uint4(0u);
    ulong slot_sum = 0;
    uint64_t err = MJ_HEADER_NO_ERROR;
    for (uint j = 0u; j < MJ_TOK_PER_THREAD; ++j) {
        ulong t = base + ulong(j);
        if (t < n) {
            MjTokenInfo info = mj_token_info(input, tok_pos, tok_kind, t, n);
            counts += uint4(info.footprint, info.skel, info.str, info.scalar);
            slot_sum += ulong(info.slot_bytes);
            err = min(err, mj_validate_token(input, tok_pos, tok_kind, t, n,
                                             input_len, padded_len));
        }
    }

    // uint4 totals via the scan's grand-total slot (every thread converges).
    tg_exclusive_scan4_256(counts, simd_lane, simd_id, parts4);

    // ulong string-byte sum: per-chunk partials can exceed u32 (one giant
    // string literal), so reduce in 64-bit via threadgroup memory + a
    // 256-step serial fold on thread 0 (cheap; the spine pattern's cost
    // model). Reuse `lanes` afterwards for the error min.
    lanes[lid] = slot_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        ulong total = 0;
        for (uint i = 0u; i < THREADGROUP_SIZE; ++i) {
            total += lanes[i];
        }
        chunk_string_bytes[tgid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    lanes[lid] = err;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        uint64_t chunk_min = MJ_HEADER_NO_ERROR;
        for (uint i = 0u; i < THREADGROUP_SIZE; ++i) {
            chunk_min = min(chunk_min, lanes[i]);
        }
        chunk_error[tgid] = chunk_min;
        chunk_counts[tgid] = parts4[8];
    }
}

// --- K6b --------------------------------------------------------------------------

// Runs in its own command buffer AFTER the CPU sync 2 (the list buffers
// cannot exist until the CPU reads the K7 totals — the exact-size
// allocation rule, same shape as K5 after CB1). Recomputes each token's
// MjTokenInfo, rebuilds the in-chunk exclusive prefix with the same
// cooperative scan, adds the K7 chunk carries, and materializes:
//
//   tape_ofs[t]   = 1 + (exclusive prefix sum of footprints)[t]
//                   The +1 is the root prologue word: reference emit_tape
//                   seeds its running tape position at 1 because tape[0] is
//                   the root word (`make_root`). tape_ofs is defined for
//                   EVERY token (footprint-0 tokens share their successor's
//                   offset), exactly like the reference's tape_pos vector.
//   skeleton      (token_index, pos, byte) at rank = skeleton prefix
//   string list   QuoteOpen token index    at rank = string prefix
//   scalar list   ScalarStart token index  at rank = scalar prefix
//
// Ranks are dense, in document order, and bounded by the K7 totals the CPU
// allocated from (K6 and K6b derive them from the same mj_token_info), so
// every output slot is written exactly once — deterministic document order
// with no global atomics, mirroring how the reference appends in its token
// loop.
kernel void apply_tape_offsets(
    device const uchar* input [[buffer(0)]],
    device const uint* tok_pos [[buffer(1)]],
    device const uchar* tok_kind [[buffer(2)]],
    device const uint4* chunk_counts [[buffer(3)]], // K7 exclusive carries
    device uint* tape_ofs [[buffer(4)]],
    device uint* skel_token_index [[buffer(5)]],
    device uint* skel_pos [[buffer(6)]],
    device uchar* skel_byte [[buffer(7)]],
    device uint* string_tokens [[buffer(8)]],
    device uint* scalar_tokens [[buffer(9)]],
    constant MjParams& params [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint4 parts4[9];

    ulong n = params.element_count; // token_total
    ulong base = ulong(tgid) * ulong(MJ_TOK_CHUNK_TOKENS) + ulong(lid) * ulong(MJ_TOK_PER_THREAD);

    MjTokenInfo info[MJ_TOK_PER_THREAD];
    uint4 sums = uint4(0u);
    for (uint j = 0u; j < MJ_TOK_PER_THREAD; ++j) {
        ulong t = base + ulong(j);
        if (t < n) {
            info[j] = mj_token_info(input, tok_pos, tok_kind, t, n);
            sums += uint4(info[j].footprint, info[j].skel, info[j].str, info[j].scalar);
        } else {
            info[j] = MjTokenInfo{ 0u, 0u, 0u, 0u, 0u, uchar(0) };
        }
    }

    // (tape words, skeleton rank, string rank, scalar rank) entering this
    // thread's first token: K7 chunk carry + in-chunk cooperative prefix.
    uint4 prefix = chunk_counts[tgid]
        + tg_exclusive_scan4_256(sums, simd_lane, simd_id, parts4);

    for (uint j = 0u; j < MJ_TOK_PER_THREAD; ++j) {
        ulong t = base + ulong(j);
        if (t < n) {
            tape_ofs[t] = 1u + prefix.x; // +1: the root prologue word
            if (info[j].skel != 0u) {
                skel_token_index[prefix.y] = uint(t);
                skel_pos[prefix.y] = tok_pos[t];
                skel_byte[prefix.y] = info[j].skel_byte;
            }
            if (info[j].str != 0u) {
                string_tokens[prefix.z] = uint(t);
            }
            if (info[j].scalar != 0u) {
                scalar_tokens[prefix.w] = uint(t);
            }
            prefix += uint4(info[j].footprint, info[j].skel, info[j].str, info[j].scalar);
        }
    }
}
