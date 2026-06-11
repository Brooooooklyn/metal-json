// common.h — shared declarations for all metal-json kernels.
//
// This header is consumed two ways:
//   1. AOT: `#include`d normally by the Metal compiler in build.rs.
//   2. runtime-shaders: textually inlined by the tiny preprocessor in
//      src/metal/context.rs (newLibraryWithSource has no include paths).
// Keep it self-contained and free of nested includes other than
// <metal_stdlib>, which the preprocessor leaves alone.

#ifndef METAL_JSON_COMMON_H
#define METAL_JSON_COMMON_H

#include <metal_stdlib>
using namespace metal;

// Threads per threadgroup for 1-D dispatches. Multiple of the Apple GPU
// SIMD width (32); revisit per-kernel during M5 tuning.
constant constexpr uint THREADGROUP_SIZE = 256;

// Kernel launch parameters, bound as a single constant buffer.
// Mirrors `MjParams` in src/metal/mod.rs — keep the layouts in sync.
struct MjParams {
    uint64_t input_len;   // total input bytes
    uint64_t element_count; // elements processed by this dispatch
    uint64_t reserved0;   // CB3 kernels: the max_depth limit
    uint64_t reserved1;   // K8 sort passes: digit shift | (identity-order flag << 8)
};

// Error codes, packed as (byte_offset << 32) | code into a u64 error word;
// the earliest byte offset wins the min reduction and the NUMERIC CODE
// ORDER breaks same-offset ties. Mirrors the constants in src/gpu/stage1.rs
// and src/gpu/stage2.rs — keep in sync (a Rust test pins them).
//
// THE SAME-OFFSET TIE-BREAK CONTRACT (codes 16..): the reference oracle
// (`reference::stage3_validate_local`) reports the first violation in token
// ITERATION order — at each token: adjacency first, then the kind rules —
// while K6 evaluates every rule in parallel and min-reduces. Whenever two
// stage-3 rules can fire at the same byte offset, the code values below are
// ordered so the min picks exactly what the reference reports:
//
//   MISSING_COLON < MISSING_COMMA   `{"a" "b"}`: the object-first-member
//       rule fires at the `{` (3 tokens before the offending token) and
//       must beat the adjacency MissingComma at the same offset.
//   MISSING_COMMA < UNEXPECTED_TOKEN   `1 *`: the adjacency check precedes
//       the scalar-first-byte kind rule within one reference iteration.
//   UNEXPECTED_TOKEN < INVALID_LITERAL   `{truth}`-style: adjacency
//       precedes the literal kind rule within one iteration.
//   MJ_ERR_UNBALANCED greatest   `{"a"{`-style: the virtual end-of-input
//       check runs after every per-token iteration in the reference, so
//       every same-offset per-token rule must beat it.
//
// UNTERMINATED_STRING never co-fires on the GPU (CB1 already rejected odd
// quote counts) and EMPTY_INPUT is a CPU-side verdict (token_total == 0) —
// both sit after the contested codes. Codes 1-6 cannot collide with 16+:
// a CB's detected error short-circuits later CBs (rejection contract).
//
// THE CB3 (stage 4) TIE-BREAK: within CB3 the competing codes are
// DEPTH_LIMIT(3), TRAILING_CONTENT(4), MISSING_COLON(16),
// UNEXPECTED_TOKEN(18) and UNBALANCED(20). Skeleton elements have distinct
// byte offsets and each element class produces at most one candidate per
// rule family, so the only same-offset pair is an OPEN bracket flagged by
// both the depth scan (DepthLimit at its pos) and the group walk
// (two-opens / leftover-open Unbalanced at the same pos). The reference
// (`reference::stage4_structure`) records phase-1 depth-scan candidates
// before any group-walk candidate, so DEPTH_LIMIT must beat UNBALANCED on
// a tie: 3 < 20 holds. CB3 codes never compete with CB2 codes (rejection
// contract), so 3/4 sitting below the Layer-1 block is harmless.
enum MjErrorCode : uint {
    MJ_OK = 0,
    MJ_ERR_UTF8 = 1,             // K1 (stage 1)
    MJ_ERR_SYNTAX = 2,           // reserved (legacy placeholder, unused)
    MJ_ERR_DEPTH_LIMIT = 3,      // CB3 (M3, depth scan)
    MJ_ERR_TRAILING_CONTENT = 4, // CB3 (M3, depth-0 separators)
    MJ_ERR_NUMBER = 5,           // K10 (M4)
    MJ_ERR_STRING = 6,           // K2 (stage 1, odd quote total)
    // K6 Layer-1 codes (CB2). Each mirrors a SyntaxErrorKind variant; the
    // numeric order IS the same-offset tie-break contract above.
    MJ_ERR_MISSING_COLON = 16,       // SyntaxErrorKind::MissingColon
    MJ_ERR_MISSING_COMMA = 17,       // SyntaxErrorKind::MissingComma
    MJ_ERR_UNEXPECTED_TOKEN = 18,    // SyntaxErrorKind::UnexpectedToken
    MJ_ERR_INVALID_LITERAL = 19,     // SyntaxErrorKind::InvalidLiteral
    MJ_ERR_UNBALANCED = 20,          // SyntaxErrorKind::UnbalancedBrackets
    MJ_ERR_UNTERMINATED_STRING = 21, // SyntaxErrorKind::UnterminatedString
    MJ_ERR_EMPTY_INPUT = 22,         // SyntaxErrorKind::EmptyInput (CPU-side)
};

// --- M2 pipeline geometry -----------------------------------------------------
// Mirrors the constants in src/stage.rs — keep in sync.

// Input bytes per bitmap word and per K1/K3/K5 thread. 64 B/thread is the
// spike-C grain floor (giant-grid scheduling costs ~2.4 ns/threadgroup, so
// thread-per-byte designs are out) and one 64-bit bitmap word per thread.
constant constexpr uint MJ_WORD_BYTES = 64;

// Bitmap words per spine chunk: 1024 words = 64 KiB of input per chunk.
// K1/K3 emit one partial (quote / token popcount) per chunk; the K2/K4
// spine scans run as a single threadgroup over those partials.
constant constexpr uint MJ_CHUNK_WORDS = 1024;

// K1 escape-carry valve: a thread resolving the backslash-run parity at its
// word's left edge peeks backward at the raw input for at most this many
// bytes. A backslash run that is still going at the cap (adversarial
// "backslash wall" input) bumps MjHeader.carry_overflow_count instead, and
// a tiny sequential fix-up pass (GPU kernel or CPU fallback over the
// flagged words) corrects those words before K3 consumes the quote bitmap.
constant constexpr uint MJ_ESCAPE_LOOKBACK_CAP = 4096;

// --- M3 token-domain geometry (CB2: K6 / K7 / K6b) ------------------------------
// Mirrors `TOKEN_CHUNK_TOKENS` in src/stage.rs — keep in sync.

// Tokens per CB2 spine chunk: one 256-thread threadgroup covers one chunk at
// 4 tokens/thread (the K3/K5 shape transplanted to the token domain). K6
// emits one partial record per chunk; the K7 spine scan runs as a single
// threadgroup over those partials; K6b consumes the scanned carries.
constant constexpr uint MJ_TOK_CHUNK_TOKENS = 1024;

// Tokens per K6/K6b thread (MJ_TOK_CHUNK_TOKENS / THREADGROUP_SIZE).
constant constexpr uint MJ_TOK_PER_THREAD = 4;

// --- M3 skeleton-domain geometry (CB3: depth scan / K8 sort / K9 pair+ctx) -----
// Mirrors `SKELETON_CHUNK_ELEMS` in src/stage.rs — keep in sync.

// Skeleton elements per CB3 spine chunk: one 256-thread threadgroup covers
// one chunk at 4 elements/thread (the K6 shape in the skeleton domain).
constant constexpr uint MJ_SKEL_CHUNK_ELEMS = 1024;

// Skeleton elements per CB3 thread (MJ_SKEL_CHUNK_ELEMS / THREADGROUP_SIZE).
constant constexpr uint MJ_SKEL_PER_THREAD = 4;

// Counting-sort digit width: 5 bits = 32 buckets per pass. The Rust side
// (`stage::sort_passes`) picks the pass count from the max_depth limit so
// the passes cover every clean sort key (see mj_sort_key).
constant constexpr uint MJ_SORT_RADIX_BITS = 5;
constant constexpr uint MJ_SORT_BUCKETS = 32;

// --- shared byte classifiers ----------------------------------------------------
// Used by K1's carry look-back (02_classify.metal) and K6's literal checks
// (06_validate.metal). Defined here because the runtime-shaders path
// compiles every .metal unit as ONE translation unit (duplicate static
// definitions across units would collide).

// JSON insignificant whitespace: space, tab, \n, \r.
static inline bool mj_is_ws_byte(uchar b) {
    return b == uchar(0x20) || b == uchar(0x09) || b == uchar(0x0A) || b == uchar(0x0D);
}

// Structural operators: { } [ ] : ,
static inline bool mj_is_op_byte(uchar b) {
    return b == uchar('{') || b == uchar('}') || b == uchar('[') || b == uchar(']')
        || b == uchar(':') || b == uchar(',');
}

// Open bracket of either container type.
static inline bool mj_is_open_byte(uchar b) {
    return b == uchar('{') || b == uchar('[');
}

// Close bracket of either container type.
static inline bool mj_is_close_byte(uchar b) {
    return b == uchar('}') || b == uchar(']');
}

// Depth weight of a skeleton byte: opens +1, closes -1, separators 0.
// The CB3 depth scan prefix-sums these (reference stage4 phase 1).
static inline int mj_skel_weight(uchar b) {
    return mj_is_open_byte(b) ? 1 : (mj_is_close_byte(b) ? -1 : 0);
}

// Largest sort key the CB3 counting sort must keep distinct: max_depth - 1
// (clean inputs only carry depths 1..=max_depth; see mj_sort_key).
static inline uint mj_key_max(uint64_t max_depth) {
    uint64_t limit = max_depth == 0 ? 1 : max_depth;
    return uint(limit - 1);
}

// CB3 counting-sort key of a skeleton element's depth.
//
// Clean inputs (the only ones whose CB3 outputs are ever read — the
// rejection contract discards everything else) have depths in
// 1..=max_depth: separators at depth 0 are TrailingContent errors and
// depths above max_depth are DepthLimit errors. So `depth - 1` is an
// order-isomorphic key in 0..=key_max, one 5-bit digit narrower than raw
// depth at the 1024 default (keys 0..=1023 fit 2 passes; raw depth 1024
// would need a 3rd). On error inputs the clamps below only keep the keys
// in sortable range so K8/K9 stay memory-safe:
//   - depth 0 (depth-0 separators, parked underflow closes) maps to key 0,
//     merging into the depth-1 group. Harmless: a depth-0 separator's
//     latest preceding bracket in the merged group is never an open (depth
//     0 means every depth-1 container has closed), so K9 still sees "no
//     enclosing opener" and reports the reference's TrailingContent; and
//     parked closes only exist after an underflow already reported at an
//     offset <= theirs.
//   - depths above max_depth clamp to key_max, merging into the deepest
//     group. Harmless: such an element only exists after a DepthLimit
//     error at an offset <= its own.
static inline uint mj_sort_key(uint depth, uint key_max) {
    uint k = depth == 0u ? 0u : depth - 1u;
    return min(k, key_max);
}

// Byte at `i`, or a space for reads past the padded buffer. The input
// buffer is space-padded to a whole number of 64-byte words
// (Stage1Buffers contract), so bytes in input_len..padded_len are already
// spaces; this guard covers probes beyond even the padding (K1 UTF-8
// continuation probes, K6 literal tails near EOF). Space is whitespace
// class, so truncated literals/sequences fail exactly like the reference.
static inline uchar mj_load_padded(device const uchar* input, ulong i, ulong padded_len) {
    return i < padded_len ? input[i] : uchar(0x20);
}

// --- K1 per-word escape-carry record (escape_info buffer, one uchar/word) ------
//
// K1 stores the carries it *used* for each word plus valve flags; the
// escape_carry_fixup kernel reads them to resolve capped look-backs.
// Mirrors the bit assignments documented on `Stage1Buffers::escape_info`
// in src/stage.rs — keep in sync.

// Bit 0: the prev_escaped carry K1 used (is the word's first byte escaped?).
constant constexpr uint MJ_CARRY_PREV_ESCAPED = 1u << 0;
// Bit 1: the prev_allows_start carry K1 used (may a scalar run start at the
// word's first byte?).
constant constexpr uint MJ_CARRY_PREV_ALLOWS = 1u << 1;
// Bit 2: the backslash run ending right before the word hit the look-back
// cap, so bit 0 is a guess (assumed even run => not escaped).
constant constexpr uint MJ_CARRY_FLAG_ESCAPE = 1u << 2;
// Bit 3: the word starts right after a `"` whose own backslash look-back hit
// the cap, so bit 1 is a guess (assumed even run => real quote => allows).
constant constexpr uint MJ_CARRY_FLAG_QUOTE = 1u << 3;

// --- Token kinds ----------------------------------------------------------------
// Written by K5 into tok_kind as uchar. Mirrors the declaration order of
// `reference::TokenKind` in src/reference/tokens.rs exactly (a test pins the
// discriminants) — keep in sync.
constant constexpr uint MJ_TOK_LBRACE = 0;       // {
constant constexpr uint MJ_TOK_RBRACE = 1;       // }
constant constexpr uint MJ_TOK_LBRACKET = 2;     // [
constant constexpr uint MJ_TOK_RBRACKET = 3;     // ]
constant constexpr uint MJ_TOK_COLON = 4;        // :
constant constexpr uint MJ_TOK_COMMA = 5;        // ,
constant constexpr uint MJ_TOK_QUOTE_OPEN = 6;   // " opening a string
constant constexpr uint MJ_TOK_QUOTE_CLOSE = 7;  // " closing a string
constant constexpr uint MJ_TOK_SCALAR_START = 8; // first byte of a scalar run

// --- Per-parse result header ---------------------------------------------------

// "No error" sentinel for MjHeader.error: all ones, so any real packed
// error value wins the atomic_min reduction.
constant constexpr uint64_t MJ_HEADER_NO_ERROR = ~0ul;

// Pack an error for the atomic_min reduction: byte offset in the high half
// (earliest offset wins), MjErrorCode in the low half (breaks ties).
static inline uint64_t mj_pack_error(uint64_t byte_offset, MjErrorCode code) {
    return (byte_offset << 32) | uint64_t(code);
}

// Per-parse result header (one 128-byte cell per parse).
// Mirrors `MjHeader` in src/metal/mod.rs — keep the layouts in sync
// (sixteen uint64_t fields, no padding; a layout test pins it).
//
// The CPU initializes `error` to MJ_HEADER_NO_ERROR and decodes
// offset = error >> 32, code = low 32 bits. Error reduction is staged (see
// MjHeaderDev below): within each stage the kernels min-reduce their
// candidates deterministically and a single writer folds the winner in.
struct MjHeader {
    uint64_t error;                // packed (offset << 32) | code
    uint64_t quote_total;          // K2: total real quotes (odd => unterminated string)
    uint64_t token_total;          // K4: total tokens (CPU sizes tok_pos/tok_kind from it)
    uint64_t carry_overflow_count; // K1: words whose look-back hit MJ_ESCAPE_LOOKBACK_CAP
    uint64_t utf8_scratch;         // K1 scratch (see MjHeaderDev.utf8_error_offset)
    uint64_t tape_word_total;      // K7: total per-token tape words (excl. the 2 root words)
    uint64_t stringbuf_total;      // K7: total string-buffer bytes (Σ raw_len + 5)
    uint64_t skeleton_total;       // K7: skeleton records (brackets + colons + commas)
    uint64_t string_total;         // K7: string-list records (QuoteOpen tokens)
    uint64_t scalar_total;         // K7: scalar-list records (ScalarStart tokens)
    uint64_t reserved[6];          // pads the header to 128 bytes
};

// "No UTF-8 error" sentinel for MjHeaderDev.utf8_error_offset.
constant constexpr uint MJ_NO_UTF8_ERROR = 0xFFFFFFFFu;

// The same 128 bytes viewed by kernels that mutate the header concurrently.
// Field-for-field layout-identical to MjHeader; the 64-bit counters that
// need concurrent updates are split into (atomic lo, plain hi) uint pairs —
// little-endian, so the CPU still reads one u64 — because 64-bit atomics
// are an Apple9+ device feature the embedded metallib cannot assume
// (the AOT target is generic macOS).
//
// Error reduction protocol, stage 1: the only many-writer error source is
// K1's UTF-8 validation, whose code is always MJ_ERR_UTF8 — so K1 threads
// reduce just the OFFSET with a 32-bit atomic_min (offsets fit: input_len
// is capped below u32::MAX), and K2 thread 0 — ordered after all of K1 by
// the serial encoder — folds the winning offset plus its own odd-quote
// verdict into the packed `error` word with plain single-writer stores.
//
// Error reduction protocol, stage 2 (M3, the revisit the original
// packed-u64 atomic_min design was deferred to): K6 errors carry VARIABLE
// codes, so a 32-bit offset atomic cannot represent them. Instead each K6
// threadgroup min-reduces its own packed candidates in threadgroup memory
// (deterministic, no device atomics) and plain-stores one u64 per chunk
// into `chunk_error`; K7 — a single threadgroup ordered after all of K6 by
// the serial encoder — min-folds those per-chunk words plus the existing
// header value into `error` from thread 0 (single writer). The totals
// below are likewise plain single-writer stores by K7 thread 0.
struct MjHeaderDev {
    uint64_t error;                // packed; K2 thread 0, then K7 thread 0
    uint64_t quote_total;          // plain store, single writer (K2 thread 0)
    uint64_t token_total;          // plain store, single writer (K4 thread 0)
    atomic_uint carry_overflow_lo; // K1 fetch_add, many writers
    uint carry_overflow_hi;        // always 0
    atomic_uint utf8_error_offset; // K1 fetch_min, many writers; MJ_NO_UTF8_ERROR = none
    uint scratch_pad;              // always 0
    uint64_t tape_word_total;      // plain store, single writer (K7 thread 0)
    uint64_t stringbuf_total;      // plain store, single writer (K7 thread 0)
    uint64_t skeleton_total;       // plain store, single writer (K7 thread 0)
    uint64_t string_total;         // plain store, single writer (K7 thread 0)
    uint64_t scalar_total;         // plain store, single writer (K7 thread 0)
    uint64_t reserved[6];
};

// K1: report an invalid UTF-8 sequence starting at `byte_offset`.
static inline void mj_report_utf8(device MjHeaderDev* header, uint64_t byte_offset) {
    atomic_fetch_min_explicit(&header->utf8_error_offset, uint(byte_offset),
                              memory_order_relaxed);
}

#endif // METAL_JSON_COMMON_H
