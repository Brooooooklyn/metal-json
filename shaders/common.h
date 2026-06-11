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
    uint64_t reserved0;   // M1+: chunk counts, token counts, ...
    uint64_t reserved1;
};

// Error codes, packed as (byte_offset << 32) | code into a u64 error word;
// earliest error wins via atomic_min. Placeholders until M1 fixes the set —
// mirrors SyntaxErrorKind in src/error.rs.
enum MjErrorCode : uint {
    MJ_OK = 0,
    MJ_ERR_UTF8 = 1,
    MJ_ERR_SYNTAX = 2,
    MJ_ERR_DEPTH_LIMIT = 3,
    MJ_ERR_TRAILING_CONTENT = 4,
    MJ_ERR_NUMBER = 5,
    MJ_ERR_STRING = 6,
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

// Per-parse result header (one 64-byte cell per parse).
// Mirrors `MjHeader` in src/metal/mod.rs — keep the layouts in sync
// (eight uint64_t fields, no padding).
//
// Kernels reduce errors into `error` with a 64-bit atomic_min (bind the
// cell as `device atomic_ulong*`); the CPU initializes the field to
// MJ_HEADER_NO_ERROR and decodes offset = error >> 32, code = low 32 bits.
struct MjHeader {
    uint64_t error;                // packed (offset << 32) | code, atomic_min
    uint64_t quote_total;          // K2: total real quotes (odd => unterminated string)
    uint64_t token_total;          // K4: total tokens (CPU sizes tok_pos/tok_kind from it)
    uint64_t carry_overflow_count; // K1: words whose look-back hit MJ_ESCAPE_LOOKBACK_CAP
    uint64_t reserved[4];          // pads the header to 64 bytes
};

// "No UTF-8 error" sentinel for MjHeaderDev.utf8_error_offset.
constant constexpr uint MJ_NO_UTF8_ERROR = 0xFFFFFFFFu;

// The same 64 bytes viewed by kernels that mutate the header concurrently.
// Field-for-field layout-identical to MjHeader; the 64-bit counters that
// need concurrent updates are split into (atomic lo, plain hi) uint pairs —
// little-endian, so the CPU still reads one u64 — because 64-bit atomics
// are an Apple9+ device feature the embedded metallib cannot assume
// (the AOT target is generic macOS; the original packed-u64 atomic_min
// design is revisited in M3 when more concurrent error sources appear).
//
// Error reduction protocol, stage 1: the only many-writer error source is
// K1's UTF-8 validation, whose code is always MJ_ERR_UTF8 — so K1 threads
// reduce just the OFFSET with a 32-bit atomic_min (offsets fit: input_len
// is capped below u32::MAX), and K2 thread 0 — ordered after all of K1 by
// the serial encoder — folds the winning offset plus its own odd-quote
// verdict into the packed `error` word with plain single-writer stores.
struct MjHeaderDev {
    uint64_t error;                // packed (offset << 32) | code; K2 thread 0 only
    uint64_t quote_total;          // plain store, single writer (K2 thread 0)
    uint64_t token_total;          // plain store, single writer (K4 thread 0)
    atomic_uint carry_overflow_lo; // K1 fetch_add, many writers
    uint carry_overflow_hi;        // always 0
    atomic_uint utf8_error_offset; // K1 fetch_min, many writers; MJ_NO_UTF8_ERROR = none
    uint scratch_pad;              // always 0
    uint64_t reserved[3];
};

// K1: report an invalid UTF-8 sequence starting at `byte_offset`.
static inline void mj_report_utf8(device MjHeaderDev* header, uint64_t byte_offset) {
    atomic_fetch_min_explicit(&header->utf8_error_offset, uint(byte_offset),
                              memory_order_relaxed);
}

#endif // METAL_JSON_COMMON_H
