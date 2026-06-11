// tape_types.h — tape format v1 constants for MSL kernels.
//
// MIRROR of src/tape.rs, which is the canonical definition of the layout.
// The `msl_header_layout_lock` unit test in src/tape.rs parses this header
// at test time and asserts that every `MJ_*` constant below exists with
// exactly the Rust value (and that nothing extra is defined here) — change
// the two files together or `cargo test` fails.
// Prose spec + worked example: docs/tape-format.md.
//
// Parsing contract for that test: every constant is declared on a single
// line of the exact shape
//     constant constexpr <type> MJ_NAME = <integer-literal>;
// with <integer-literal> decimal or 0x hex (suffixes tolerated). Inline
// helper *functions* are free-form; only `constant constexpr` lines are
// layout-locked.
//
// Like common.h this header is consumed both by the AOT Metal compiler and
// by the textual include inliner in src/metal/context.rs (runtime-shaders),
// so it stays self-contained apart from <metal_stdlib>.

#ifndef METAL_JSON_TAPE_TYPES_H
#define METAL_JSON_TAPE_TYPES_H

#include <metal_stdlib>
using namespace metal;

// --- Tape word encoding -----------------------------------------------------
// Every tape entry is one 64-bit word:
//     word = ((ulong)tag << MJ_TAPE_TAG_SHIFT) | payload
// with the ASCII tag in the top byte and a 56-bit payload below it.
constant constexpr uint  MJ_TAPE_TAG_SHIFT = 56;
constant constexpr ulong MJ_TAPE_PAYLOAD_MASK = 0x00FFFFFFFFFFFFFF;

// --- Tags (ASCII, identical to simdjson's tape characters) ------------------
constant constexpr uchar MJ_TAG_ROOT = 0x72;         // 'r'
constant constexpr uchar MJ_TAG_START_OBJECT = 0x7B; // '{'
constant constexpr uchar MJ_TAG_END_OBJECT = 0x7D;   // '}'
constant constexpr uchar MJ_TAG_START_ARRAY = 0x5B;  // '['
constant constexpr uchar MJ_TAG_END_ARRAY = 0x5D;    // ']'
constant constexpr uchar MJ_TAG_STRING = 0x22;       // '"'
constant constexpr uchar MJ_TAG_INT64 = 0x6C;        // 'l' (next word: i64 bits)
constant constexpr uchar MJ_TAG_UINT64 = 0x75;       // 'u' (next word: u64 value)
constant constexpr uchar MJ_TAG_DOUBLE = 0x64;       // 'd' (next word: f64 bits)
constant constexpr uchar MJ_TAG_TRUE = 0x74;         // 't'
constant constexpr uchar MJ_TAG_FALSE = 0x66;        // 'f'
constant constexpr uchar MJ_TAG_NULL = 0x6E;         // 'n'

// --- Container payloads ------------------------------------------------------
// Open ('{' '['):  bits 0..32  index one past the matching close word
//                  bits 32..56 direct-child count, saturated at MAX
// Close ('}' ']'): bits 0..32  index of the matching open word
constant constexpr ulong MJ_CONTAINER_INDEX_MASK = 0xFFFFFFFF;
constant constexpr uint  MJ_CONTAINER_COUNT_SHIFT = 32;
constant constexpr uint  MJ_CONTAINER_COUNT_MAX = 0xFFFFFF;

// --- String payloads ---------------------------------------------------------
// '"' payload = byte offset (56 bits) into the string buffer where a
// [u32 LE length][unescaped utf8 bytes][NUL] record starts.
constant constexpr ulong MJ_STRING_OFFSET_MASK = 0x00FFFFFFFFFFFFFF;
constant constexpr uint  MJ_STRING_RECORD_HEADER_BYTES = 4; // u32 LE length
constant constexpr uint  MJ_STRING_RECORD_TRAILER_BYTES = 1; // NUL

// --- Encode/decode helpers (mirror the helpers in src/tape.rs) ---------------

static inline ulong mj_make_entry(uchar tag, ulong payload) {
    return ((ulong)tag << MJ_TAPE_TAG_SHIFT) | (payload & MJ_TAPE_PAYLOAD_MASK);
}

static inline uchar mj_tape_tag(ulong word) {
    return uchar(word >> MJ_TAPE_TAG_SHIFT);
}

static inline ulong mj_tape_payload(ulong word) {
    return word & MJ_TAPE_PAYLOAD_MASK;
}

// Open word for '{' or '['; count saturates at MJ_CONTAINER_COUNT_MAX.
static inline ulong mj_make_open(uchar tag, uint end_index, uint count) {
    uint saturated = min(count, MJ_CONTAINER_COUNT_MAX);
    return mj_make_entry(
        tag, ((ulong)saturated << MJ_CONTAINER_COUNT_SHIFT) | (ulong)end_index);
}

// Close word for '}' or ']'.
static inline ulong mj_make_close(uchar tag, uint open_index) {
    return mj_make_entry(tag, (ulong)open_index);
}

static inline ulong mj_make_string(ulong stringbuf_offset) {
    return mj_make_entry(MJ_TAG_STRING, stringbuf_offset);
}

static inline uint mj_container_end_index(ulong word) {
    return uint(word & MJ_CONTAINER_INDEX_MASK);
}

static inline uint mj_container_count(ulong word) {
    return uint(word >> MJ_CONTAINER_COUNT_SHIFT) & MJ_CONTAINER_COUNT_MAX;
}

static inline uint mj_container_open_index(ulong word) {
    return uint(word & MJ_CONTAINER_INDEX_MASK);
}

static inline ulong mj_string_offset(ulong word) {
    return word & MJ_TAPE_PAYLOAD_MASK;
}

#endif // METAL_JSON_TAPE_TYPES_H
