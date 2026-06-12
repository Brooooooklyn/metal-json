// 13_strings.metal — K11 `strings_unescape` + its offset feeder
// `string_record_offsets`: the M4 string stage.
//
//   string_record_offsets  1 threadgroup / 1024-token chunk (the K6b shape):
//                          refines the K7 per-chunk string carries into the
//                          per-string record offsets — the exclusive prefix
//                          sum of `raw_len + 5` over QuoteOpen tokens in
//                          document order (docs/tape-format.md's pinned
//                          offset-allocation policy). Pure token-position
//                          math: it runs BEFORE any unescaping, which is
//                          exactly why the policy uses raw lengths.
//   strings_unescape       1 thread / string-list entry: validate + unescape
//                          one string literal into its
//                          [u32 LE len][content][NUL] record and write the
//                          string tape word at tape_ofs[token]. Escape-free
//                          strings (the overwhelming majority) are copied
//                          by a lane-parallel (short) or simdgroup-
//                          cooperative (long) clean scan; strings that hit
//                          a backslash / control byte are compacted across
//                          the threadgroup and rerun through the sequential
//                          unescape path with full surrogate-pair
//                          validation (see the kernel comment).
//   structure_finalize     (10_pair_ctx.metal, REUSED) 1 threadgroup:
//                          min-fold the per-chunk K11 error words into
//                          header.error.
//
// The bit-exact spec is the CPU oracle `reference::stage6_strings`
// (src/reference/strings.rs): escapes `\" \\ \/ \b \f \n \r \t`,
// `\uXXXX` (case-insensitive hex) including UTF-16 surrogate pairs, the
// legal backslash-u-0000 interior NUL; rejected are lone/inverted surrogates, bad
// or short hex, unknown escapes (all MJ_ERR_STRING_ESCAPE at the
// backslash) and raw control bytes < 0x20 (MJ_ERR_STRING_CONTROL at the
// byte). Each thread reports its string's FIRST (leftmost) error; string
// extents are disjoint and in document order, so the packed-min reduction
// reproduces the reference's first-error verdict exactly (and the two K11
// codes can never tie at one offset: a byte is either a backslash or a
// control character, never both).
//
// Geometry: `string_record_offsets` mirrors K6b (one 256-thread group per
// MJ_TOK_CHUNK_TOKENS tokens, MJ_TOK_PER_THREAD each); `strings_unescape`
// runs MJ_STR_CHUNK_STRINGS = 256 strings per threadgroup, one per thread.
// Both dispatch as FULL threadgroups (Dispatch::Threadgroups) — the
// cooperative scans / reductions below are convergent; out-of-range
// threads contribute zeros / no-error and still reach every barrier.
//
// LONG-STRING VALVE (the availability fix for the v1 thread-per-string
// cliff): one thread owns one whole string, so without a valve a single
// multi-MB string would serialize the entire parse on one lane (latency,
// potential command-buffer timeout). Strings whose raw_len exceeds
// MJ_LONG_STRING_THRESHOLD are therefore NOT processed here: the thread
// appends the STRING-LIST INDEX to a long-string fixup list via a device
// atomic counter (order irrelevant; the CPU sorts) and contributes
// no-error to the fold. After the command buffer completes, the CPU patch
// pass (`gpu::strings::patch_long_strings`, the reference unescaper)
// writes the flagged records into their precomputed slots — possible
// precisely because record offsets are pure token-position math
// (docs/tape-format.md's raw-length allocation), independent of content.
// This is the K10 number-fixup pattern applied to K11.
//
// Memory safety: every write lands inside the exact-size allocations the
// CPU made from the K7 totals — record bytes stay inside the record's own
// `raw_len + 5` slot (unescaped output is never longer than raw input, the
// reference's `bytes.len() <= raw.len()` invariant), and the tape word
// lands at tape_ofs[token] < the tape length passed in reserved0 (checked
// defensively). Gap bytes between a shrunk record's NUL and the next slot
// are left UNSPECIFIED (never written) per the pinned gap policy.

#include "common.h"
#include "tape_types.h"
#include "tg_scan.h"

// --- M4 string error codes -------------------------------------------------------
// Extend the MjErrorCode space (common.h tops out at MJ_ERR_EMPTY_INPUT =
// 22). Defined here rather than in common.h so the M4 scalar kernels can
// land independently; fold into the MjErrorCode enum at parser
// integration. Mirror ERR_STRING_ESCAPE / ERR_STRING_CONTROL in
// src/gpu/strings.rs — keep in sync (a Rust test parses this file and pins
// them). No same-offset tie-break constraint exists for these codes (see
// the header comment above), so their order is free.
constant constexpr uint MJ_ERR_STRING_ESCAPE = 23;  // SyntaxErrorKind::InvalidStringEscape
constant constexpr uint MJ_ERR_STRING_CONTROL = 24; // SyntaxErrorKind::ControlCharacterInString

// Strings per K11 threadgroup: one per thread.
constant constexpr uint MJ_STR_CHUNK_STRINGS = THREADGROUP_SIZE;

// Long-string valve threshold: strings with raw_len STRICTLY ABOVE this
// many bytes are deferred to the CPU patch pass instead of being walked by
// one GPU thread. 16384 (one 16 KiB page) is long enough that real-world
// strings almost never cross it (JSON string values are overwhelmingly
// sub-KB; the GPU keeps the whole hot path), yet short enough that the
// worst case a single lane ever owns is one page-sized walk — never
// megabytes — bounding K11's serial tail regardless of input shape.
// Mirror LONG_STRING_THRESHOLD in src/gpu/strings.rs — keep in sync (a
// Rust test parses this file and pins them).
constant constexpr uint MJ_LONG_STRING_THRESHOLD = 16384;

// Fast-path block width in bytes (the vectorized clean-scan grain of the
// careful per-thread path).
constant constexpr uint MJ_STR_BLOCK = 16;

// Strings with raw_len STRICTLY ABOVE this many bytes are copied
// simdgroup-cooperatively (32 lanes × consecutive bytes — coalesced and
// divergence-free) instead of by one lane's byte loop. 32 keeps the
// lane-parallel phase's worst case at one block while the cooperative
// phase pays one ~5-instruction setup per string, which only wins once a
// string spans multiple 32-byte blocks. (Twitter-shaped data: ~91% of
// strings are ≤ 32 B but the ≥ 32 B tail owns ~half the string bytes and,
// pre-split, made every SIMD lane wait out the longest string's byte walk.)
constant constexpr uint MJ_STR_COOP_CUT = 32;

// --- string_record_offsets ---------------------------------------------------------

// One threadgroup per 1024-token chunk, 4 tokens per thread (the K6b
// shape). For every QuoteOpen token, materialize the byte offset its
// string record starts at:
//
//   offset = chunk_string_bytes[chunk]   (K7 exclusive chunk carry)
//          + in-chunk exclusive prefix of slot bytes over earlier tokens
//
// written to record_offsets[rank] with rank = chunk_counts[chunk].z (the
// K7 string-count carry) + the in-chunk exclusive string rank — the same
// dense document-order ranks K6b used for string_tokens, so entry s of
// the two arrays describes the same string. Slot bytes (`raw_len + 5`)
// recompute mj_token_info's slot_bytes (06_validate.metal) from the same
// inputs, so these offsets are exactly the K6/K7 partials' refinement;
// the grand total equals header.stringbuf_total.
//
// The in-chunk slot prefix is 64-bit (one string literal can exceed u32);
// like K7's string-byte scan it uses a 256-lane threadgroup ladder folded
// serially by thread 0 — simdgroup scan intrinsics are 32-bit, and a
// 256-step serial fold is noise at this grain.
kernel void string_record_offsets(
    device const uint* tok_pos [[buffer(0)]],
    device const uchar* tok_kind [[buffer(1)]],
    device const uint4* chunk_counts [[buffer(2)]],       // K7 exclusive carries
    device const ulong* chunk_string_bytes [[buffer(3)]], // K7 exclusive carries
    device ulong* record_offsets [[buffer(4)]],
    constant MjParams& params [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint parts[9];
    threadgroup ulong lanes[THREADGROUP_SIZE + 1];

    ulong n = params.element_count; // token_total
    ulong base = ulong(tgid) * ulong(MJ_TOK_CHUNK_TOKENS) + ulong(lid) * ulong(MJ_TOK_PER_THREAD);

    bool is_str[MJ_TOK_PER_THREAD];
    ulong slot[MJ_TOK_PER_THREAD];
    uint str_sum = 0u;
    ulong slot_sum = 0;
    for (uint j = 0u; j < MJ_TOK_PER_THREAD; ++j) {
        ulong t = base + ulong(j);
        is_str[j] = false;
        slot[j] = 0;
        if (t < n && uint(tok_kind[t]) == MJ_TOK_QUOTE_OPEN) {
            is_str[j] = true;
            // The close quote is the very next token (post-CB1 even quote
            // total); the t+1 guard is defensive only, mirroring
            // mj_token_info. Slot = raw_len + 4-byte length prefix + NUL.
            uint raw_len = (t + 1 < n) ? (tok_pos[t + 1] - tok_pos[t] - 1u) : 0u;
            slot[j] = ulong(raw_len)
                + ulong(MJ_STRING_RECORD_HEADER_BYTES + MJ_STRING_RECORD_TRAILER_BYTES);
            str_sum += 1u;
            slot_sum += slot[j];
        }
    }

    // In-chunk exclusive string rank (32-bit cooperative scan).
    uint rank = chunk_counts[tgid].z
        + tg_exclusive_scan_256(str_sum, simd_lane, simd_id, parts);

    // In-chunk exclusive slot-byte prefix (64-bit ladder).
    lanes[lid] = slot_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        ulong run = 0;
        for (uint k = 0u; k < THREADGROUP_SIZE; ++k) {
            ulong v = lanes[k];
            lanes[k] = run;
            run += v;
        }
        lanes[THREADGROUP_SIZE] = run;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong offset = chunk_string_bytes[tgid] + lanes[lid];

    for (uint j = 0u; j < MJ_TOK_PER_THREAD; ++j) {
        if (is_str[j]) {
            record_offsets[rank] = offset;
            rank += 1u;
            offset += slot[j];
        }
    }
}

// --- strings_unescape ----------------------------------------------------------------

// 4 hex digits (case-insensitive) at raw[at..at+4], or -1 when short or
// non-hex. Mirrors reference `hex4`.
static inline int mj_str_hex4(device const uchar* raw, ulong at, ulong raw_len) {
    if (at + 4 > raw_len) {
        return -1;
    }
    int v = 0;
    for (uint k = 0u; k < 4u; ++k) {
        uchar c = raw[at + ulong(k)];
        uint d;
        if (c >= uchar('0') && c <= uchar('9')) {
            d = uint(c - uchar('0'));
        } else if (c >= uchar('a') && c <= uchar('f')) {
            d = uint(c - uchar('a')) + 10u;
        } else if (c >= uchar('A') && c <= uchar('F')) {
            d = uint(c - uchar('A')) + 10u;
        } else {
            return -1;
        }
        v = v * 16 + int(d);
    }
    return v;
}

// Validate + unescape string-list entry `s`. On success writes the
// [u32 LE len][content][NUL] record at record_offsets[s] and the string
// tape word at tape_ofs[token], and returns MJ_HEADER_NO_ERROR; on the
// string's first (leftmost) error returns the packed error word and
// guarantees nothing about the record bytes (rejected outputs are never
// read — the rejection contract). Mirrors reference `unescape` plus the
// record/tape emission of `emit_tape`.
static inline uint64_t mj_str_unescape_one(
    device const uchar* input,
    device const uint* tok_pos,
    device const uint* string_tokens,
    device const ulong* record_offsets,
    device const uint* tape_ofs,
    device uchar* stringbuf,
    device ulong* tape,
    ulong s,
    ulong token_total,
    ulong tape_words)
{
    ulong t = ulong(string_tokens[s]);
    if (t + 1 >= token_total) {
        // Defensive only: post-CB2 every QuoteOpen has an adjacent close.
        return MJ_HEADER_NO_ERROR;
    }
    ulong open_pos = ulong(tok_pos[t]);
    ulong close_pos = ulong(tok_pos[t + 1]);
    ulong raw_len = close_pos - open_pos - 1;
    device const uchar* raw = input + (open_pos + 1);
    ulong base = open_pos + 1; // absolute offset of raw[0], for errors

    ulong rec = record_offsets[s];
    device uchar* dst = stringbuf + rec + ulong(MJ_STRING_RECORD_HEADER_BYTES);

    ulong i = 0; // raw cursor
    ulong o = 0; // content cursor (o <= i always: unescaping never grows)
    while (i < raw_len) {
        // FAST PATH: copy 16-byte blocks until one contains an escape or a
        // control byte. Writes are eager (dst[o..o+16] = raw block) — safe
        // because o + 16 <= i + 16 <= raw_len keeps them inside the
        // record's content area, and a dirty block's bytes are overwritten
        // by the careful steps below (or discarded with the whole output
        // on an error).
        while (i + ulong(MJ_STR_BLOCK) <= raw_len) {
            bool clean = true;
            for (uint k = 0u; k < MJ_STR_BLOCK; ++k) {
                uchar b = raw[i + ulong(k)];
                dst[o + ulong(k)] = b;
                clean = clean && (b >= uchar(0x20)) && (b != uchar('\\'));
            }
            if (!clean) {
                break;
            }
            i += ulong(MJ_STR_BLOCK);
            o += ulong(MJ_STR_BLOCK);
        }

        // CAREFUL STEPS: handle up to one block's worth of units (bytes or
        // whole escapes) before retrying the fast loop, bounding the
        // re-scan cost at a dirty block to one extra 16-byte check.
        for (uint step = 0u; step < MJ_STR_BLOCK && i < raw_len; ++step) {
            uchar b = raw[i];
            if (b < uchar(0x20)) {
                // Raw control characters must be escaped.
                return mj_pack_error(base + i, MjErrorCode(MJ_ERR_STRING_CONTROL));
            }
            if (b != uchar('\\')) {
                // UTF-8 continuation bytes were validated in stage 1; copy
                // verbatim. (An unescaped `"` cannot appear: the extent
                // ends at the first unescaped quote.)
                dst[o] = b;
                o += 1;
                i += 1;
                continue;
            }

            // Escape errors point at the backslash (reference parity).
            uint64_t esc = mj_pack_error(base + i, MjErrorCode(MJ_ERR_STRING_ESCAPE));
            if (i + 1 >= raw_len) {
                // A trailing lone backslash cannot occur via stages 1-3
                // (it would have escaped the closing quote); stay graceful.
                return esc;
            }
            uchar d = raw[i + 1];
            switch (d) {
                case '"':
                case '\\':
                case '/':
                    dst[o] = d;
                    o += 1;
                    i += 2;
                    break;
                case 'b':
                    dst[o] = uchar(0x08);
                    o += 1;
                    i += 2;
                    break;
                case 'f':
                    dst[o] = uchar(0x0C);
                    o += 1;
                    i += 2;
                    break;
                case 'n':
                    dst[o] = uchar(0x0A);
                    o += 1;
                    i += 2;
                    break;
                case 'r':
                    dst[o] = uchar(0x0D);
                    o += 1;
                    i += 2;
                    break;
                case 't':
                    dst[o] = uchar(0x09);
                    o += 1;
                    i += 2;
                    break;
                case 'u': {
                    int first = mj_str_hex4(raw, i + 2, raw_len);
                    if (first < 0) {
                        return esc; // short or non-hex digits
                    }
                    uint cp;
                    ulong consumed;
                    if (first >= 0xD800 && first <= 0xDBFF) {
                        // High surrogate: must be chased by a low-surrogate
                        // escape.
                        if (i + 8 > raw_len || raw[i + 6] != uchar('\\')
                            || raw[i + 7] != uchar('u')) {
                            return esc;
                        }
                        int low = mj_str_hex4(raw, i + 8, raw_len);
                        if (low < 0xDC00 || low > 0xDFFF) {
                            return esc; // bad hex or not a low surrogate
                        }
                        cp = 0x10000u + ((uint(first) - 0xD800u) << 10)
                            + (uint(low) - 0xDC00u);
                        consumed = 12;
                    } else if (first >= 0xDC00 && first <= 0xDFFF) {
                        // Lone / inverted low surrogate.
                        return esc;
                    } else {
                        cp = uint(first);
                        consumed = 6;
                    }
                    // UTF-8 encode (cp <= 0x10FFFF and never a surrogate
                    // by construction).
                    if (cp < 0x80u) {
                        dst[o] = uchar(cp);
                        o += 1;
                    } else if (cp < 0x800u) {
                        dst[o] = uchar(0xC0u | (cp >> 6));
                        dst[o + 1] = uchar(0x80u | (cp & 0x3Fu));
                        o += 2;
                    } else if (cp < 0x10000u) {
                        dst[o] = uchar(0xE0u | (cp >> 12));
                        dst[o + 1] = uchar(0x80u | ((cp >> 6) & 0x3Fu));
                        dst[o + 2] = uchar(0x80u | (cp & 0x3Fu));
                        o += 3;
                    } else {
                        dst[o] = uchar(0xF0u | (cp >> 18));
                        dst[o + 1] = uchar(0x80u | ((cp >> 12) & 0x3Fu));
                        dst[o + 2] = uchar(0x80u | ((cp >> 6) & 0x3Fu));
                        dst[o + 3] = uchar(0x80u | (cp & 0x3Fu));
                        o += 4;
                    }
                    i += consumed;
                    break;
                }
                default:
                    // Unknown escapes (`\q`, `\x41`, `\u{1F600}`, ...).
                    return esc;
            }
        }
    }

    // The record: [u32 LE length][content][NUL]. The length counts content
    // bytes only; byte stores because `rec` has no alignment guarantee
    // (slots are raw_len + 5). o <= raw_len < 2^32, so the cast is exact.
    uint len = uint(o);
    stringbuf[rec] = uchar(len & 0xFFu);
    stringbuf[rec + 1] = uchar((len >> 8) & 0xFFu);
    stringbuf[rec + 2] = uchar((len >> 16) & 0xFFu);
    stringbuf[rec + 3] = uchar((len >> 24) & 0xFFu);
    dst[o] = uchar(0);

    // The string tape word at this token's tape position.
    ulong pos = ulong(tape_ofs[t]);
    if (pos < tape_words) { // defensive; always true on accepted inputs
        tape[pos] = mj_make_string(rec);
    }
    return MJ_HEADER_NO_ERROR;
}

// Finish a CLEAN (copied-verbatim) record: the [u32 LE len] header, the
// trailing NUL, and the string tape word — exactly the record/tape epilogue
// of mj_str_unescape_one with `o == raw_len` (clean strings never shrink).
static inline void mj_str_finish_record(
    device uchar* stringbuf,
    device ulong* tape,
    device const uint* tape_ofs,
    ulong rec,
    ulong len,
    ulong t,
    ulong tape_words)
{
    stringbuf[rec] = uchar(len & 0xFFu);
    stringbuf[rec + 1] = uchar((len >> 8) & 0xFFu);
    stringbuf[rec + 2] = uchar((len >> 16) & 0xFFu);
    stringbuf[rec + 3] = uchar((len >> 24) & 0xFFu);
    stringbuf[rec + ulong(MJ_STRING_RECORD_HEADER_BYTES) + len] = uchar(0);
    ulong pos = ulong(tape_ofs[t]);
    if (pos < tape_words) { // defensive; always true on accepted inputs
        tape[pos] = mj_make_string(rec);
    }
}

// K11: one thread per string-list entry, MJ_STR_CHUNK_STRINGS entries per
// threadgroup. Validates + unescapes every string at or under the
// long-string threshold, writes the records and string tape words, and
// min-reduces one packed error word per chunk into chunk_error (thread 0
// plain store, no device atomics — the K6 pattern); `structure_finalize`
// folds those into header.error afterwards. Strings ABOVE the threshold
// are appended (string-list index) to the long-string fixup list instead —
// the CPU patch pass owns their records, tape words AND their error
// verdicts (see the valve note in the header comment).
//
// EXECUTION SHAPE (the M5 divergence fix; semantics unchanged): strings
// without escapes or control bytes are verbatim copies, and on real data
// they are the overwhelming majority (~93% on twitter), so the kernel is
// organized as a clean-copy attempt with the original sequential
// validator/unescaper demoted to a rare per-thread fallback:
//
//   phase 1  raw_len ≤ MJ_STR_COOP_CUT: each lane scan-copies its own
//            string byte-wise (bounded at one 32-byte block — short
//            strings no longer wait out a long neighbor's walk);
//   phase 2  raw_len > MJ_STR_COOP_CUT: the whole simdgroup copies one
//            such string at a time, 32 consecutive bytes per step —
//            coalesced loads/stores, zero divergence;
//   phase 3  strings whose scan hit a backslash or control byte rerun
//            through mj_str_unescape_one UNCHANGED (its leftmost-error
//            verdicts and record bytes are the pinned semantics; the
//            phase-1/2 prefix writes it overwrites — or strands as gap
//            bytes, which are contractually unspecified).
//
// Every phase-1/2 write stays inside the string's own record content area
// (`dst[idx]`, idx < raw_len < slot), so the eager copy can never touch a
// neighboring record.
//
//   params.element_count = string_total
//   params.reserved0     = tape length in words (defensive bound)
//   params.reserved1     = token_total (defensive bound)
kernel void strings_unescape(
    device const uchar* input [[buffer(0)]],
    device const uint* tok_pos [[buffer(1)]],
    device const uint* string_tokens [[buffer(2)]],
    device const ulong* record_offsets [[buffer(3)]],
    device const uint* tape_ofs [[buffer(4)]],
    device uchar* stringbuf [[buffer(5)]],
    device ulong* tape [[buffer(6)]],
    device ulong* chunk_error [[buffer(7)]],
    device atomic_uint* long_count [[buffer(8)]],
    device uint* long_list [[buffer(9)]],
    constant MjParams& params [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup ulong lanes[THREADGROUP_SIZE];

    ulong n = params.element_count; // string_total
    ulong token_total = params.reserved1;
    ulong tape_words = params.reserved0;
    ulong s = ulong(tgid) * ulong(MJ_STR_CHUNK_STRINGS) + ulong(lid);

    uint64_t err = MJ_HEADER_NO_ERROR;

    // Per-lane metadata (consecutive s → coalesced loads). raw_len is pure
    // token-position math (the same `close - open - 1`
    // mj_str_unescape_one recomputes), so the routing needs no content
    // reads.
    bool todo = false; // a string this kernel owns (in range, not long)
    ulong t = 0;
    ulong open_pos = 0;
    ulong raw_len = 0;
    if (s < n) {
        t = ulong(string_tokens[s]);
        if (t + 1 < token_total) { // defensive, mirrors mj_str_unescape_one
            open_pos = ulong(tok_pos[t]);
            raw_len = ulong(tok_pos[t + 1]) - open_pos - 1;
            if (raw_len > ulong(MJ_LONG_STRING_THRESHOLD)) {
                // The long-string valve: defer to the CPU patch pass —
                // append the string-list index (at most one append per
                // thread, so slot < string_total and the index-sized list
                // the runner allocated cannot overflow). err stays
                // NO_ERROR: this string's verdict belongs to the CPU.
                uint slot = atomic_fetch_add_explicit(long_count, 1u,
                                                      memory_order_relaxed);
                long_list[slot] = uint(s);
            } else {
                todo = true;
            }
        }
    }

    // --- phase 1: lane-parallel clean scan-copy (short strings) -----------
    bool dirty = false; // hit a backslash / control byte → phase 3
    bool coop = todo && raw_len > ulong(MJ_STR_COOP_CUT);
    if (todo && !coop) {
        device const uchar* raw = input + (open_pos + 1);
        ulong rec = record_offsets[s];
        device uchar* dst = stringbuf + rec + ulong(MJ_STRING_RECORD_HEADER_BYTES);
        bool clean = true;
        for (ulong i = 0; i < raw_len; ++i) {
            uchar b = raw[i];
            dst[i] = b;
            clean = clean && (b >= uchar(0x20)) && (b != uchar('\\'));
        }
        if (clean) {
            mj_str_finish_record(stringbuf, tape, tape_ofs, rec, raw_len, t,
                                 tape_words);
        } else {
            dirty = true;
        }
    }

    // --- phase 2: simdgroup-cooperative clean scan-copy (long strings) ----
    // One balloted string at a time; every lane re-reads its metadata from
    // the same addresses (broadcast loads — cache-served, no 64-bit
    // shuffles needed). The ballot mask is simdgroup-uniform, so the loop
    // and the simd_all vote below execute convergently.
    ulong sg_base = ulong(tgid) * ulong(MJ_STR_CHUNK_STRINGS) + ulong(simd_id) * 32ul;
    uint mask = uint(static_cast<simd_vote::vote_t>(simd_ballot(coop)));
    while (mask != 0u) {
        uint u = ctz(mask);
        mask &= mask - 1u;
        ulong su = sg_base + ulong(u);
        ulong tu = ulong(string_tokens[su]);
        ulong open_u = ulong(tok_pos[tu]);
        ulong len_u = ulong(tok_pos[tu + 1]) - open_u - 1;
        ulong rec_u = record_offsets[su];
        device const uchar* raw = input + (open_u + 1);
        device uchar* dst = stringbuf + rec_u + ulong(MJ_STRING_RECORD_HEADER_BYTES);
        bool u_dirty = false;
        for (ulong blk = 0; blk < len_u; blk += 32ul) {
            ulong idx = blk + ulong(simd_lane);
            uchar b = uchar(0x20); // out-of-range lanes vote clean
            if (idx < len_u) {
                b = raw[idx];
                dst[idx] = b;
            }
            bool ok = (b >= uchar(0x20)) && (b != uchar('\\'));
            if (!simd_all(ok)) {
                u_dirty = true;
                break;
            }
        }
        if (simd_lane == u) { // the owner lane records the outcome
            if (u_dirty) {
                dirty = true;
            } else {
                mj_str_finish_record(stringbuf, tape, tape_ofs, rec_u, len_u,
                                     tu, tape_words);
            }
        }
    }

    // --- phase 3: the careful path for strings with escapes ---------------
    // Compacted across the threadgroup first: on escape-sprinkled data
    // (twitter: ~7% of strings) almost every SIMD group would otherwise
    // host one or two dirty strings and stall all 32 lanes for one lane's
    // sequential unescape walk. Queueing the dirty lids through threadgroup
    // memory packs them into the fewest possible SIMD groups; the rest of
    // the threadgroup sails through. Reassignment is safe because a dirty
    // string's record/tape writes and error verdict are functions of the
    // string alone, and the chunk verdict below is an order-independent
    // min over the whole threadgroup (which lane computed it is
    // unobservable).
    threadgroup atomic_uint dirty_n;
    threadgroup uchar dirty_lids[THREADGROUP_SIZE];
    if (lid == 0u) {
        atomic_store_explicit(&dirty_n, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dirty) {
        uint q = atomic_fetch_add_explicit(&dirty_n, 1u, memory_order_relaxed);
        dirty_lids[q] = uchar(lid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint dn = atomic_load_explicit(&dirty_n, memory_order_relaxed);
    if (lid < dn) {
        ulong ds = ulong(tgid) * ulong(MJ_STR_CHUNK_STRINGS)
            + ulong(dirty_lids[lid]);
        err = mj_str_unescape_one(input, tok_pos, string_tokens,
                                  record_offsets, tape_ofs, stringbuf,
                                  tape, ds, token_total, tape_words);
    }

    // Deterministic per-chunk min (every thread reaches the barriers; the
    // helper above returns rather than exiting the kernel): barrier tree
    // instead of a serial 256-element walk by thread 0.
    lanes[lid] = err;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = THREADGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            lanes[lid] = min(lanes[lid], lanes[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        chunk_error[tgid] = lanes[0];
    }
}
