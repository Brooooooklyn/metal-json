// bitmap_u2.h — uint2 64-bit bitmap vocabulary shared by the M2-M4 kernels.
//
// BINDING DECISION (docs/spikes.md, spike B): the bitmap kernels (K1
// classify, K3 token mask, K5 token scatter) keep their 64-byte-chunk
// bitmaps as `uint2` — two 32-bit words with explicit carries — NOT `ulong`.
// Measured on the target machine (Apple M5 Max) the uint2 formulation is
// ~2x faster in the exact K1-style op mix: it reaches the ~550 GB/s memory
// ceiling while the ulong variant stays ALU-bound below half of it (Apple
// GPUs have 32-bit ALUs; loop-carried ulong shifts/adds lower poorly).
// ulong remains fine for cold paths and self-checks — 01_u2_selftest.metal
// deliberately uses it as the slow-but-native oracle for these helpers.
//
// Representation: a 64-bit bitmap word w is `uint2(lo, hi)`:
//     w.x = lo = bits  0..31  (input bytes 0..31 of the 64-byte chunk)
//     w.y = hi = bits 32..63  (input bytes 32..63)
// Stored to memory this is exactly the little-endian u64 layout, so the
// Rust side reads bitmap buffers directly as `&[u64]` and diffs them
// against the CPU reference's `Vec<u64>` with no conversion.
//
// Shift-amount contract: shl64_u2/shr64_u2 accept ANY amount — 0 returns x
// unchanged and >= 64 returns 0 (unlike native C/MSL shifts, where a count
// >= the bit width is undefined behavior).
//
// Like common.h this header is consumed both by the AOT Metal compiler and
// by the textual include inliner in src/metal/context.rs (runtime-shaders),
// so it stays self-contained apart from <metal_stdlib>.

#ifndef METAL_JSON_BITMAP_U2_H
#define METAL_JSON_BITMAP_U2_H

#include <metal_stdlib>
using namespace metal;

// --- construction / projection ----------------------------------------------

// Build a 64-bit bitmap word from its 32-bit halves.
static inline uint2 make_u2(uint lo, uint hi) {
    return uint2(lo, hi);
}

// Low 32 bits (bits 0..31 = bytes 0..31 of the 64-byte chunk).
static inline uint lo_u2(uint2 x) {
    return x.x;
}

// High 32 bits (bits 32..63 = bytes 32..63 of the 64-byte chunk).
static inline uint hi_u2(uint2 x) {
    return x.y;
}

// --- bitwise logic (componentwise; trivially carry-free) ---------------------

// ~x
static inline uint2 not_u2(uint2 x) {
    return ~x;
}

// a | b
static inline uint2 or_u2(uint2 a, uint2 b) {
    return a | b;
}

// a & b
static inline uint2 and_u2(uint2 a, uint2 b) {
    return a & b;
}

// a ^ b
static inline uint2 xor_u2(uint2 a, uint2 b) {
    return a ^ b;
}

// --- shifts -------------------------------------------------------------------

// x << s for any s (s == 0 -> x, s >= 64 -> 0). Bits move from lo into hi
// across the 32-bit seam. The three-way branch keeps every native shift
// count strictly inside 0..31 (a native shift by 32 is undefined).
static inline uint2 shl64_u2(uint2 x, uint s) {
    if (s == 0u) {
        return x;
    }
    if (s >= 64u) {
        return uint2(0u);
    }
    if (s >= 32u) {
        return make_u2(0u, x.x << (s - 32u));
    }
    return make_u2(x.x << s, (x.y << s) | (x.x >> (32u - s)));
}

// Logical x >> s for any s (s == 0 -> x, s >= 64 -> 0). Mirror of shl64_u2:
// bits move from hi into lo across the seam.
static inline uint2 shr64_u2(uint2 x, uint s) {
    if (s == 0u) {
        return x;
    }
    if (s >= 64u) {
        return uint2(0u);
    }
    if (s >= 32u) {
        return make_u2(x.y >> (s - 32u), 0u);
    }
    return make_u2((x.x >> s) | (x.y << (32u - s)), x.y >> s);
}

// --- arithmetic ----------------------------------------------------------------

// a + b (mod 2^64); carry_out = 1 iff the true sum overflowed 64 bits.
// The low-half carry propagates into hi explicitly. This is the 64-bit
// add-with-carry simdjson's find_escaped trick needs (it adds the
// odd-position backslash starts to the backslash bitmap and the escape
// state for the next word is the carry-out).
static inline uint2 add64_u2(uint2 a, uint2 b, thread uint& carry_out) {
    uint lo = a.x + b.x;
    uint carry_lo = lo < a.x ? 1u : 0u;
    uint hi_raw = a.y + b.y;
    uint hi = hi_raw + carry_lo;
    // Overflow of either the raw high add or the +carry_lo step (the two
    // cannot both wrap: if hi_raw wrapped it is <= 0xFFFFFFFE).
    carry_out = (hi_raw < a.y || hi < hi_raw) ? 1u : 0u;
    return make_u2(lo, hi);
}

// a - b (mod 2^64); borrow_out = 1 iff b > a. Mirror of add64_u2 with the
// low-half borrow propagating into hi.
static inline uint2 sub64_u2(uint2 a, uint2 b, thread uint& borrow_out) {
    uint lo = a.x - b.x;
    uint borrow_lo = a.x < b.x ? 1u : 0u;
    uint hi_raw = a.y - b.y;
    uint hi = hi_raw - borrow_lo;
    borrow_out = (a.y < b.y || hi_raw < borrow_lo) ? 1u : 0u;
    return make_u2(lo, hi);
}

// --- reductions / scans ----------------------------------------------------------

// Number of set bits, 0..64. Two native 32-bit popcounts (a single 64-bit
// popcount costs ~3x a uint one on Apple ALUs per spike A, Q3).
static inline uint popcount64_u2(uint2 x) {
    return popcount(x.x) + popcount(x.y);
}

// Inclusive prefix-XOR: output bit i = parity of input bits 0..=i. This is
// the in-string mask primitive of K3: for a quote bitmap, output bit i is 1
// iff byte i is inside a string (or is its opening quote), modulo the
// chunk-parity carry the caller XORs in afterwards.
//
// Formulation of the 6-step u64 shift ladder (m ^= m<<1; ... m ^= m<<32) in
// uint2: the five in-word steps run on both halves at once (vector shifts
// do not cross components), then the s = 32 step collapses to a seam fix —
// after the in-word ladder, bit 31 of lo is the parity of ALL 32 low bits,
// and if that parity is odd every hi bit flips. Broadcast and XOR.
static inline uint2 prefix_xor64_u2(uint2 x) {
    x ^= x << 1u;
    x ^= x << 2u;
    x ^= x << 4u;
    x ^= x << 8u;
    x ^= x << 16u;
    x.y ^= 0u - (x.x >> 31); // all-ones iff lo's total parity is odd
    return x;
}

// Leading-zero count from bit 63 downward; 64 for x == 0.
// (metal's clz(uint(0)) == 32, which makes the zero case fall out.)
static inline uint clz64_u2(uint2 x) {
    return x.y != 0u ? clz(x.y) : 32u + clz(x.x);
}

// Trailing-zero count from bit 0 upward; 64 for x == 0.
// (metal's ctz(uint(0)) == 32, same fall-out.) ctz is the K5 token-scatter
// bit iterator: position of the next token bit within the word.
static inline uint ctz64_u2(uint2 x) {
    return x.x != 0u ? ctz(x.x) : 32u + ctz(x.y);
}

#endif // METAL_JSON_BITMAP_U2_H
