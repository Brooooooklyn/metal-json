// 01_u2_selftest.metal — throwaway validation kernel for bitmap_u2.h.
//
// Every uint2 helper is exercised against an expected value computed
// independently with native ulong arithmetic. Apple GPUs DO support ulong —
// it is merely ~2x slower in the hot bitmap mix (docs/spikes.md, spike B),
// which is exactly what banished it from the real kernels and exactly what
// makes it a fine in-kernel oracle here. clz/ctz expectations come from
// plain bit-scan loops instead, so they do not depend on 64-bit builtins.
//
// One thread per (a, b) test pair. fail[i] is a bitmask of failed checks
// (0 = all pass); bit assignments mirror FAIL_NAMES in tests/kernels.rs —
// keep the two in sync.

#include "common.h"
#include "bitmap_u2.h"

// uint2 (lo, hi) -> the ulong it represents.
static inline ulong u2_back(uint2 x) {
    return ulong(x.x) | (ulong(x.y) << 32);
}

// ulong -> uint2 (lo, hi).
static inline uint2 u2_from(ulong v) {
    return make_u2(uint(v), uint(v >> 32));
}

kernel void u2_selftest(
    device const ulong* a [[buffer(0)]],
    device const ulong* b [[buffer(1)]],
    device uint* fail [[buffer(2)]],
    constant MjParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.element_count) {
        return;
    }
    ulong av = a[gid];
    ulong bv = b[gid];
    uint2 a2 = u2_from(av);
    uint2 b2 = u2_from(bv);
    uint bad = 0u;

    // bit 0: make_u2 / lo_u2 / hi_u2 roundtrip
    if (u2_back(a2) != av || lo_u2(a2) != uint(av) || hi_u2(a2) != uint(av >> 32)) {
        bad |= 1u << 0;
    }
    // bits 1-4: componentwise logic
    if (u2_back(not_u2(a2)) != ~av) {
        bad |= 1u << 1;
    }
    if (u2_back(or_u2(a2, b2)) != (av | bv)) {
        bad |= 1u << 2;
    }
    if (u2_back(and_u2(a2, b2)) != (av & bv)) {
        bad |= 1u << 3;
    }
    if (u2_back(xor_u2(a2, b2)) != (av ^ bv)) {
        bad |= 1u << 4;
    }

    // bits 5/6: every shift amount 0..=64 (the helpers define >= 64 as 0;
    // the native expectation must guard that case, where ulong shifts are
    // undefined).
    for (uint s = 0u; s <= 64u; ++s) {
        ulong shl_expect = 0;
        ulong shr_expect = 0;
        if (s < 64u) {
            shl_expect = av << s;
            shr_expect = av >> s;
        }
        if (u2_back(shl64_u2(a2, s)) != shl_expect) {
            bad |= 1u << 5;
        }
        if (u2_back(shr64_u2(a2, s)) != shr_expect) {
            bad |= 1u << 6;
        }
    }

    // bits 7/8: add64_u2 sum + carry-out
    {
        uint carry = 99u; // poison: must be overwritten
        ulong sum = av + bv;
        if (u2_back(add64_u2(a2, b2, carry)) != sum) {
            bad |= 1u << 7;
        }
        if (carry != (sum < av ? 1u : 0u)) {
            bad |= 1u << 8;
        }
    }

    // bits 9/10: sub64_u2 difference + borrow-out
    {
        uint borrow = 99u;
        if (u2_back(sub64_u2(a2, b2, borrow)) != av - bv) {
            bad |= 1u << 9;
        }
        if (borrow != (av < bv ? 1u : 0u)) {
            bad |= 1u << 10;
        }
    }

    // bit 11: popcount64_u2 vs native ulong popcount
    if (popcount64_u2(a2) != uint(popcount(av))) {
        bad |= 1u << 11;
    }

    // bit 12: prefix_xor64_u2 vs the native ulong 6-step shift ladder
    {
        ulong px = av;
        px ^= px << 1;
        px ^= px << 2;
        px ^= px << 4;
        px ^= px << 8;
        px ^= px << 16;
        px ^= px << 32;
        if (u2_back(prefix_xor64_u2(a2)) != px) {
            bad |= 1u << 12;
        }
    }

    // bits 13/14: clz64_u2 / ctz64_u2 vs bit-scan loops (64 for zero input)
    {
        uint clz_expect = 64u;
        for (uint k = 0u; k < 64u; ++k) {
            if ((av >> (63u - k)) & ulong(1)) {
                clz_expect = k;
                break;
            }
        }
        if (clz64_u2(a2) != clz_expect) {
            bad |= 1u << 13;
        }
        uint ctz_expect = 64u;
        for (uint k = 0u; k < 64u; ++k) {
            if ((av >> k) & ulong(1)) {
                ctz_expect = k;
                break;
            }
        }
        if (ctz64_u2(a2) != ctz_expect) {
            bad |= 1u << 14;
        }
    }

    fail[gid] = bad;
}
