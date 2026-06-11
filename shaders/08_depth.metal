// 08_depth.metal — the CB3 depth scan over the skeleton (reference stage 4
// phase 1), as the hierarchical reduce → spine → apply triple:
//
//   depth_partials  1 threadgroup / MJ_SKEL_CHUNK_ELEMS = 1024-element
//                   chunk: per-chunk sum of the ±1 bracket weights
//   depth_spine     1 threadgroup: exclusive scan of the chunk sums (the
//                   signed depth entering each chunk)
//   depth_apply     per element: depth value + the two depth-scan error
//                   checks, one min-reduced error word per chunk
//
// The bit-exact spec is `reference::stage4_structure` phase 1
// (src/reference/structure.rs):
//
//   open  -> depth AFTER increment (P + 1); P + 1 > max_depth is
//            MJ_ERR_DEPTH_LIMIT at the open's pos
//   close -> depth BEFORE decrement (P); P <= 0 is the close-below-root
//            underflow, MJ_ERR_UNBALANCED at the close's pos
//   sep   -> current depth (P)
//
// where P is the exclusive prefix sum of the weights — the depth before
// the element.
//
// DOCUMENTED DEVIATION (verdict-preserving): the reference *parks*
// underflowed closes at depth 0 and does not decrement below 0, while a
// prefix sum keeps going negative. The two formulations agree on every
// depth up to and including the FIRST underflow (before it the clamped and
// unclamped depths are equal; the underflow itself is flagged by both at
// the same close, with the same offset and code). Past that point this
// stage has already produced the input's verdict — later depths only need
// to stay in sortable range (the stores clamp negatives to 0; mj_sort_key
// clamps high), because the rejection contract discards every CB3 output
// of a rejected input. Clean inputs never underflow, so their depths are
// bit-identical to the reference's.
//
// Arithmetic widths: per-chunk weight sums fit int (|sum| <= 1024), but
// the running depth across chunks is genuinely 64-bit — a skeleton can
// exceed 2^31 elements at max input size — so the spine and the applied
// prefix use long.
//
// All three kernels are dispatched as FULL 256-thread threadgroups
// (Dispatch::Threadgroups); the cooperative scan in depth_apply is
// convergent. No buffer here carries a zero/init precondition: every
// chunk_depth / depths / chunk_error entry is plain-stored exactly once
// per dispatch.

#include "common.h"
#include "tg_scan.h"

// --- depth_partials -------------------------------------------------------------

// Per-chunk signed weight sum (the reduce half of the hierarchical scan).
kernel void depth_partials(
    device const uchar* skel_byte [[buffer(0)]],
    device long* chunk_depth [[buffer(1)]],
    constant MjParams& params [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup int lanes[THREADGROUP_SIZE];

    ulong m = params.element_count; // skeleton_total
    ulong base = ulong(tgid) * ulong(MJ_SKEL_CHUNK_ELEMS)
        + ulong(lid) * ulong(MJ_SKEL_PER_THREAD);

    int sum = 0;
    for (uint j = 0u; j < MJ_SKEL_PER_THREAD; ++j) {
        ulong t = base + ulong(j);
        if (t < m) {
            sum += mj_skel_weight(skel_byte[t]);
        }
    }
    lanes[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        long total = 0;
        for (uint i = 0u; i < THREADGROUP_SIZE; ++i) {
            total += long(lanes[i]);
        }
        chunk_depth[tgid] = total;
    }
}

// --- depth_spine ----------------------------------------------------------------

// One threadgroup: in-place exclusive scan of the chunk weight sums — the
// signed depth entering each chunk (the K2/K4/K7 spine shape, in long; a
// 256-step serial ladder fold is noise for a spine kernel).
kernel void depth_spine(
    device long* chunk_depth [[buffer(0)]],
    constant MjParams& params [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup long lanes[THREADGROUP_SIZE + 1];

    uint n = uint(params.element_count); // skeleton chunks
    uint per = (n + THREADGROUP_SIZE - 1u) / THREADGROUP_SIZE;
    uint base = lid * per;

    long sum = 0;
    for (uint k = 0u; k < per; ++k) {
        uint idx = base + k;
        if (idx < n) {
            sum += chunk_depth[idx];
        }
    }
    lanes[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        long run = 0;
        for (uint i = 0u; i < THREADGROUP_SIZE; ++i) {
            long t = lanes[i];
            lanes[i] = run;
            run += t;
        }
        lanes[THREADGROUP_SIZE] = run;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long run = lanes[lid];
    for (uint k = 0u; k < per; ++k) {
        uint idx = base + k;
        if (idx < n) {
            long c = chunk_depth[idx];
            chunk_depth[idx] = run;
            run += c;
        }
    }
}

// --- depth_apply ----------------------------------------------------------------

// Per element: materialize the depth and evaluate the two depth-scan error
// checks; min-reduce one packed error word per chunk into chunk_error
// (plain single-writer store by thread 0 — the K6 error shape; K9 later
// min-folds its own candidates into the same buffer, and
// structure_finalize folds the buffer into the header).
//
// In-chunk exclusive weight prefix via the uint cooperative scan with a +4
// bias per thread: a thread's 4-element weight sum is in [-4, +4], so
// (sum + 4) is in [0, 8] and the scanned total stays tiny; subtracting
// 4 * lid recovers the signed prefix.
kernel void depth_apply(
    device const uchar* skel_byte [[buffer(0)]],
    device const uint* skel_pos [[buffer(1)]],
    device const long* chunk_depth [[buffer(2)]], // exclusive chunk carries
    device uint* depths [[buffer(3)]],
    device ulong* chunk_error [[buffer(4)]],
    constant MjParams& params [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint parts[9];
    threadgroup ulong lanes[THREADGROUP_SIZE];

    ulong m = params.element_count; // skeleton_total
    long max_depth = long(params.reserved0);
    ulong base = ulong(tgid) * ulong(MJ_SKEL_CHUNK_ELEMS)
        + ulong(lid) * ulong(MJ_SKEL_PER_THREAD);

    int w[MJ_SKEL_PER_THREAD];
    int tsum = 0;
    for (uint j = 0u; j < MJ_SKEL_PER_THREAD; ++j) {
        ulong t = base + ulong(j);
        w[j] = (t < m) ? mj_skel_weight(skel_byte[t]) : 0;
        tsum += w[j];
    }

    uint scanned = tg_exclusive_scan_256(uint(tsum + 4), simd_lane, simd_id, parts);
    long prefix = chunk_depth[tgid] + long(int(scanned) - 4 * int(lid));

    uint64_t err = MJ_HEADER_NO_ERROR;
    for (uint j = 0u; j < MJ_SKEL_PER_THREAD; ++j) {
        ulong t = base + ulong(j);
        if (t < m) {
            uchar b = skel_byte[t];
            ulong pos = ulong(skel_pos[t]);
            if (mj_is_open_byte(b)) {
                long d = prefix + 1;
                if (d > max_depth) {
                    err = min(err, mj_pack_error(pos, MJ_ERR_DEPTH_LIMIT));
                }
                depths[t] = uint(max(d, 0l));
            } else if (mj_is_close_byte(b)) {
                if (prefix <= 0) {
                    // Close with nothing open (`1]`, `{}}`): the reference's
                    // close-below-root underflow, at the close's offset.
                    err = min(err, mj_pack_error(pos, MJ_ERR_UNBALANCED));
                }
                depths[t] = uint(max(prefix, 0l));
            } else {
                depths[t] = uint(max(prefix, 0l));
            }
            prefix += long(w[j]);
        }
    }

    lanes[lid] = err;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        uint64_t chunk_min = MJ_HEADER_NO_ERROR;
        for (uint i = 0u; i < THREADGROUP_SIZE; ++i) {
            chunk_min = min(chunk_min, lanes[i]);
        }
        chunk_error[tgid] = chunk_min;
    }
}
