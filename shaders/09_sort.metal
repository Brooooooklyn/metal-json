// 09_sort.metal — K8: stable counting sort of the skeleton by depth
// (reference stage 4 phase 2), as LSD radix passes of MJ_SORT_RADIX_BITS =
// 5-bit digits over the keys from mj_sort_key. Each pass is the classic
// counting-sort triple:
//
//   sort_hist         1 threadgroup / 1024-element chunk: 32-bucket digit
//                     histogram of the chunk, stored BUCKET-MAJOR
//                     (hist[bucket * chunks + chunk])
//   sort_matrix_scan  1 threadgroup: flat exclusive scan over the
//                     32 x chunks bucket-major matrix — afterwards
//                     hist[b * c + i] is the global output slot where chunk
//                     i's first digit-b element lands (elements with a
//                     smaller digit anywhere, plus digit-b elements in
//                     earlier chunks)
//   sort_scatter      1 threadgroup / chunk: stable scatter — each element
//                     goes to matrix carry + its in-chunk stable rank
//                     within its bucket
//
// The Rust side picks the pass count from the max_depth limit
// (`stage::sort_passes`): 1 pass covers key_max < 32, 2 passes the 1024
// default, and deeper limits just add passes. Pass k sorts by digit
// (key >> 5k) & 31; LSD passes over a stable per-pass sort yield the final
// (key, document-order) order.
//
// STABILITY IS CORRECTNESS-CRITICAL, not an optimization detail: within a
// depth group, document order is exactly what makes brackets strictly
// alternate open/close, which K9's adjacent pairing (and the reference's
// group walk) relies on. Stability here comes from three nested orders,
// each preserving document order within a bucket:
//   1. the bucket-major matrix scan orders chunks of one bucket by chunk
//      index (= document order across chunks),
//   2. the in-chunk cross-thread scan orders threads by lid (= document
//      order across threads, since thread t owns elements [4t, 4t+4)),
//   3. each thread walks its 4 elements in element order.
//
// Pass 0 reads the identity ordering (the document-order skeleton itself,
// signalled by the MjParams.reserved1 flag) so no identity permutation is
// ever materialized; later passes read the previous pass's output
// (ping-pong buffers, orchestrated in src/gpu/stage3.rs).
//
// Dispatched as FULL 256-thread threadgroups; the cooperative scans are
// convergent. No zero/init preconditions: sort_hist plain-stores all
// 32 x chunks entries, and the scatter writes a permutation (every output
// slot exactly once — the histogram counts every element, so the scanned
// slots partition 0..m).

#include "common.h"
#include "tg_scan.h"

// Digit of the element's sort key for the current pass.
static inline uint mj_sort_digit(uint depth, uint key_max, uint shift) {
    return (mj_sort_key(depth, key_max) >> shift) & (MJ_SORT_BUCKETS - 1u);
}

// Decode the pass parameters packed into MjParams.reserved1 by the
// orchestration: low byte = digit shift, bit 8 = "input ordering is the
// identity" (pass 0).
static inline uint mj_sort_shift(uint64_t reserved1) {
    return uint(reserved1 & 0xFFul);
}
static inline bool mj_sort_identity(uint64_t reserved1) {
    return (reserved1 >> 8) != 0;
}

// --- sort_hist ------------------------------------------------------------------

kernel void sort_hist(
    device const uint* depths [[buffer(0)]],
    device const uint* order_in [[buffer(1)]], // ignored on identity passes
    device uint* hist [[buffer(2)]],           // 32 x chunks, bucket-major
    constant MjParams& params [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup atomic_uint tg_hist[MJ_SORT_BUCKETS];
    if (lid < MJ_SORT_BUCKETS) {
        atomic_store_explicit(&tg_hist[lid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong m = params.element_count; // skeleton_total
    uint c = uint((m + ulong(MJ_SKEL_CHUNK_ELEMS) - 1) / ulong(MJ_SKEL_CHUNK_ELEMS));
    uint key_max = mj_key_max(params.reserved0);
    uint shift = mj_sort_shift(params.reserved1);
    bool identity = mj_sort_identity(params.reserved1);
    ulong base = ulong(tgid) * ulong(MJ_SKEL_CHUNK_ELEMS)
        + ulong(lid) * ulong(MJ_SKEL_PER_THREAD);

    for (uint j = 0u; j < MJ_SKEL_PER_THREAD; ++j) {
        ulong t = base + ulong(j);
        if (t < m) {
            uint e = identity ? uint(t) : order_in[t];
            uint digit = mj_sort_digit(depths[e], key_max, shift);
            // Threadgroup-local counting only; counts are order-independent,
            // so the result is deterministic despite the atomics.
            atomic_fetch_add_explicit(&tg_hist[digit], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < MJ_SORT_BUCKETS) {
        hist[lid * c + tgid] = atomic_load_explicit(&tg_hist[lid], memory_order_relaxed);
    }
}

// --- sort_matrix_scan -----------------------------------------------------------

// One threadgroup: in-place flat exclusive scan over the bucket-major
// histogram matrix (element_count = 32 * chunks). uint suffices: the grand
// total is the skeleton size, which fits u32 by the input-size cap.
kernel void sort_matrix_scan(
    device uint* hist [[buffer(0)]],
    constant MjParams& params [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint parts[9];

    uint n = uint(params.element_count); // 32 * chunks
    uint per = (n + THREADGROUP_SIZE - 1u) / THREADGROUP_SIZE;
    uint base = lid * per;

    uint sum = 0u;
    for (uint k = 0u; k < per; ++k) {
        uint idx = base + k;
        if (idx < n) {
            sum += hist[idx];
        }
    }
    uint running = tg_exclusive_scan_256(sum, simd_lane, simd_id, parts);
    for (uint k = 0u; k < per; ++k) {
        uint idx = base + k;
        if (idx < n) {
            uint v = hist[idx];
            hist[idx] = running;
            running += v;
        }
    }
}

// --- sort_scatter ---------------------------------------------------------------

kernel void sort_scatter(
    device const uint* depths [[buffer(0)]],
    device const uint* order_in [[buffer(1)]], // ignored on identity passes
    device const uint* hist [[buffer(2)]],     // scanned matrix carries
    device uint* order_out [[buffer(3)]],
    constant MjParams& params [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint4 parts4[9];

    ulong m = params.element_count;
    uint c = uint((m + ulong(MJ_SKEL_CHUNK_ELEMS) - 1) / ulong(MJ_SKEL_CHUNK_ELEMS));
    uint key_max = mj_key_max(params.reserved0);
    uint shift = mj_sort_shift(params.reserved1);
    bool identity = mj_sort_identity(params.reserved1);
    ulong base = ulong(tgid) * ulong(MJ_SKEL_CHUNK_ELEMS)
        + ulong(lid) * ulong(MJ_SKEL_PER_THREAD);

    uint elem[MJ_SKEL_PER_THREAD];
    uint digit[MJ_SKEL_PER_THREAD];
    bool valid[MJ_SKEL_PER_THREAD];
    // Per-thread bucket counts as 8 uint4 lanes (32 buckets), so the
    // cross-thread stable ranks come from 8 uses of the 4-component scan.
    uint4 cnt[8];
    for (uint g = 0u; g < 8u; ++g) {
        cnt[g] = uint4(0u);
    }
    for (uint j = 0u; j < MJ_SKEL_PER_THREAD; ++j) {
        ulong t = base + ulong(j);
        valid[j] = t < m;
        elem[j] = 0u;
        digit[j] = 0u;
        if (valid[j]) {
            elem[j] = identity ? uint(t) : order_in[t];
            digit[j] = mj_sort_digit(depths[elem[j]], key_max, shift);
            cnt[digit[j] >> 2][digit[j] & 3u] += 1u;
        }
    }

    // Exclusive cross-thread prefix per bucket (8 sequential 4-component
    // scans sharing one parts4 array — documented safe in tg_scan.h). All
    // threads participate in all 8: the scans are convergent.
    uint4 tbase[8];
    for (uint g = 0u; g < 8u; ++g) {
        tbase[g] = tg_exclusive_scan4_256(cnt[g], simd_lane, simd_id, parts4);
    }

    for (uint j = 0u; j < MJ_SKEL_PER_THREAD; ++j) {
        if (valid[j]) {
            uint d = digit[j];
            // Stable in-thread rank: earlier same-digit elements of this
            // thread (at most 3 — a cheap rescan beats a 32-counter array).
            uint within = 0u;
            for (uint jj = 0u; jj < j; ++jj) {
                if (valid[jj] && digit[jj] == d) {
                    within += 1u;
                }
            }
            uint4 tb = tbase[d >> 2];
            uint thread_rank = (d & 3u) == 0u ? tb.x
                : (d & 3u) == 1u              ? tb.y
                : (d & 3u) == 2u              ? tb.z
                                              : tb.w;
            uint dst = hist[d * c + tgid] + thread_rank + within;
            order_out[dst] = elem[j];
        }
    }
}
