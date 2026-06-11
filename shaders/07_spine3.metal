// 07_spine3.metal — K7 `spine3`, the multi-component CB2 spine scan.
//
// Runs as ONE 256-thread threadgroup (Dispatch::Threadgroups(1)) over the
// per-token-chunk partials K6 produced, between K6 and the CPU sync 2:
//
//   1. chunk_counts (uint4: tape words, skeleton, string, scalar counts)
//      -> rewritten IN PLACE as 4 exclusive prefix sums — the chunk carries
//      K6b adds to its in-chunk ranks; the 4 grand totals go to the header
//      (tape_word_total, skeleton_total, string_total, scalar_total), from
//      which the CPU exact-allocates the tape and the three lists.
//   2. chunk_string_bytes (ulong: Σ raw_len + 5 per chunk) -> the same
//      in-place exclusive scan in 64-bit (a single string literal can
//      exceed u32, and the grand total exceeds u32 on dense-string inputs
//      like `["",""...]`, which is why the tape format gives string
//      offsets 40 bits). The total goes to header.stringbuf_total — the
//      CPU sync 2 string-buffer allocation — and the per-chunk carries
//      stay in place for the M4 string kernel (K11) to consume.
//   3. chunk_error (ulong, packed (offset << 32) | code, min-reduced per
//      chunk by K6) -> min-folded, together with the existing header value,
//      into header.error by thread 0 (single writer; the serial encoder
//      orders K7 after all of K6). Earliest offset wins; the MjErrorCode
//      numeric order breaks ties — the common.h tie-break contract.
//
// 64-bit components use per-thread ulong accumulation plus a 256-entry
// threadgroup ladder folded serially by thread 0: simdgroup scan intrinsics
// are 32-bit, and a one-time 256-step serial fold is noise for a spine
// kernel (same cost model as the K2/K4 spines).
//
// Capacity: each thread owns ceil(chunks/256) consecutive entries; the max
// input (u32::MAX - 64 bytes of single-byte tokens) yields ~4.2M chunks ->
// ~16K entries/thread, well within bounds (correctness-first; M5 retunes
// the chunk grain if this spine ever shows up in a profile).

#include "common.h"
#include "tg_scan.h"

kernel void spine3(
    device uint4* chunk_counts [[buffer(0)]],
    device ulong* chunk_string_bytes [[buffer(1)]],
    device const ulong* chunk_error [[buffer(2)]],
    device MjHeaderDev* header [[buffer(3)]],
    constant MjParams& params [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint4 parts4[9];
    threadgroup ulong lanes[THREADGROUP_SIZE + 1];

    uint n = uint(params.element_count); // token chunks
    uint per = (n + THREADGROUP_SIZE - 1u) / THREADGROUP_SIZE;
    uint base = lid * per;

    // --- 1) uint4 counts: in-place 4-component exclusive scan -------------
    uint4 sum = uint4(0u);
    for (uint k = 0u; k < per; ++k) {
        uint idx = base + k;
        if (idx < n) {
            sum += chunk_counts[idx];
        }
    }
    uint4 running = tg_exclusive_scan4_256(sum, simd_lane, simd_id, parts4);
    for (uint k = 0u; k < per; ++k) {
        uint idx = base + k;
        if (idx < n) {
            uint4 c = chunk_counts[idx];
            chunk_counts[idx] = running;
            running += c;
        }
    }
    uint4 totals = parts4[8];

    // --- 2) string bytes: in-place 64-bit exclusive scan -------------------
    ulong ssum = 0;
    for (uint k = 0u; k < per; ++k) {
        uint idx = base + k;
        if (idx < n) {
            ssum += chunk_string_bytes[idx];
        }
    }
    lanes[lid] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        ulong run = 0;
        for (uint i = 0u; i < THREADGROUP_SIZE; ++i) {
            ulong t = lanes[i];
            lanes[i] = run;
            run += t;
        }
        lanes[THREADGROUP_SIZE] = run; // grand total
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong srun = lanes[lid];
    for (uint k = 0u; k < per; ++k) {
        uint idx = base + k;
        if (idx < n) {
            ulong c = chunk_string_bytes[idx];
            chunk_string_bytes[idx] = srun;
            srun += c;
        }
    }
    ulong stringbuf_total = lanes[THREADGROUP_SIZE];

    // --- 3) error fold + totals (single writer) ----------------------------
    uint64_t emin = MJ_HEADER_NO_ERROR;
    for (uint k = 0u; k < per; ++k) {
        uint idx = base + k;
        if (idx < n) {
            emin = min(emin, chunk_error[idx]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // lanes reuse fence
    lanes[lid] = emin;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        uint64_t folded = header->error; // K2 left NO_ERROR on clean inputs
        for (uint i = 0u; i < THREADGROUP_SIZE; ++i) {
            folded = min(folded, lanes[i]);
        }
        header->error = folded;
        header->tape_word_total = ulong(totals.x);
        header->skeleton_total = ulong(totals.y);
        header->string_total = ulong(totals.z);
        header->scalar_total = ulong(totals.w);
        header->stringbuf_total = stringbuf_total;
    }
}
