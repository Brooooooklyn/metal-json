// 03_spine.metal — K2 `spine_quote_scan` and K4 `spine_token_scan`.
//
// Each runs as ONE 256-thread threadgroup (Dispatch::Threadgroups(1)) and
// rewrites its per-chunk count buffer IN PLACE as an exclusive prefix sum:
//
//   K2: chunk quote popcounts (from K1)  -> quote-rank carries; the low bit
//       of each carry seeds the chunk's in-string parity in K3/K5. The
//       grand total goes to header.quote_total; an odd total means an
//       unterminated string somewhere, reported as MJ_ERR_STRING packed
//       with offset = input_len (provisional: stage 3 (M3) refines it to
//       the open quote's offset; using input_len keeps every real
//       byte-addressed error — UTF-8 at < input_len — winning atomic_min,
//       matching the reference's stage-order policy).
//   K4: chunk token popcounts (from K3)  -> token-rank carries for the K5
//       scatter. The grand total goes to header.token_total, which the CPU
//       reads after CB1 to size tok_pos/tok_kind exactly.
//
// Capacity: each thread owns ceil(chunks/256) consecutive entries. The max
// input (u32::MAX - 64 bytes) yields 65536 chunks -> 256 entries/thread.

#include "common.h"
#include "tg_scan.h"

// In-place exclusive scan over counts[0..n); returns the grand total to
// every thread. Must be called by all 256 threads of the group.
static inline uint mj_spine_scan_inplace(
    device uint* counts,
    uint n,
    uint lid,
    uint simd_lane,
    uint simd_id,
    threadgroup uint* parts)
{
    uint per = (n + THREADGROUP_SIZE - 1u) / THREADGROUP_SIZE;
    uint base = lid * per;

    uint sum = 0u;
    for (uint k = 0u; k < per; ++k) {
        uint idx = base + k;
        if (idx < n) {
            sum += counts[idx];
        }
    }

    uint prefix = tg_exclusive_scan_256(sum, simd_lane, simd_id, parts);

    uint running = prefix;
    for (uint k = 0u; k < per; ++k) {
        uint idx = base + k;
        if (idx < n) {
            uint c = counts[idx];
            counts[idx] = running;
            running += c;
        }
    }
    return parts[8];
}

kernel void spine_quote_scan(
    device uint* chunk_quote_counts [[buffer(0)]],
    device MjHeaderDev* header [[buffer(1)]],
    constant MjParams& params [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint parts[9];
    uint total = mj_spine_scan_inplace(chunk_quote_counts, uint(params.element_count), lid,
                                       simd_lane, simd_id, parts);
    if (lid == 0u) {
        header->quote_total = ulong(total);

        // Fold stage 1's error sources into the packed error word (single
        // writer: K1 has fully completed under the serial encoder, so the
        // utf8 scratch cell is stable; see MjHeaderDev in common.h).
        uint64_t error = header->error; // MJ_HEADER_NO_ERROR from the CPU
        uint utf8_offset = atomic_load_explicit(&header->utf8_error_offset,
                                                memory_order_relaxed);
        if (utf8_offset != MJ_NO_UTF8_ERROR) {
            error = min(error, mj_pack_error(utf8_offset, MJ_ERR_UTF8));
        }
        if ((total & 1u) != 0u) {
            error = min(error, mj_pack_error(params.input_len, MJ_ERR_STRING));
        }
        header->error = error;
    }
}

kernel void spine_token_scan(
    device uint* chunk_token_counts [[buffer(0)]],
    device MjHeaderDev* header [[buffer(1)]],
    constant MjParams& params [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint parts[9];
    uint total = mj_spine_scan_inplace(chunk_token_counts, uint(params.element_count), lid,
                                       simd_lane, simd_id, parts);
    if (lid == 0u) {
        header->token_total = ulong(total);
    }
}
