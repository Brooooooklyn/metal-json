// tg_scan.h — threadgroup-wide prefix sums for the M2 kernels (K2-K5).
//
// All cooperating kernels run full 256-thread threadgroups
// (`Dispatch::Threadgroups`, never `Threads`): every thread must reach the
// scan call — out-of-range threads contribute 0 — because simdgroup scans
// and threadgroup barriers are convergent operations.
//
// Geometry: THREADGROUP_SIZE = 256 = 8 simdgroups of 32 (the Apple GPU SIMD
// width; src/gpu/stage1.rs asserts the pipeline supports 256-wide groups).
//
// Like common.h this header is consumed both by the AOT Metal compiler and
// by the textual include inliner in src/metal/context.rs (runtime-shaders),
// so it stays self-contained apart from <metal_stdlib>.

#ifndef METAL_JSON_TG_SCAN_H
#define METAL_JSON_TG_SCAN_H

#include <metal_stdlib>
using namespace metal;

// Exclusive prefix sum of `v` across a full 256-thread threadgroup.
//
// `parts` must be a threadgroup array of at least 9 uints shared by the
// whole group. On return every thread holds the sum of all lower-indexed
// threads' `v`, and `parts[8]` holds the grand total (valid for every
// thread). Safe to call repeatedly with the same `parts`: the call begins
// with a barrier that fences the previous call's reads.
static inline uint tg_exclusive_scan_256(
    uint v,
    uint simd_lane,
    uint simd_id,
    threadgroup uint* parts)
{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint lane_prefix = simd_prefix_exclusive_sum(v);
    if (simd_lane == 31u) {
        parts[simd_id] = lane_prefix + v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u && simd_lane == 0u) {
        uint running = 0u;
        for (uint i = 0u; i < 8u; ++i) {
            uint t = parts[i];
            parts[i] = running;
            running += t;
        }
        parts[8] = running;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return parts[simd_id] + lane_prefix;
}

#endif // METAL_JSON_TG_SCAN_H
