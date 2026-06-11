// 00_smoke.metal — M0 smoke kernels proving the toolchain end to end:
// AOT compile -> metallib -> library load -> pipeline -> dispatch -> readback.

#include "common.h"

// out[i] = a[i] + b[i] over uint buffers.
kernel void smoke_add(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant MjParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.element_count) {
        return;
    }
    out[gid] = a[gid] + b[gid];
}

// out[i] = popcount(in[i]) — proves 64-bit integer (ulong) buffers and ops
// work end to end (the bitmap pipeline depends on them).
kernel void smoke_popcount64(
    device const ulong* in [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant MjParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.element_count) {
        return;
    }
    out[gid] = uint(popcount(in[gid]));
}
