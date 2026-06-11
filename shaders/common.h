// common.h — shared declarations for all metal-json kernels.
//
// This header is consumed two ways:
//   1. AOT: `#include`d normally by the Metal compiler in build.rs.
//   2. runtime-shaders: textually inlined by the tiny preprocessor in
//      src/metal/context.rs (newLibraryWithSource has no include paths).
// Keep it self-contained and free of nested includes other than
// <metal_stdlib>, which the preprocessor leaves alone.

#ifndef METAL_JSON_COMMON_H
#define METAL_JSON_COMMON_H

#include <metal_stdlib>
using namespace metal;

// Threads per threadgroup for 1-D dispatches. Multiple of the Apple GPU
// SIMD width (32); revisit per-kernel during M5 tuning.
constant constexpr uint THREADGROUP_SIZE = 256;

// Kernel launch parameters, bound as a single constant buffer.
// Mirrors `MjParams` in src/metal/mod.rs — keep the layouts in sync.
struct MjParams {
    uint64_t input_len;   // total input bytes
    uint64_t element_count; // elements processed by this dispatch
    uint64_t reserved0;   // M1+: chunk counts, token counts, ...
    uint64_t reserved1;
};

// Error codes, packed as (byte_offset << 32) | code into a u64 error word;
// earliest error wins via atomic_min. Placeholders until M1 fixes the set —
// mirrors SyntaxErrorKind in src/error.rs.
enum MjErrorCode : uint {
    MJ_OK = 0,
    MJ_ERR_UTF8 = 1,
    MJ_ERR_SYNTAX = 2,
    MJ_ERR_DEPTH_LIMIT = 3,
    MJ_ERR_TRAILING_CONTENT = 4,
    MJ_ERR_NUMBER = 5,
    MJ_ERR_STRING = 6,
};

#endif // METAL_JSON_COMMON_H
