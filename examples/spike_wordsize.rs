//! Spike B (M0): bitmap word size for the K1/K3/K5 bitmap kernels.
//!
//! Question: on Apple GPUs (32-bit ALUs, emulated 64-bit integer ops), should
//! the M2 bitmap kernels keep their 64-byte-chunk bitmaps as a single `ulong`,
//! or as `uint2` (lo/hi pair with explicit carries across the 32-bit
//! boundary)?
//!
//! Method: two kernels do identical K1-style work per thread — load 64 input
//! bytes as 4x `uint4`, build quote/backslash/structural bitmaps via per-byte
//! compares and shifts, then run a representative bitmap-op sequence (the
//! simdjson `find_escaped` ladder incl. a 64-bit add with carry-out, the
//! 6-step prefix-xor shift ladder, plus masking shifts/ands). Variant 1 uses
//! `ulong` bitmaps; variant 2 uses `uint2` with hand-written carry/shift
//! plumbing. Both write one 8-byte checksum per chunk; outputs must match
//! bit-for-bit (and match a scalar CPU model on sampled chunks).
//!
//! `rounds=1` is the representative K1 ALU:traffic mix; `rounds=8` repeats the
//! bitmap-op sequence (with a per-round dependency) to amplify ALU cost in
//! case `rounds=1` is purely memory-bound.
//!
//! Timing: median of 10 runs after 3 warm-ups, command-buffer
//! GPUStartTime/GPUEndTime deltas (host-timing fallback if zero).
//!
//! Run: `cargo run --release --example spike_wordsize`
//!
//! This spike is deliberately self-contained (own MSL source, raw
//! objc2-metal) so it does not perturb the crate's shader set or wrappers.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::time::Instant;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResourceOptions, MTLSize,
};

/// 64-byte chunks, one per thread. 16Mi chunks = 1 GiB of input traffic.
const CHUNKS: usize = 16 * 1024 * 1024;
const INPUT_BYTES: usize = CHUNKS * 64;
const OUT_BYTES: usize = CHUNKS * 8;
const THREADGROUP: usize = 256;
const WARMUP: usize = 3;
const RUNS: usize = 10;

/// Mirrors `struct SpikeParams` in the MSL source below.
#[repr(C)]
#[derive(Clone, Copy)]
struct SpikeParams {
    chunk_count: u32,
    rounds: u32,
}

const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct SpikeParams {
    uint chunk_count;
    uint rounds;
};

// ============================ variant 1: ulong ============================

// simdjson prefix-xor: 6-step shift ladder on a 64-bit word.
inline ulong prefix_xor64(ulong x) {
    x ^= x << 1;
    x ^= x << 2;
    x ^= x << 4;
    x ^= x << 8;
    x ^= x << 16;
    x ^= x << 32;
    return x;
}

kernel void spike_bitmap_ulong(
    device const uint4 *input    [[buffer(0)]],
    device ulong *out            [[buffer(1)]],
    constant SpikeParams &p      [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]])
{
    if (gid >= p.chunk_count) { return; }

    // K1-style bitmap build: 64 bytes -> 3 bitmaps, per-byte compare + shift.
    ulong quote = 0, bslash = 0, structural = 0;
    for (uint c = 0; c < 4; ++c) {
        uint4 v = input[gid * 4 + c];
        for (uint lane = 0; lane < 4; ++lane) {
            uint w = v[lane];
            for (uint k = 0; k < 4; ++k) {
                uint b = (w >> (k * 8u)) & 0xffu;
                uint bit = c * 16u + lane * 4u + k;
                quote      |= ulong(b == 0x22u) << bit;
                bslash     |= ulong(b == 0x5cu) << bit;
                bool s = (b == 0x7bu) || (b == 0x7du) || (b == 0x5bu) ||
                         (b == 0x5du) || (b == 0x3au) || (b == 0x2cu);
                structural |= ulong(s) << bit;
            }
        }
    }

    const ulong even_bits = 0x5555555555555555UL;
    ulong acc = 0;
    ulong prev_escaped = ulong(gid & 1u);
    for (uint r = 0; r < p.rounds; ++r) {
        // simdjson find_escaped on a round-varying backslash mask.
        ulong bs = bslash ^ acc;
        ulong b = bs & ~prev_escaped;
        ulong follows_escape = (b << 1) | prev_escaped;
        ulong odd_starts = b & ~even_bits & ~follows_escape;
        ulong sum = odd_starts + b;            // 64-bit add ...
        ulong carry = ulong(sum < odd_starts); // ... with carry-out
        ulong invert = sum << 1;
        ulong escaped = (even_bits ^ invert) & follows_escape;

        ulong quote_real = quote & ~escaped;
        ulong in_string = prefix_xor64(quote_real); // 6-step ladder
        ulong struct_live = structural & ~in_string;
        acc ^= escaped ^ in_string ^ struct_live ^ (struct_live << 1);
        prev_escaped = carry;
    }
    out[gid] = acc ^ quote ^ bslash ^ structural;
}

// ============================ variant 2: uint2 ============================
// Bitmaps as uint2: .x = bits 0..31, .y = bits 32..63 (little-endian ulong
// layout), explicit carry/shift plumbing across the 32-bit boundary.

// 64-bit logical shift left by a compile-time constant 1..31.
inline uint2 shl64(uint2 x, uint k) {
    return uint2(x.x << k, (x.y << k) | (x.x >> (32u - k)));
}

inline uint2 prefix_xor64_u2(uint2 x) {
    x ^= shl64(x, 1);
    x ^= shl64(x, 2);
    x ^= shl64(x, 4);
    x ^= shl64(x, 8);
    x ^= shl64(x, 16);
    x.y ^= x.x; // the `x ^= x << 32` step
    return x;
}

kernel void spike_bitmap_uint2(
    device const uint4 *input    [[buffer(0)]],
    device uint2 *out            [[buffer(1)]],
    constant SpikeParams &p      [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]])
{
    if (gid >= p.chunk_count) { return; }

    uint2 quote = uint2(0u), bslash = uint2(0u), structural = uint2(0u);
    for (uint c = 0; c < 2; ++c) { // bytes 0..31 -> lo word
        uint4 v = input[gid * 4 + c];
        for (uint lane = 0; lane < 4; ++lane) {
            uint w = v[lane];
            for (uint k = 0; k < 4; ++k) {
                uint b = (w >> (k * 8u)) & 0xffu;
                uint bit = c * 16u + lane * 4u + k;
                quote.x      |= uint(b == 0x22u) << bit;
                bslash.x     |= uint(b == 0x5cu) << bit;
                bool s = (b == 0x7bu) || (b == 0x7du) || (b == 0x5bu) ||
                         (b == 0x5du) || (b == 0x3au) || (b == 0x2cu);
                structural.x |= uint(s) << bit;
            }
        }
    }
    for (uint c = 2; c < 4; ++c) { // bytes 32..63 -> hi word
        uint4 v = input[gid * 4 + c];
        for (uint lane = 0; lane < 4; ++lane) {
            uint w = v[lane];
            for (uint k = 0; k < 4; ++k) {
                uint b = (w >> (k * 8u)) & 0xffu;
                uint bit = c * 16u + lane * 4u + k - 32u;
                quote.y      |= uint(b == 0x22u) << bit;
                bslash.y     |= uint(b == 0x5cu) << bit;
                bool s = (b == 0x7bu) || (b == 0x7du) || (b == 0x5bu) ||
                         (b == 0x5du) || (b == 0x3au) || (b == 0x2cu);
                structural.y |= uint(s) << bit;
            }
        }
    }

    const uint2 even_bits = uint2(0x55555555u, 0x55555555u);
    uint2 acc = uint2(0u);
    uint2 prev_escaped = uint2(gid & 1u, 0u);
    for (uint r = 0; r < p.rounds; ++r) {
        uint2 bs = bslash ^ acc;
        uint2 b = bs & ~prev_escaped;
        uint2 follows_escape = shl64(b, 1) | prev_escaped;
        uint2 odd_starts = b & ~even_bits & ~follows_escape;
        // 64-bit add with explicit carry across the 32-bit boundary.
        uint sum_lo  = odd_starts.x + b.x;
        uint c0      = uint(sum_lo < odd_starts.x);
        uint sum_hi1 = odd_starts.y + b.y;
        uint c1      = uint(sum_hi1 < odd_starts.y);
        uint sum_hi  = sum_hi1 + c0;
        uint c2      = uint(sum_hi < sum_hi1);
        uint2 sum    = uint2(sum_lo, sum_hi);
        uint carry   = c1 | c2;
        uint2 invert = shl64(sum, 1);
        uint2 escaped = (even_bits ^ invert) & follows_escape;

        uint2 quote_real = quote & ~escaped;
        uint2 in_string = prefix_xor64_u2(quote_real);
        uint2 struct_live = structural & ~in_string;
        acc ^= escaped ^ in_string ^ struct_live ^ shl64(struct_live, 1);
        prev_escaped = uint2(carry, 0u);
    }
    out[gid] = acc ^ quote ^ bslash ^ structural;
}
"#;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Scalar CPU model of the kernels (64-bit semantics, mirrors variant 1).
fn cpu_reference(chunk: &[u8], gid: u32, rounds: u32) -> u64 {
    let (mut quote, mut bslash, mut structural) = (0u64, 0u64, 0u64);
    for (i, &b) in chunk.iter().enumerate() {
        quote |= u64::from(b == 0x22) << i;
        bslash |= u64::from(b == 0x5c) << i;
        let s = matches!(b, 0x7b | 0x7d | 0x5b | 0x5d | 0x3a | 0x2c);
        structural |= u64::from(s) << i;
    }
    fn prefix_xor(mut x: u64) -> u64 {
        x ^= x << 1;
        x ^= x << 2;
        x ^= x << 4;
        x ^= x << 8;
        x ^= x << 16;
        x ^= x << 32;
        x
    }
    const EVEN: u64 = 0x5555_5555_5555_5555;
    let mut acc = 0u64;
    let mut prev_escaped = u64::from(gid & 1);
    for _ in 0..rounds {
        let bs = bslash ^ acc;
        let b = bs & !prev_escaped;
        let follows_escape = (b << 1) | prev_escaped;
        let odd_starts = b & !EVEN & !follows_escape;
        let (sum, carry) = odd_starts.overflowing_add(b);
        let invert = sum << 1;
        let escaped = (EVEN ^ invert) & follows_escape;
        let quote_real = quote & !escaped;
        let in_string = prefix_xor(quote_real);
        let struct_live = structural & !in_string;
        acc ^= escaped ^ in_string ^ struct_live ^ (struct_live << 1);
        prev_escaped = u64::from(carry);
    }
    acc ^ quote ^ bslash ^ structural
}

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

fn make_pso(device: &Device, library_msl: &str, name: &str) -> Pso {
    let source = NSString::from_str(library_msl);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .unwrap_or_else(|e| panic!("MSL compile failed: {}", e.localizedDescription()));
    let function = library
        .newFunctionWithName(&NSString::from_str(name))
        .unwrap_or_else(|| panic!("kernel `{name}` not found"));
    device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|e| panic!("PSO `{name}` failed: {}", e.localizedDescription()))
}

fn alloc(device: &Device, len: usize) -> Buffer {
    device
        .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
        .unwrap_or_else(|| panic!("buffer alloc of {len} bytes failed"))
}

/// One dispatch; returns (seconds, used_gpu_timestamps).
fn run_once(queue: &Queue, pso: &Pso, input: &Buffer, out: &Buffer, params: SpikeParams) -> (f64, bool) {
    let cmd = queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    enc.setComputePipelineState(pso);
    // SAFETY: buffers outlive this synchronous dispatch; offsets in bounds.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 0);
        enc.setBuffer_offset_atIndex(Some(out), 0, 1);
        enc.setBytes_length_atIndex(
            NonNull::from(&params).cast::<c_void>(),
            size_of::<SpikeParams>(),
            2,
        );
    }
    let grid = MTLSize { width: CHUNKS, height: 1, depth: 1 };
    let group = MTLSize { width: THREADGROUP, height: 1, depth: 1 };
    enc.dispatchThreads_threadsPerThreadgroup(grid, group);
    enc.endEncoding();

    let host_start = Instant::now();
    cmd.commit();
    cmd.waitUntilCompleted();
    let host_secs = host_start.elapsed().as_secs_f64();

    assert_ne!(
        cmd.status(),
        MTLCommandBufferStatus::Error,
        "command buffer error: {:?}",
        cmd.error().map(|e| e.localizedDescription().to_string())
    );

    let gpu_secs = cmd.GPUEndTime() - cmd.GPUStartTime();
    if gpu_secs > 0.0 { (gpu_secs, true) } else { (host_secs, false) }
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn out_words(buf: &Buffer) -> &[u64] {
    // SAFETY: shared-storage buffer, page-aligned contents, no GPU work in
    // flight when this is called; OUT_BYTES is the allocation size.
    unsafe { core::slice::from_raw_parts(buf.contents().cast::<u64>().as_ptr(), OUT_BYTES / 8) }
}

fn main() {
    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let queue = device.newCommandQueue().expect("no command queue");
    println!(
        "spike_wordsize: {CHUNKS} chunks x 64 B = {:.2} GiB input, tg={THREADGROUP}, \
         warmup={WARMUP}, timed_runs={RUNS} (median)",
        INPUT_BYTES as f64 / (1u64 << 30) as f64
    );
    println!("device: {}", device.name());

    let pso_ulong = make_pso(&device, MSL, "spike_bitmap_ulong");
    let pso_uint2 = make_pso(&device, MSL, "spike_bitmap_uint2");

    let input = alloc(&device, INPUT_BYTES);
    // Fill input in place with deterministic pseudo-random bytes.
    {
        // SAFETY: fresh shared buffer, CPU-exclusive access at this point.
        let words = unsafe {
            core::slice::from_raw_parts_mut(input.contents().cast::<u64>().as_ptr(), INPUT_BYTES / 8)
        };
        let mut state = 0x243F_6A88_85A3_08D3u64;
        for w in words.iter_mut() {
            *w = splitmix64(&mut state);
        }
    }
    let input_bytes_view =
        // SAFETY: same buffer, read-only view for the CPU reference check.
        unsafe { core::slice::from_raw_parts(input.contents().cast::<u8>().as_ptr(), INPUT_BYTES) };

    let out_ulong = alloc(&device, OUT_BYTES);
    let out_uint2 = alloc(&device, OUT_BYTES);

    for rounds in [1u32, 8u32] {
        let params = SpikeParams { chunk_count: CHUNKS as u32, rounds };
        let label = if rounds == 1 {
            "rounds=1 (representative K1 mix)"
        } else {
            "rounds=8 (ALU-amplified)"
        };
        println!("\n{label}");

        let mut results: Vec<(&str, f64)> = Vec::new();
        for (name, pso, out) in [
            ("ulong", &pso_ulong, &out_ulong),
            ("uint2", &pso_uint2, &out_uint2),
        ] {
            for _ in 0..WARMUP {
                run_once(&queue, pso, &input, out, params);
            }
            let mut times = Vec::with_capacity(RUNS);
            let mut all_gpu_timed = true;
            for _ in 0..RUNS {
                let (secs, gpu_timed) = run_once(&queue, pso, &input, out, params);
                all_gpu_timed &= gpu_timed;
                times.push(secs);
            }
            let med = median(&mut times);
            let gbps_in = INPUT_BYTES as f64 / med / 1e9;
            let gbps_total = (INPUT_BYTES + OUT_BYTES) as f64 / med / 1e9;
            println!(
                "  {name:>5}: median {:8.3} ms  ->  {gbps_in:7.1} GB/s input  \
                 ({gbps_total:.1} GB/s incl. {} MiB out){}",
                med * 1e3,
                OUT_BYTES >> 20,
                if all_gpu_timed { "" } else { "  [host-timed fallback]" },
            );
            results.push((name, gbps_in));
        }

        // Correctness: variant outputs must match bit-for-bit ...
        let a = out_words(&out_ulong);
        let b = out_words(&out_uint2);
        assert_eq!(a, b, "rounds={rounds}: ulong and uint2 outputs differ");
        let checksum = a.iter().fold(0u64, |acc, &w| acc ^ w);
        // ... and match the scalar CPU model on sampled chunks.
        for gid in [0usize, 1, 255, 65_537, 12_345_678, CHUNKS - 1] {
            let expect = cpu_reference(&input_bytes_view[gid * 64..gid * 64 + 64], gid as u32, rounds);
            assert_eq!(a[gid], expect, "rounds={rounds}: CPU model mismatch at chunk {gid}");
        }
        println!("  outputs match (xor checksum {checksum:#018x}, CPU model spot-checks ok)");

        let ratio = results[1].1 / results[0].1;
        println!("  uint2 / ulong throughput ratio: {ratio:.3}x");
    }
}
