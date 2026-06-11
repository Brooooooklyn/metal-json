//! M0 SPIKE A — 64x64->128-bit multiplication strategy for the Eisel-Lemire
//! f64 kernel (M4).
//!
//! Questions answered on real hardware:
//!   1. Does `mulhi(ulong, ulong)` compile in MSL (runtime `newLibraryWithSource`)?
//!   2. Throughput of full 128-bit products (hi+lo) via `mulhi(ulong)` vs a
//!      pure 4-limb 32-bit `mul`/`mulhi` formulation (and a mixed
//!      `(ulong)uint*uint` portable formulation as a third data point).
//!   3. Do `ulong` add/shift/popcount look natively fine (vs a `uint` chain)?
//!
//! Methodology: 64M ulong elements per input array (512 MiB each, 1.5 GiB
//! traffic per dispatch), kernels run R dependent rounds of a full 128-bit
//! multiply with hi mixed into the next operands (defeats DCE, forces the
//! real dependency chain). R=8 and R=32 per variant; the delta isolates pure
//! ALU cost from memory traffic. 3 warmup runs, median of 7 timed runs using
//! command-buffer GPUStartTime/GPUEndTime (host-timing fallback). Every
//! kernel's output is verified against a CPU u128 oracle on a prefix.
//!
//! Run: `cargo run --release --example spike_mulhi`
//!
//! This spike deliberately holds its own MSL source and drives objc2-metal
//! directly (the crate wrappers stay untouched; the embedded-library
//! `MetalContext` path is not what we are measuring).

use std::time::Instant;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResourceOptions, MTLSize,
};

/// Elements per input array. 1<<26 = 67,108,864 ulongs = 512 MiB per array;
/// a + b + out = 1.5 GiB of traffic per dispatch.
const N: usize = 1 << 26;
const WARMUP: usize = 3;
const TIMED: usize = 7;
const THREADGROUP: usize = 256;
/// Prefix of each output buffer checked against the CPU oracle.
const VERIFY: usize = 65_536;
/// Bytes moved per element per dispatch (read a, read b, write out).
const TRAFFIC_PER_ELEM: f64 = 24.0;

// ---------------------------------------------------------------------------
// MSL source
// ---------------------------------------------------------------------------

/// Tiny standalone probe: does `mulhi(ulong, ulong)` even compile at runtime?
const PROBE_MULHI_ULONG: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void probe_mulhi_ulong(device const ulong* a [[buffer(0)]],
                              device const ulong* b [[buffer(1)]],
                              device ulong* out     [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    out[gid] = mulhi(a[gid], b[gid]) ^ (a[gid] * b[gid]);
}
"#;

/// Full 128-bit product via the 64-bit builtin: 1x mulhi(ulong) + 1x ulong mul.
const BODY_MULHI64: &str = r#"
        ulong hi = mulhi(x, y);
        ulong lo = x * y;"#;

/// Pure 4-limb 32-bit formulation: 4x mul(uint) + 4x mulhi(uint) + carry adds.
/// No 64-bit multiplies at all; ulong only for the final hi/lo packing (which
/// the chain consumes, as the Eisel-Lemire kernel would).
const BODY_LIMBS32: &str = r#"
        uint x0 = (uint)x, x1 = (uint)(x >> 32);
        uint y0 = (uint)y, y1 = (uint)(y >> 32);
        uint p00l = x0 * y0, p00h = mulhi(x0, y0);
        uint p01l = x0 * y1, p01h = mulhi(x0, y1);
        uint p10l = x1 * y0, p10h = mulhi(x1, y0);
        uint p11l = x1 * y1, p11h = mulhi(x1, y1);
        uint t1 = p00h + p01l;  uint c1 = (uint)(t1 < p01l);
        uint t2 = t1 + p10l;    uint c2 = (uint)(t2 < p10l);
        uint cm = c1 + c2;
        uint t3 = p01h + p10h;  uint c3 = (uint)(t3 < p10h);
        uint t4 = t3 + p11l;    uint c4 = (uint)(t4 < p11l);
        uint t5 = t4 + cm;      uint c5 = (uint)(t5 < cm);
        uint hh = p11h + c3 + c4 + c5;
        ulong lo = ((ulong)t2 << 32) | p00l;
        ulong hi = ((ulong)hh << 32) | t5;"#;

/// Mixed/portable formulation (simdjson's umul128 fallback): 4x
/// (ulong)uint*uint partial products accumulated with 64-bit adds/shifts.
const BODY_LIMBS64: &str = r#"
        uint x0 = (uint)x, x1 = (uint)(x >> 32);
        uint y0 = (uint)y, y1 = (uint)(y >> 32);
        ulong p00 = (ulong)x0 * y0;
        ulong p01 = (ulong)x0 * y1;
        ulong p10 = (ulong)x1 * y0;
        ulong p11 = (ulong)x1 * y1;
        ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
        ulong lo = (mid << 32) | (uint)p00;
        ulong hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);"#;

/// ulong add/shift/xor/popcount dependency chain (32 rounds), and the same
/// chain on uint for comparison (dispatched over 2N elements => identical
/// memory traffic).
const OPS_KERNELS: &str = r#"
kernel void ops_chain64(device const ulong* a [[buffer(0)]],
                        device const ulong* b [[buffer(1)]],
                        device ulong* out     [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    ulong x = a[gid];
    ulong y = b[gid];
    for (uint r = 0; r < 32u; ++r) {
        x = (x << 13) | (x >> 51);
        x += y;
        y ^= (ulong)popcount(x);
        y = (y << 7) | (y >> 57);
    }
    out[gid] = x ^ y;
}

kernel void ops_chain32(device const uint* a [[buffer(0)]],
                        device const uint* b [[buffer(1)]],
                        device uint* out     [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    uint x = a[gid];
    uint y = b[gid];
    for (uint r = 0; r < 32u; ++r) {
        x = (x << 13) | (x >> 19);
        x += y;
        y ^= (uint)popcount(x);
        y = (y << 7) | (y >> 25);
    }
    out[gid] = x ^ y;
}
"#;

/// A multiply-chain kernel: R dependent rounds of full 128-bit multiply with
/// hi mixed into both next operands. Identical chain semantics across all
/// variants, so one CPU oracle verifies them all.
fn mul_kernel(name: &str, rounds: u32, body: &str) -> String {
    format!(
        r#"
kernel void {name}(device const ulong* a [[buffer(0)]],
                   device const ulong* b [[buffer(1)]],
                   device ulong* out     [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {{
    ulong x = a[gid];
    ulong y = b[gid] | 1;
    for (uint r = 0; r < {rounds}u; ++r) {{{body}
        x = lo ^ hi;
        y += (hi | 1);
    }}
    out[gid] = x ^ y;
}}
"#
    )
}

// ---------------------------------------------------------------------------
// CPU oracles
// ---------------------------------------------------------------------------

fn cpu_mul_chain(a: u64, b: u64, rounds: u32) -> u64 {
    let mut x = a;
    let mut y = b | 1;
    for _ in 0..rounds {
        let p = (x as u128).wrapping_mul(y as u128);
        let lo = p as u64;
        let hi = (p >> 64) as u64;
        x = lo ^ hi;
        y = y.wrapping_add(hi | 1);
    }
    x ^ y
}

fn cpu_ops_chain64(a: u64, b: u64) -> u64 {
    let mut x = a;
    let mut y = b;
    for _ in 0..32 {
        x = x.rotate_left(13);
        x = x.wrapping_add(y);
        y ^= x.count_ones() as u64;
        y = y.rotate_left(7);
    }
    x ^ y
}

fn cpu_ops_chain32(a: u32, b: u32) -> u32 {
    let mut x = a;
    let mut y = b;
    for _ in 0..32 {
        x = x.rotate_left(13);
        x = x.wrapping_add(y);
        y ^= x.count_ones();
        y = y.rotate_left(7);
    }
    x ^ y
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------
// Metal plumbing (self-contained; GPU timestamps need raw command buffers)
// ---------------------------------------------------------------------------

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

fn compile(device: &Device, source: &str) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    device
        .newLibraryWithSource_options_error(&NSString::from_str(source), None)
        .map_err(|e| e.localizedDescription().to_string())
}

fn pso(device: &Device, library: &ProtocolObject<dyn MTLLibrary>, name: &str) -> Pso {
    let f = library
        .newFunctionWithName(&NSString::from_str(name))
        .unwrap_or_else(|| panic!("kernel `{name}` not found"));
    device
        .newComputePipelineStateWithFunction_error(&f)
        .unwrap_or_else(|e| panic!("PSO `{name}`: {}", e.localizedDescription()))
}

fn alloc(device: &Device, bytes: usize) -> Buffer {
    device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("buffer alloc")
}

/// One synchronous dispatch; returns (seconds, used_gpu_timestamps).
fn run_once(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pso: &Pso,
    bufs: [&Buffer; 3],
    threads: usize,
) -> (f64, bool) {
    let cmd = queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("compute encoder");
    enc.setComputePipelineState(pso);
    for (i, buf) in bufs.iter().enumerate() {
        // SAFETY: buffers outlive this synchronous dispatch; offset 0 valid.
        unsafe { enc.setBuffer_offset_atIndex(Some(buf), 0, i) };
    }
    let grid = MTLSize { width: threads, height: 1, depth: 1 };
    let group = MTLSize { width: THREADGROUP, height: 1, depth: 1 };
    enc.dispatchThreads_threadsPerThreadgroup(grid, group);
    enc.endEncoding();

    let t0 = Instant::now();
    cmd.commit();
    cmd.waitUntilCompleted();
    let host_s = t0.elapsed().as_secs_f64();
    assert_ne!(
        cmd.status(),
        MTLCommandBufferStatus::Error,
        "command buffer error: {:?}",
        cmd.error().map(|e| e.localizedDescription().to_string())
    );
    let gpu_s = cmd.GPUEndTime() - cmd.GPUStartTime();
    if gpu_s > 0.0 { (gpu_s, true) } else { (host_s, false) }
}

/// Warm up, then return the median of TIMED runs (seconds).
fn bench(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pso: &Pso,
    bufs: [&Buffer; 3],
    threads: usize,
) -> (f64, bool) {
    for _ in 0..WARMUP {
        run_once(queue, pso, bufs, threads);
    }
    let mut times = Vec::with_capacity(TIMED);
    let mut all_gpu = true;
    for _ in 0..TIMED {
        let (t, used_gpu) = run_once(queue, pso, bufs, threads);
        all_gpu &= used_gpu;
        times.push(t);
    }
    times.sort_by(f64::total_cmp);
    (times[TIMED / 2], all_gpu)
}

fn buffer_as_slice<T>(buf: &Buffer, n: usize) -> &[T] {
    // SAFETY: shared-storage buffer of at least n * size_of::<T>() bytes; no
    // command buffer is in flight when we read (run_once always waits).
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr().cast::<T>(), n) }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let queue = device.newCommandQueue().expect("no command queue");
    println!("device: {}", device.name());
    println!(
        "elements: {N} ({} MiB per ulong array, {:.2} GiB traffic per dispatch)\n",
        N * 8 / (1024 * 1024),
        N as f64 * TRAFFIC_PER_ELEM / (1024.0 * 1024.0 * 1024.0)
    );

    // ---- Q1: does mulhi(ulong) compile at runtime? --------------------------
    let mulhi_ulong_ok = match compile(&device, PROBE_MULHI_ULONG) {
        Ok(_) => {
            println!("Q1: mulhi(ulong, ulong) runtime-compiles: YES");
            true
        }
        Err(e) => {
            println!("Q1: mulhi(ulong, ulong) runtime-compiles: NO\n    error: {e}");
            false
        }
    };

    // ---- Build the benchmark library ---------------------------------------
    // (name, rounds, body); mulhi64 variants only if Q1 passed.
    let mut variants: Vec<(&str, u32, &str)> = Vec::new();
    if mulhi_ulong_ok {
        variants.push(("mulhi64", 8, BODY_MULHI64));
        variants.push(("mulhi64", 32, BODY_MULHI64));
    }
    variants.push(("limbs32", 8, BODY_LIMBS32));
    variants.push(("limbs32", 32, BODY_LIMBS32));
    variants.push(("limbs64", 8, BODY_LIMBS64));
    variants.push(("limbs64", 32, BODY_LIMBS64));

    let mut source = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
    for (name, rounds, body) in &variants {
        source.push_str(&mul_kernel(&format!("mul_{name}_r{rounds}"), *rounds, body));
    }
    source.push_str(OPS_KERNELS);
    let library = compile(&device, &source).expect("benchmark library failed to compile");

    // ---- Buffers ------------------------------------------------------------
    let bytes = N * size_of::<u64>();
    let a = alloc(&device, bytes);
    let b = alloc(&device, bytes);
    let out = alloc(&device, bytes);
    {
        // SAFETY: no GPU work in flight; buffers are N u64s.
        let a_host = unsafe { std::slice::from_raw_parts_mut(a.contents().as_ptr().cast::<u64>(), N) };
        let b_host = unsafe { std::slice::from_raw_parts_mut(b.contents().as_ptr().cast::<u64>(), N) };
        let mut s1 = 0x0123_4567_89AB_CDEFu64;
        let mut s2 = 0xFEDC_BA98_7654_3210u64;
        for i in 0..N {
            a_host[i] = splitmix64(&mut s1);
            b_host[i] = splitmix64(&mut s2);
        }
    }
    let a_check: Vec<u64> = buffer_as_slice::<u64>(&a, VERIFY).to_vec();
    let b_check: Vec<u64> = buffer_as_slice::<u64>(&b, VERIFY).to_vec();

    // ---- Q2: multiply variants ----------------------------------------------
    println!("\nQ2: full 128-bit multiply throughput (median of {TIMED} runs, {WARMUP} warmups)");
    println!(
        "{:<10} {:>6} {:>12} {:>16} {:>14} {:>10}",
        "variant", "rounds", "median ms", "mul128/s", "elem/s", "GB/s"
    );
    // (name, rounds) -> median seconds
    let mut results: Vec<(&str, u32, f64)> = Vec::new();
    for (name, rounds, _) in &variants {
        let kernel = format!("mul_{name}_r{rounds}");
        let p = pso(&device, &library, &kernel);
        let (t, used_gpu) = bench(&queue, &p, [&a, &b, &out], N);

        // Verify against CPU oracle on a prefix.
        let got = buffer_as_slice::<u64>(&out, VERIFY);
        for i in 0..VERIFY {
            let want = cpu_mul_chain(a_check[i], b_check[i], *rounds);
            assert_eq!(got[i], want, "{kernel}: mismatch at element {i}");
        }

        let muls = N as f64 * *rounds as f64 / t;
        println!(
            "{:<10} {:>6} {:>12.3} {:>16.3e} {:>14.3e} {:>10.1}{}",
            name,
            rounds,
            t * 1e3,
            muls,
            N as f64 / t,
            N as f64 * TRAFFIC_PER_ELEM / t / 1e9,
            if used_gpu { "" } else { "  [host-timed]" }
        );
        results.push((name, *rounds, t));
    }

    // ALU-isolated cost: (t_r32 - t_r8) covers 24 extra rounds with identical
    // memory traffic.
    println!("\nALU-isolated 128-bit multiply rate (from r32 - r8 delta, memory traffic cancels):");
    let names: Vec<&str> = {
        let mut v: Vec<&str> = results.iter().map(|(n, _, _)| *n).collect();
        v.dedup();
        v
    };
    let mut alu_rates: Vec<(&str, f64)> = Vec::new();
    for name in names {
        let t8 = results.iter().find(|(n, r, _)| *n == name && *r == 8).unwrap().2;
        let t32 = results.iter().find(|(n, r, _)| *n == name && *r == 32).unwrap().2;
        let dt = t32 - t8;
        if dt > 0.0 {
            let rate = N as f64 * 24.0 / dt;
            println!("  {name:<10} {rate:.3e} mul128/s  ({:.3} ns per mul128 per-thread-equivalent)", dt / (N as f64 * 24.0) * 1e9);
            alu_rates.push((name, rate));
        } else {
            println!("  {name:<10} r32 not slower than r8; memory-bound at both round counts");
        }
    }
    if let Some(best) = alu_rates.iter().max_by(|x, y| x.1.total_cmp(&y.1)) {
        println!("  winner (ALU): {}", best.0);
    }

    // ---- Q3: ulong vs uint add/shift/xor/popcount chains ---------------------
    println!("\nQ3: ulong vs uint add/shift/xor/popcount chain (32 dependent rounds each)");
    let p64 = pso(&device, &library, "ops_chain64");
    let (t64, g64) = bench(&queue, &p64, [&a, &b, &out], N);
    {
        let got = buffer_as_slice::<u64>(&out, VERIFY);
        for i in 0..VERIFY {
            assert_eq!(got[i], cpu_ops_chain64(a_check[i], b_check[i]), "ops_chain64 mismatch at {i}");
        }
    }
    let p32 = pso(&device, &library, "ops_chain32");
    // Same buffers viewed as uint => 2N elements, identical byte traffic.
    let (t32, g32) = bench(&queue, &p32, [&a, &b, &out], 2 * N);
    {
        let got = buffer_as_slice::<u32>(&out, 2 * VERIFY);
        let a32 = buffer_as_slice::<u32>(&a, 2 * VERIFY);
        let b32 = buffer_as_slice::<u32>(&b, 2 * VERIFY);
        for i in 0..2 * VERIFY {
            assert_eq!(got[i], cpu_ops_chain32(a32[i], b32[i]), "ops_chain32 mismatch at {i}");
        }
    }
    let r64 = N as f64 * 32.0 / t64;
    let r32 = 2.0 * N as f64 * 32.0 / t32;
    println!(
        "  ops_chain64: {:>8.3} ms  {:.3e} rounds/s{}",
        t64 * 1e3, r64, if g64 { "" } else { "  [host-timed]" }
    );
    println!(
        "  ops_chain32: {:>8.3} ms  {:.3e} rounds/s  ({} elements){}",
        t32 * 1e3, r32, 2 * N, if g32 { "" } else { "  [host-timed]" }
    );
    println!(
        "  u64 round throughput = {:.2}x u32 (0.5x = clean 2x32 lowering, i.e. natively fine)",
        r64 / r32
    );
}
